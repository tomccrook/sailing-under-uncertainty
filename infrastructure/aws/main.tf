locals {
  enabled = var.create_resources ? 1 : 0
  compute = var.create_resources && var.create_compute ? 1 : 0
  public  = var.create_resources && var.public_hosting ? 1 : 0
  buckets = var.create_resources ? toset(var.public_hosting ? ["research", "public"] : ["research"]) : toset([])
}

resource "aws_s3_bucket" "artifacts" {
  for_each      = local.buckets
  bucket        = "${var.name}-${var.account_id}-${each.key}"
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  for_each                = local.buckets
  bucket                  = aws_s3_bucket.artifacts[each.key].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "artifacts" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.artifacts[each.key].id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.artifacts[each.key].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.artifacts[each.key].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "research" {
  count  = local.enabled
  bucket = aws_s3_bucket.artifacts["research"].id
  rule {
    id     = "temporary-only"
    status = "Enabled"
    filter {
      prefix = "tmp/"
    }
    expiration {
      days = 7
    }
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
  depends_on = [aws_s3_bucket_versioning.artifacts]
}

resource "aws_s3_bucket_policy" "research" {
  count  = local.enabled
  bucket = aws_s3_bucket.artifacts["research"].id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Sid      = "TLSOnly", Effect = "Deny", Principal = "*", Action = "s3:*",
      Resource = [aws_s3_bucket.artifacts["research"].arn, "${aws_s3_bucket.artifacts["research"].arn}/*"],
      Condition = {
        Bool = {
          "aws:SecureTransport" = "false"
        }
      }
      }
    ]
    }
  )
}

resource "aws_ecr_repository" "cpu" {
  count                = local.enabled
  name                 = "${var.name}-cpu"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "cpu" {
  count      = local.enabled
  repository = aws_ecr_repository.cpu[0].name
  policy = jsonencode({
    rules = [{
      rulePriority = 1, description = "Expire untagged build images after seven days",
      selection = {
        tagStatus = "untagged", countType = "sinceImagePushed", countUnit = "days", countNumber = 7
      },
      action = {
        type = "expire"
      }
      }
    ]
    }
  )
}

resource "aws_cloudwatch_log_group" "jobs" {
  count             = local.compute
  name              = "/${var.name}/cpu-batch"
  retention_in_days = 7
}

resource "aws_iam_role" "execution" {
  count = local.compute
  name  = "${var.name}-task-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }, Action = "sts:AssumeRole"
      }
    ]
    }
  )
}

resource "aws_iam_role_policy" "execution" {
  count = local.compute
  role  = aws_iam_role.execution[0].id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      {
        Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*"
      },
      {
        Effect = "Allow", Action = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"], Resource = aws_ecr_repository.cpu[0].arn
      },
      {
        Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.jobs[0].arn}:*"
      }
    ]
    }
  )
}

resource "aws_iam_role" "job" {
  count              = local.compute
  name               = "${var.name}-research-job"
  assume_role_policy = aws_iam_role.execution[0].assume_role_policy
}

resource "aws_iam_role_policy" "job" {
  count = local.compute
  role  = aws_iam_role.job[0].id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      {
        Effect = "Allow", Action = ["s3:GetObject"], Resource = ["${aws_s3_bucket.artifacts["research"].arn}/jobs/specs/*", "${aws_s3_bucket.artifacts["research"].arn}/jobs/results/*", "${aws_s3_bucket.artifacts["research"].arn}/artifacts/sha256/*"]
      },
      {
        Effect = "Allow", Action = ["s3:PutObject"], Resource = ["${aws_s3_bucket.artifacts["research"].arn}/jobs/results/*", "${aws_s3_bucket.artifacts["research"].arn}/artifacts/sha256/*"]
      }
    ]
    }
  )
}

resource "aws_batch_compute_environment" "cpu" {
  count = local.compute
  name  = "${var.name}-cpu"
  type  = "MANAGED"
  state = "ENABLED"
  compute_resources {
    type               = "FARGATE"
    max_vcpus          = var.max_vcpus
    subnets            = var.subnet_ids
    security_group_ids = var.security_group_ids
  }
}

resource "aws_batch_job_queue" "cpu" {
  count    = local.compute
  name     = "${var.name}-internal-cpu"
  state    = var.queue_enabled ? "ENABLED" : "DISABLED"
  priority = 1
  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.cpu[0].arn
  }
}

resource "aws_batch_job_definition" "cpu" {
  count                 = local.compute
  name                  = "${var.name}-cpu-worker"
  type                  = "container"
  platform_capabilities = ["FARGATE"]
  timeout {
    attempt_duration_seconds = 720
  }
  retry_strategy {
    attempts = 2
    evaluate_on_exit {
      action       = "RETRY"
      on_exit_code = "75"
    }
    evaluate_on_exit {
      action           = "RETRY"
      on_status_reason = "Task failed to start*"
    }
    evaluate_on_exit {
      action    = "EXIT"
      on_reason = "*"
    }
  }
  container_properties = jsonencode({
    image            = var.image, jobRoleArn = aws_iam_role.job[0].arn,
    executionRoleArn = aws_iam_role.execution[0].arn,
    resourceRequirements = [{
      type = "VCPU", value = "2"
      }, {
      type = "MEMORY", value = "8192"
      }
    ],
    environment = [{ name = "AWS_DEFAULT_REGION", value = var.region }],
    runtimePlatform = {
      cpuArchitecture = "X86_64", operatingSystemFamily = "LINUX"
    },
    networkConfiguration = {
      assignPublicIp = "ENABLED"
    },
    fargatePlatformConfiguration = {
      platformVersion = "LATEST"
    },
    logConfiguration = {
      logDriver = "awslogs", options = {
        "awslogs-group"  = aws_cloudwatch_log_group.jobs[0].name,
        "awslogs-region" = var.region, "awslogs-stream-prefix" = "research"
      }
    }
    }
  )
  depends_on = [aws_iam_role_policy.execution, aws_iam_role_policy.job]
}

output "artifact_bucket" {
  value = var.create_resources ? aws_s3_bucket.artifacts["research"].id : null
}

output "queue_arn" {
  value = local.compute == 1 ? aws_batch_job_queue.cpu[0].arn : null
}

output "job_definition_arn" {
  value = local.compute == 1 ? aws_batch_job_definition.cpu[0].arn : null
}

output "public_url" {
  value = local.public == 1 ? "https://${aws_cloudfront_distribution.public[0].domain_name}" : null
}
