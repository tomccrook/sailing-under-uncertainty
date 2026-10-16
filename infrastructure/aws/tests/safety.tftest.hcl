mock_provider "aws" {
  mock_resource "aws_batch_compute_environment" {
    defaults = { arn = "arn:aws:batch:us-east-1:111122223333:compute-environment/mock" }
  }
  mock_resource "aws_iam_role" {
    defaults = { arn = "arn:aws:iam::111122223333:role/mock" }
  }
  mock_resource "aws_ecr_repository" {
    defaults = { arn = "arn:aws:ecr:us-east-1:111122223333:repository/mock" }
  }
  mock_resource "aws_s3_bucket" {
    defaults = { arn = "arn:aws:s3:::sail-mock", bucket_regional_domain_name = "sail-mock.s3.us-east-1.amazonaws.com" }
  }
  mock_resource "aws_cloudfront_distribution" {
    defaults = { arn = "arn:aws:cloudfront::111122223333:distribution/mock" }
  }
}

run "zero_spend_defaults" {
  command = plan
  assert {
    condition     = length(aws_s3_bucket.artifacts) == 0 && length(aws_batch_compute_environment.cpu) == 0 && length(aws_cloudfront_distribution.public) == 0
    error_message = "Defaults must plan no paid resources."
  }
}

run "unbudgeted_provisioning_rejected" {
  command = plan
  variables { create_resources = true }
  expect_failures = [var.create_resources]
}

run "mock_bounded_cpu" {
  command = apply
  variables {
    create_resources        = true
    create_compute          = true
    authorised_budget_usd   = 100
    authorisation_reference = "mock test; no actual approval or AWS account"
    approved_cost_sha256    = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    account_id              = "111122223333"
    subnet_ids              = ["subnet-0123456789abcdef0"]
    security_group_ids      = ["sg-0123456789abcdef0"]
    image                   = "111122223333.dkr.ecr.us-east-1.amazonaws.com/sail-lab-cpu@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  }
  assert {
    condition     = aws_batch_compute_environment.cpu[0].compute_resources[0].max_vcpus == 4 && aws_batch_compute_environment.cpu[0].compute_resources[0].type == "FARGATE" && aws_batch_job_queue.cpu[0].state == "DISABLED"
    error_message = "CPU capacity must be bounded and queue initially disabled."
  }
  assert {
    condition     = jsondecode(aws_batch_job_definition.cpu[0].container_properties).resourceRequirements[0].value == "2" && aws_batch_job_definition.cpu[0].timeout[0].attempt_duration_seconds == 720 && aws_batch_job_definition.cpu[0].retry_strategy[0].attempts == 2
    error_message = "Tasks, retries and timeouts must match the cost plan."
  }
  assert {
    condition     = length(aws_s3_bucket.artifacts) == 1 && length(aws_cloudfront_distribution.public) == 0 && aws_s3_bucket_public_access_block.artifacts["research"].block_public_policy
    error_message = "Research artifacts must be private and not shared with public hosting."
  }
}

run "unbudgeted_hosting_rejected" {
  command = plan
  variables { public_hosting = true }
  expect_failures = [var.public_hosting]
}

run "mock_separate_static_hosting" {
  command = apply
  variables {
    create_resources             = true
    create_compute               = true
    authorised_budget_usd        = 100
    authorisation_reference      = "mock test; no actual approval or AWS account"
    approved_cost_sha256         = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    account_id                   = "111122223333"
    subnet_ids                   = ["subnet-0123456789abcdef0"]
    security_group_ids           = ["sg-0123456789abcdef0"]
    image                        = "111122223333.dkr.ecr.us-east-1.amazonaws.com/sail-lab-cpu@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    public_hosting               = true
    public_hosting_budget_usd    = 100
    approved_hosting_cost_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  }
  assert {
    condition     = length(aws_s3_bucket.artifacts) == 2 && length(aws_cloudfront_origin_access_control.public) == 1 && aws_cloudfront_distribution.public[0].default_cache_behavior[0].allowed_methods == toset(["GET", "HEAD", "OPTIONS"])
    error_message = "Public deployment must use a separate private REST origin with static methods."
  }
}

run "mock_storage_bootstrap" {
  command = plan
  variables {
    create_resources        = true
    authorised_budget_usd   = 100
    authorisation_reference = "mock only; no external approval"
    approved_cost_sha256    = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    account_id              = "111122223333"
  }
  assert {
    condition     = length(aws_ecr_repository.cpu) == 1 && length(aws_s3_bucket.artifacts) == 1 && length(aws_batch_compute_environment.cpu) == 0
    error_message = "Storage can bootstrap before the private image exists; no compute is started."
  }
}
