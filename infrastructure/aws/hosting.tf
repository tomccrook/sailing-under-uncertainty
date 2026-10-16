resource "aws_cloudfront_origin_access_control" "public" {
  count                             = local.public
  name                              = "${var.name}-public-oac"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_distribution" "public" {
  count               = local.public
  enabled             = true
  default_root_object = "index.html"
  price_class         = "PriceClass_100"
  origin {
    domain_name              = aws_s3_bucket.artifacts["public"].bucket_regional_domain_name
    origin_id                = "static-private-rest"
    origin_access_control_id = aws_cloudfront_origin_access_control.public[0].id
    s3_origin_config {
      origin_access_identity = ""
    }
  }
  default_cache_behavior {
    target_origin_id       = "static-private-rest"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
  }
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

resource "aws_s3_bucket_policy" "public" {
  count  = local.public
  bucket = aws_s3_bucket.artifacts["public"].id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      {
        Sid = "CloudFrontOnly", Effect = "Allow", Principal = {
          Service = "cloudfront.amazonaws.com"
        },
        Action = "s3:GetObject", Resource = "${aws_s3_bucket.artifacts["public"].arn}/*",
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = aws_cloudfront_distribution.public[0].arn
          }
        }
      },
      {
        Sid      = "TLSOnly", Effect = "Deny", Principal = "*", Action = "s3:*",
        Resource = [aws_s3_bucket.artifacts["public"].arn, "${aws_s3_bucket.artifacts["public"].arn}/*"],
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
