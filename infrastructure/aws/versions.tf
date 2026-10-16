terraform {
  required_version = ">= 1.14.0, < 1.15.0"
  required_providers {
    aws = {
      source = "hashicorp/aws", version = "= 6.15.0"
    }
  }
}

provider "aws" {
  region                      = var.region
  skip_metadata_api_check     = true
  skip_credentials_validation = !var.create_resources
  skip_requesting_account_id  = !var.create_resources
  access_key                  = var.create_resources ? null : "offline-validation"
  secret_key                  = var.create_resources ? null : "offline-validation"
  default_tags {
    tags = {
      Owner = "Tom Crook", Project = "Sail Lab", ManagedBy = "Terraform"
    }
  }
}
