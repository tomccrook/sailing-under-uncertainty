variable "create_resources" {
  type    = bool
  default = false
  validation {
    condition = !var.create_resources || (
      var.authorised_budget_usd > 0 && length(var.authorisation_reference) > 0 &&
      can(regex("^[a-f0-9]{64}$", var.approved_cost_sha256)) &&
      var.account_id != "000000000000"
    )
    error_message = "Provisioning needs an explicit budget/reference/cost hash and account."
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "account_id" {
  type    = string
  default = "000000000000"
  validation {
    condition     = can(regex("^[0-9]{12}$", var.account_id))
    error_message = "Account must have 12 digits."
  }
}

variable "authorised_budget_usd" {
  type    = number
  default = 0
}

variable "authorisation_reference" {
  type    = string
  default = ""
}

variable "approved_cost_sha256" {
  type    = string
  default = ""
}

variable "name" {
  type    = string
  default = "sail-lab"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,25}$", var.name))
    error_message = "Use a short lowercase resource name."
  }
}

variable "subnet_ids" {
  type    = list(string)
  default = []
}

variable "security_group_ids" {
  type    = list(string)
  default = []
}

variable "image" {
  type    = string
  default = "not-configured"
}

variable "queue_enabled" {
  type    = bool
  default = false
}

variable "max_vcpus" {
  type    = number
  default = 4
  validation {
    condition     = var.max_vcpus == 4
    error_message = "Initial capacity is fixed at four vCPUs (two two-vCPU tasks)."
  }
}

variable "public_hosting" {
  type    = bool
  default = false
  validation {
    condition = !var.public_hosting || (
      var.create_resources && var.public_hosting_budget_usd > 0 &&
      can(regex("^[a-f0-9]{64}$", var.approved_hosting_cost_sha256))
    )
    error_message = "Public hosting requires a separate explicit hosting budget and approved traffic/storage cost plan."
  }
}

variable "public_hosting_budget_usd" {
  type    = number
  default = 0
}

variable "approved_hosting_cost_sha256" {
  type    = string
  default = ""
}

variable "create_compute" {
  type    = bool
  default = false
  validation {
    condition = !var.create_compute || (
      var.create_resources && length(var.subnet_ids) > 0 && length(var.security_group_ids) > 0 &&
      can(regex("@sha256:[a-f0-9]{64}$", var.image)) &&
      startswith(var.image, "${var.account_id}.dkr.ecr.${var.region}.amazonaws.com/${var.name}-cpu@sha256:")
    )
    error_message = "Compute needs private storage, public subnets/security groups and an immutable image uploaded to the owned ECR repository."
  }
}
