variable "region" {
  description = "AWS region (ap-south-1 = Mumbai)"
  type        = string
  default     = "ap-south-1"
}

variable "project" {
  description = "Prefix for every resource name"
  type        = string
  default     = "demand-forecast"
}

variable "image_tag" {
  description = "Tag of the image in ECR used when the function is first created"
  type        = string
  default     = "latest"
}

variable "lambda_memory_mb" {
  description = "Lambda memory (CPU scales with it). 1024 MB keeps cold starts with pandas + LightGBM reasonable."
  type        = number
  default     = 1024
}

variable "throttle_burst" {
  type    = number
  default = 10
}

variable "throttle_rate" {
  description = "Steady-state requests per second allowed through API Gateway"
  type        = number
  default     = 5
}

variable "budget_email" {
  description = "Email for cost alerts. Leave empty to skip creating the budget."
  type        = string
  default     = ""
}

variable "budget_usd" {
  type    = string
  default = "5"
}

variable "github_repo" {
  description = "owner/name of the GitHub repository allowed to deploy"
  type        = string
  default     = "ameerykb/Retail-demand-forecasting"
}

variable "create_github_oidc_provider" {
  description = "Set false if this AWS account already has the GitHub OIDC provider"
  type        = bool
  default     = true
}
