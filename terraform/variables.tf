variable "region" {
  type    = string
  default = "eu-west-1"
}

variable "function_name" {
  type    = string
  default = "churn-api"
}

variable "image_tag" {
  description = "ECR image tag the Lambda runs, normally the short git commit hash."
  type        = string
}