output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "function_url" {
  value = aws_lambda_function_url.api.function_url
}