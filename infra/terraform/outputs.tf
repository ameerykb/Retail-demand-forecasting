output "api_url" {
  description = "Public base URL of the API (add /docs for Swagger)"
  value       = aws_apigatewayv2_api.http.api_endpoint
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "model_bucket" {
  value = aws_s3_bucket.model.bucket
}

output "model_s3_uri" {
  value = "s3://${aws_s3_bucket.model.bucket}/${local.model_prefix}"
}

output "lambda_function_name" {
  value = aws_lambda_function.api.function_name
}

output "github_deploy_role_arn" {
  description = "Put this in the GitHub repo variable AWS_ROLE_ARN"
  value       = aws_iam_role.github_deploy.arn
}
