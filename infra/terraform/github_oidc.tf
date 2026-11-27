# Lets the GitHub Actions deploy workflow use AWS WITHOUT stored access keys:
# GitHub issues a short-lived OIDC token, AWS swaps it for temporary credentials,
# and only workflows running on this repo's main branch are trusted.
#
# Off by default: accounts on AWS's Free plan (the project-based sign-up) are not
# allowed to manage identity providers (iam:*Provider* is denied), so there you
# deploy from your laptop instead. Set enable_github_deploy = true on a Paid plan.

locals {
  github_deploy = var.enable_github_deploy
  create_oidc   = var.enable_github_deploy && var.create_github_oidc_provider
  lookup_oidc   = var.enable_github_deploy && !var.create_github_oidc_provider
}

resource "aws_iam_openid_connect_provider" "github" {
  count           = local.create_oidc ? 1 : 0
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"] # not checked by AWS for GitHub, but required by the API
}

data "aws_iam_openid_connect_provider" "github" {
  count = local.lookup_oidc ? 1 : 0
  url   = "https://token.actions.githubusercontent.com"
}

locals {
  github_oidc_arn = local.create_oidc ? aws_iam_openid_connect_provider.github[0].arn : (local.lookup_oidc ? data.aws_iam_openid_connect_provider.github[0].arn : null)
}

resource "aws_iam_role" "github_deploy" {
  count = local.github_deploy ? 1 : 0
  name  = "${var.project}-github-deploy"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.github_oidc_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:ref:refs/heads/main"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "github_deploy" {
  count = local.github_deploy ? 1 : 0
  name  = "deploy-api"
  role  = aws_iam_role.github_deploy[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EcrLogin"
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Sid    = "PushImage"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:CompleteLayerUpload",
          "ecr:GetDownloadUrlForLayer", "ecr:InitiateLayerUpload", "ecr:PutImage", "ecr:UploadLayerPart"
        ]
        Resource = aws_ecr_repository.api.arn
      },
      {
        Sid      = "UpdateFunction"
        Effect   = "Allow"
        Action   = ["lambda:UpdateFunctionCode", "lambda:GetFunction", "lambda:GetFunctionConfiguration"]
        Resource = aws_lambda_function.api.arn
      }
    ]
  })
}
