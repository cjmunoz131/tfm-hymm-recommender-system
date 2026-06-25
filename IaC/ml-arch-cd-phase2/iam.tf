# ==============================================================================
# IAM ROLE — Endpoint Execution Role
# ==============================================================================
# Role que asume el contenedor del endpoint para:
#   - Descargar model.tar.gz desde S3 (resuelto desde Model Package)
#   - Desencriptar con KMS
#   - Pull de imagen ECR (PyTorch inference)
#   - Escribir logs a CloudWatch

resource "aws_iam_role" "sagemaker_endpoint_role" {
  provider = aws.account1
  name     = "${var.project}-sm-endpoint-iar-${terraform.workspace}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_endpoint_policy" {
  provider = aws.account1
  name     = "${var.project}-sm-endpoint-policy-${terraform.workspace}"
  role     = aws_iam_role.sagemaker_endpoint_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ModelAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.sagemaker_assets_bucket}",
          "arn:aws:s3:::${var.sagemaker_assets_bucket}/*"
        ]
      },
      {
        Sid    = "KMSDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:GenerateDataKey*"
        ]
        Resource = [var.storage_kms_key_id]
      },
      {
        Sid    = "ECRPull"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/Endpoints/*"
      },
      {
        Sid    = "CloudWatchMetrics"
        Effect = "Allow"
        Action = ["cloudwatch:PutMetricData"]
        Resource = "*"
      }
    ]
  })
}