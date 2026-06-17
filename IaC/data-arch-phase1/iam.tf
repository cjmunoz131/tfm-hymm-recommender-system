resource "aws_kms_key_policy" "kms_key_access" {
  provider = aws.account1
  key_id   = module.aws_security_keys_integration_layer_module.kms_key_id
  policy   = data.aws_iam_policy_document.kms_key_access.json
}

resource "aws_kms_key_policy" "storage_kms_key_access" {
  provider = aws.account1
  key_id   = module.aws_security_keys_kms_storage_layer_module.kms_key_id
  policy   = data.aws_iam_policy_document.kms_key_access.json
}

data "aws_iam_policy_document" "kms_key_access" {
  provider = aws.account1
  statement {
    sid    = "Enable IAM User Permissions"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
    actions = [
      "kms:*"
    ]
    resources = [
      "*"
    ]
  }
  statement {
    sid    = "Enable Cloudwatch access to KMS Key"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["logs.${data.aws_region.current.name}.amazonaws.com"]
    }
    actions = [
      "kms:Encrypt*",
      "kms:Decrypt*",
      "kms:ReEncrypt*",
      "kms:GenerateDataKey*",
      "kms:Describe*"
    ]
    resources = [
      "*"
    ]
    condition {
      test     = "ArnLike"
      variable = "kms:EncryptionContext:aws:logs:arn"
      values = [
        "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      ]
    }
  }
}

# ---------------------------------------------------------------
# SAGEMAKER ROLE — Permisos KMS Storage + Glue Catalog
# ---------------------------------------------------------------
# Permite al rol de SageMaker (usado por Feature Store y Processing Jobs)
# acceder al KMS de storage y al Glue Data Catalog en Gold layer.

resource "aws_iam_policy" "sagemaker_kms_glue_access" {
  provider    = aws.account1
  name        = "${var.project}-sagemaker-kms-glue-access"
  description = "Permisos de KMS storage y Glue Catalog para SageMaker (Feature Store + Processing)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowKMSStorageAccess"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey",
          "kms:CreateGrant"
        ]
        Resource = [module.aws_security_keys_kms_storage_layer_module.kms_key_arn]
      },
      {
        Sid    = "AllowGlueCatalogAccess"
        Effect = "Allow"
        Action = [
          "glue:GetDatabase",
          "glue:GetDatabases",
          "glue:GetTable",
          "glue:GetTables",
          "glue:GetPartitions",
          "glue:CreateTable",
          "glue:UpdateTable",
          "glue:DeleteTable",
          "glue:BatchCreatePartition",
          "glue:BatchDeletePartition",
          "glue:GetPartition",
          "glue:CreatePartition",
          "glue:DeletePartition",
          "glue:UpdatePartition"
        ]
        Resource = [
          "arn:aws:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:catalog",
          "arn:aws:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:database/*",
          "arn:aws:glue:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:table/*/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_kms_glue_attach" {
  provider   = aws.account1
  role       = element(split("/", module.sagemaker-notebook-instance.sagemaker_instance_role_arn), length(split("/", module.sagemaker-notebook-instance.sagemaker_instance_role_arn)) - 1)
  policy_arn = aws_iam_policy.sagemaker_kms_glue_access.arn
}


# ---------------------------------------------------------------
# SAGEMAKER ROLE — Acceso S3 al bucket Gold (Feature Store offline)
# ---------------------------------------------------------------
resource "aws_iam_role_policy" "sagemaker_s3_gold_access" {
  provider = aws.account1
  name     = "${var.project}-sagemaker-s3-gold-access"
  role     = element(split("/", module.sagemaker-notebook-instance.sagemaker_instance_role_arn), length(split("/", module.sagemaker-notebook-instance.sagemaker_instance_role_arn)) - 1)

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowS3GoldBucketAccess"
        Effect = "Allow"
        Action = [
          "s3:GetBucketAcl",
          "s3:GetBucketLocation",
          "s3:ListBucket",
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts"
        ]
        Resource = [
          module.aws_storage_gold_objects_s3_bucket_layer_module.bucket_arn,
          "${module.aws_storage_gold_objects_s3_bucket_layer_module.bucket_arn}/*"
        ]
      }
    ]
  })
}


# ---------------------------------------------------------------
# SAGEMAKER ROLE — Invocación de Amazon Bedrock (Nova Multimodal Embeddings)
# ---------------------------------------------------------------
resource "aws_iam_role_policy" "sagemaker_bedrock_invoke" {
  provider = aws.account1
  name     = "${var.project}-sagemaker-bedrock-invoke"
  role     = element(split("/", module.sagemaker-notebook-instance.sagemaker_instance_role_arn), length(split("/", module.sagemaker-notebook-instance.sagemaker_instance_role_arn)) - 1)

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowBedrockInvokeModel"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          "arn:aws:bedrock:${data.aws_region.current.name}::foundation-model/amazon.nova-2-multimodal-embeddings-v1:0",
          "arn:aws:bedrock:${data.aws_region.current.name}::foundation-model/amazon.nova-*"
        ]
      }
    ]
  })
}
