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

# 1. Definición del Documento de Confianza (Trust Policy)
# data "aws_iam_policy_document" "sfn_assume_role" {
#   provider = aws.account1
#   statement {
#     effect  = "Allow"
#     actions = ["sts:AssumeRole"]

#     principals {
#       type        = "Service"
#       identifiers = ["states.${data.aws_region.current.name}.amazonaws.com"]
#     }
#   }
# }

# # 2. Creación del Rol de IAM
# resource "aws_iam_role" "sfn_role" {
#   provider           = aws.account1
#   name               = "hymmrec-master-data-pipeline-sfn-role-dev"
#   assume_role_policy = data.aws_iam_policy_document.sfn_assume_role.json
# }
# # 3. Política para Logs y EventBridge (Managed Rules)
# resource "aws_iam_policy" "sfn_internal_ops" {
#   provider    = aws.account1
#   name        = "hymmrec-sfn-internal-ops-policy"
#   description = "Permisos para CloudWatch Logs y Managed Rules de EventBridge"

#   policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [
#       {
#         # Permisos de Logs que ya tenías
#         Effect = "Allow"
#         Action = [
#           "logs:CreateLogDelivery",
#           "logs:GetLogDelivery",
#           "logs:UpdateLogDelivery",
#           "logs:DeleteLogDelivery",
#           "logs:ListLogDeliveries",
#           "logs:PutResourcePolicy",
#           "logs:DescribeResourcePolicies",
#           "logs:DescribeLogGroups"
#         ]
#         Resource = "*"
#       }
#     ]
#   })
# }

# # 4. Adjuntar Políticas (Usando role_policy_attachment para evitar colisiones)
# resource "aws_iam_role_policy_attachment" "attach_internal_ops" {
#   provider   = aws.account1
#   role       = aws_iam_role.sfn_role.name
#   policy_arn = aws_iam_policy.sfn_internal_ops.arn
# }