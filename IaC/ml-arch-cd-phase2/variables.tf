variable "project" {
  type        = string
  description = "Deployment project"
  default     = "hymmrec"
}

variable "provisioner" {
  type        = string
  description = "Infraestructure provisioner"
  default     = "Terraform"
}

variable "owner" {
  type        = string
  description = "Project Owner"
  default     = "cjmunoz"
}

variable "org_unit" {
  type        = string
  description = "Organizational unit"
  default     = "products_crew"
}

variable "fin_unit" {
  type        = string
  description = "finance unit"
  default     = "vice_technology"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

########################

variable "storage_kms_key_id" {
  description = "KMS key ARN for encryption at rest"
  type        = string
  default     = "arn:aws:kms:us-east-1:697682206292:key/25b8c612-11f5-4f9e-bfa7-9d9fb69ecc64"
}

variable "sagemaker_assets_bucket" {
  description = "S3 bucket with packaged model artifacts (Platinum)"
  type        = string
  default     = "hymmrec-sagemaker-assets"
}

variable "endpoint_instance_type" {
  description = "Instance type for SageMaker endpoints"
  type        = string
  default     = "ml.m5.large"
}

# ==============================================================================
# Model Package ARNs (from SageMaker Model Registry - Approved versions)
# ==============================================================================
# Estos ARNs se actualizan cuando se aprueba una nueva versión del modelo.
# Formato: arn:aws:sagemaker:REGION:ACCOUNT:model-package/GROUP_NAME/VERSION
# Obtener con:
#   aws sagemaker list-model-packages --model-package-group-name hymmrec-multimodal-recommender \
#       --model-approval-status Approved --query 'ModelPackageSummaryList[0].ModelPackageArn'

variable "full_model_package_arn" {
  description = "ARN del Model Package aprobado para Full Model (Two-Heads)"
  type        = string
  default     = "hymmrec-model-sm-pg/1"
}

variable "user_tower_model_package_arn" {
  description = "ARN del Model Package aprobado para User Tower"
  type        = string
  default     = "hymmrec-model-sm-pg/1"
}

variable "full_endpoint_name" {
  type = string
  default = "full-model"
}

variable "full_endpoint_config_name" {
  type = string
  default = "full-model"
}

variable "full_model_sagemaker_name" {
  type = string
  default = "full-model"
}

variable "user_tower_endpoint_name" {
  type = string
  default = "user-tower"
}

variable "user_tower_endpoint_config_name" {
  type = string
  default = "user-tower"
}

variable "user_tower_model_sagemaker_name" {
  type = string
  default = "user-tower"
}

variable "pytorch_inference_image" {
  description = "PyTorch inference container image URI (us-east-1, CPU, py310)"
  type        = string
  default     = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker"
}