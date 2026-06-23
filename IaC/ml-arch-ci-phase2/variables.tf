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
  description = "value"
  type        = string
  default     = "arn:aws:kms:us-east-1:697682206292:key/25b8c612-11f5-4f9e-bfa7-9d9fb69ecc64"
}

variable "offline_feature_store_table" {
  description = "offline feature store table"
  type = string
  default = "hymmrec_feature_interactions"
}

variable "glue_database_name" {
  description = "gold database name"
  type = string
  default = "hymmrec_tfm_ml_feature_store_gold"
}


variable "sagemaker_execution_role_arn" {
  description = "sagemaker arn role"
  type = string
  default = "arn:aws:iam::697682206292:role/sgmkr-notebook-tfm-hymm-rec-ml-iar-dev"
}

variable "gold_bucket_name" {
  description = "gold bucket"
  type = string
  default = "hymmrec-dilkehousegold01"
}

variable "ml_use_case" {
  type = string
  description = "use case in gold layer"
  default = "ml_feature_store"
}

variable "package_group_name" {
  type = string
  description = "package group name"
  default = "model"
}