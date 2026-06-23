data "aws_caller_identity" "current" {
  provider = aws.account1
}
data "aws_partition" "current" {
  provider = aws.account1
}
data "aws_region" "current" {
  provider = aws.account1
}

locals {
  partition = data.aws_partition.current.partition
}

module "hymmrec_feature_group" {
  providers = {
    aws.main = aws.account1
  }
  source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-governance-model-dev-featurestore-sagemaker"

  enable_sagemaker_feature_group = true
  project                        = var.project
  feature_group_name             = "interactions"
  record_identifier_name         = "recordid"
  event_time_name                = "eventtime"
  sagemaker_feature_group_role_arn = var.sagemaker_execution_role_arn
  sagemaker_feature_group_feature_definition = [
    { feature_name = "userId", feature_type = "Integral" },
    { feature_name = "movieId", feature_type = "Integral" },
    { feature_name = "rating", feature_type = "Fractional" },
    { feature_name = "timestamp", feature_type = "Integral" },
    { feature_name = "generos", feature_type = "String" },
    { feature_name = "rating_scaled", feature_type = "Fractional" },
    { feature_name = "userId_idx", feature_type = "Integral" },
    { feature_name = "movieId_idx", feature_type = "Integral" },
    # IMPORTANTE: Estos nombres deben coincidir con la concatenación exacta que hace tu módulo:
    { feature_name = "hymmrec_eventtime_sm_et_fn", feature_type = "String" },
    { feature_name = "hymmrec_recordid_sm_ri_fn", feature_type = "String" }
  ]

  # ==========================================
  # CONFIGURACIÓN DEL OFFLINE STORE
  # ==========================================
  offline_table_format = "Iceberg"
  sagemaker_feature_group_s3_storage_config = [
    {
      s3_uri     = "s3://${var.gold_bucket_name}/data/${var.ml_use_case}/"
      kms_key_id = var.storage_kms_key_id
    }
  ]

  sagemaker_feature_group_data_catalog_config = [
    {
      catalog    = "AwsDataCatalog"
      database   = var.glue_database_name
      table_name = var.offline_feature_store_table
    }
  ]
  sagemaker_feature_group_security_config = []
}

module "aws_ml_gov_model_serving_hymmrec_package_group_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-governance-model-serving-packagegroup-sagemaker"

  enable_sagemaker_feature_group = true
  project                        = var.project
  package_group_name             = var.package_group_name
}