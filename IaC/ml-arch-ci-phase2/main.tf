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

  enable_sagemaker_package_group = true
  project                        = var.project
  package_group_name             = var.package_group_name
}


######## SAGEAMAKER DOMAIN STUDIO ###########
module "aws-ml-compute-model-dev-domain-sagemaker-layer-module" {
  providers = {
    aws.main = aws.account1
  }
  source                                   = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-compute-model-dev-domain-sagemaker"
  project                                  = var.project
  sagemaker_domain_auth_mode               = var.sagemaker_domain_auth_mode
  sagemaker_domain_vpc_id                  = var.vpc_id
  sagemaker_domain_subnet_ids              = var.private_subnet_id_list
  sagemaker_domain_kms_key_id              = var.integration_kms_key_id
  sagemaker_domain_app_network_access_type = var.sagemaker_domain_app_network_access_type
  domain_security_groups                   = [aws_security_group.sagemaker_sg.id]
  sagemaker_domain_jupyter_server_app_settings = {
  }
  auto_shutdown_enabled = false
  time_series_forecasting_settings_enabled = "ENABLED"
  efs_retention_policy                     = var.efs_retention_policy
}


resource "aws_security_group" "sagemaker_sg" {
  provider    = aws.account1
  name        = "sagemaker-domainstudio-${var.project}-sg"
  description = "Allow certain NFS and TCP inbound traffic"
  vpc_id      = var.vpc_id

  ingress {
    description = "NFS traffic over TCP on port 2049 between the domain and EFS volume"
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "TCP traffic between JupyterServer app and the KernelGateway apps"
    from_port   = 8192
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }

  egress {
    description      = "Allow all outbound traffic"
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}