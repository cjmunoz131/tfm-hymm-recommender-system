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

######## MLOPS PIPELINES ##############
module "aws-ml-governance-model-ops-hymmrec-pipelines-sagemaker-layer-module" {
  providers = {
    aws.main = aws.account1
  }
  source          = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-governance-model-ops-pipelines-sagemaker"
  project         = var.project
  enable_sagemaker_pipeline = true
  pipeline-sm-name = var.sagemaker_pipeline_name
  source_definition_path = "${path.root}/sm-dag-pipelines"
  vars_map = {}
  create_role = true
  role_name = "${var.project}-${var.sagemaker_pipeline_name}-smp-iar-${terraform.workspace}"
  custom_policy_path = "${path.root}/extra-policies/sagemaker-pipeline"
  create_terraform_style = false
  parameters_custom_policy_map = {
    region                  = data.aws_region.current.name
    account_id              = data.aws_caller_identity.current.account_id
    project                 = var.project
    pipeline_role_name      = var.sm_pipeline_execution_role_name
    s3_sagemaker_assets_arn = "arn:aws:s3:::${var.sagemaker_scripts_bucket}"
    s3_datalake_gold_arn    = "arn:aws:s3:::${var.gold_bucket_name}"
    s3_datalake_silver_arn  = "arn:aws:s3:::${var.silver_bucket_name}"
    kms_arn                 = var.storage_kms_key_id
  }
}


# ==============================================================================
# SAGEMAKER PIPELINE SCRIPTS — Upload dev/ scripts to S3
# ==============================================================================
# Terraform sube recursivamente los scripts de processing, training, evaluation
# e inference a S3 para que el SageMaker Pipeline los referencie en runtime.
# Estructura en S3:
#   s3://hymmrec-sagemaker-assets/sagemaker-scripts/
#     ├── feng-data-preparing/   (processing scripts)
#     ├── training/              (HPO + training scripts + dependencies)
#     ├── evaluation/            (evaluation + packaging scripts)
#     └── inference/             (inference scripts)
# ==============================================================================

locals {
  sagemaker_scripts_bucket = var.sagemaker_scripts_bucket
  scripts_s3_prefix        = "sagemaker-scripts"

  # Collect all .py files from dev/ subdirectories (excluding notebook-scripts/)
  processing_scripts = fileset("${path.module}/dev/feng-data-preparing", "*.py")
  training_scripts   = fileset("${path.module}/dev/training", "*.py")
  evaluation_scripts = fileset("${path.module}/dev/evaluation", "*.py")
  inference_scripts  = fileset("${path.module}/dev/inference", "*.py")
}

# --- Processing scripts (feng-data-preparing) ---
resource "aws_s3_object" "processing_scripts" {
  provider = aws.account1
  for_each = local.processing_scripts

  bucket       = local.sagemaker_scripts_bucket
  key          = "${local.scripts_s3_prefix}/feng-data-preparing/${each.value}"
  source       = "${path.module}/dev/feng-data-preparing/${each.value}"
  etag         = filemd5("${path.module}/dev/feng-data-preparing/${each.value}")
  content_type = "text/x-python"

  tags = {
    project = var.project
    phase   = "processing"
  }
}

# --- Training scripts (HPO + train + dependencies) ---
resource "aws_s3_object" "training_scripts" {
  provider = aws.account1
  for_each = local.training_scripts

  bucket       = local.sagemaker_scripts_bucket
  key          = "${local.scripts_s3_prefix}/training/${each.value}"
  source       = "${path.module}/dev/training/${each.value}"
  etag         = filemd5("${path.module}/dev/training/${each.value}")
  content_type = "text/x-python"

  tags = {
    project = var.project
    phase   = "training"
  }
}

# --- Evaluation scripts ---
resource "aws_s3_object" "evaluation_scripts" {
  provider = aws.account1
  for_each = local.evaluation_scripts

  bucket       = local.sagemaker_scripts_bucket
  key          = "${local.scripts_s3_prefix}/evaluation/${each.value}"
  source       = "${path.module}/dev/evaluation/${each.value}"
  etag         = filemd5("${path.module}/dev/evaluation/${each.value}")
  content_type = "text/x-python"

  tags = {
    project = var.project
    phase   = "evaluation"
  }
}

# --- Inference scripts ---
resource "aws_s3_object" "inference_scripts" {
  provider = aws.account1
  for_each = local.inference_scripts

  bucket       = local.sagemaker_scripts_bucket
  key          = "${local.scripts_s3_prefix}/inference/${each.value}"
  source       = "${path.module}/dev/inference/${each.value}"
  etag         = filemd5("${path.module}/dev/inference/${each.value}")
  content_type = "text/x-python"

  tags = {
    project = var.project
    phase   = "inference"
  }
}


# ==============================================================================
# SECURITY GROUP — SageMaker Domain Studio
# ==============================================================================

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