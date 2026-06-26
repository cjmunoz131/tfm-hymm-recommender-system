# ==============================================================================
# HYMM-REC CD Phase 2: Model Deployment (Model + Endpoints)
# ==============================================================================
# Despliega los modelos aprobados del Model Registry como endpoints:
#   - Full Model: predicción completa (user, item) → rating + interaction + atención
#   - User Tower: user_id → embedding 64D (para búsqueda ANN en OpenSearch)
#
# Flujo:
#   Model Package (Approved) → aws_sagemaker_model → Endpoint Config → Endpoint
# ==============================================================================

data "aws_caller_identity" "current" {
  provider = aws.account1
}
data "aws_partition" "current" {
  provider = aws.account1
}
data "aws_region" "current" {
  provider = aws.account1
}

# ==============================================================================
# FULL MODEL — SageMaker Model (from Model Package Registry)
# ==============================================================================

module "aws_ml_compute_model_serving_hymmrec_full_model_layer_model" {
  providers = {
    aws.main = aws.account1
  }
  source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-compute-model-serving-deployment-sagemaker"

  enable_sagemaker_model              = true
  project                             = var.project
  sagemaker_model_name                = var.full_model_sagemaker_name
  sagemaker_model_execution_role_arn  = aws_iam_role.sagemaker_endpoint_role.arn
  sagemaker_model_enable_network_isolation = false

  sagemaker_model_primary_container = [
    {
      image          = var.pytorch_inference_image
      model_data_url = "s3://${var.sagemaker_assets_bucket}/hymmrec/packaged-models/full-model/full_model.tar.gz"
    }
  ]

  sagemaker_model_container  = []
  sagemaker_model_vpc_config = []
}

# ==============================================================================
# FULL MODEL — Endpoint
# ==============================================================================

# module "aws_sagemaker_gov_model_serving_full_model_endpoint_layer_module" {
#   providers = {
#     aws.main = aws.account1
#   }
#   source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-governance-model-serving-endpoint-sagemaker"

#   enable_sagemaker_endpoint = true
#   project                   = var.project
#   endpoint-name             = var.full_endpoint_name

#   enable_sagemaker_endpoint_configuration = true
#   endpoint_configuration_name             = var.full_endpoint_config_name
#   sagemaker_endpoint_configuration_kms_key_arn = var.storage_kms_key_id
#   sagemaker_endpoint_configuration_production_variants = [
#     {
#       variant_name           = "AllTraffic"
#       model_name             = module.aws_ml_compute_model_serving_hymmrec_full_model_layer_model.sagemaker_model_id
#       initial_instance_count = 1
#       instance_type          = var.endpoint_instance_type
#     }
#   ]

#   # Autoscaling
#   enable_sagemaker_default_autoscaling = true
#   endpoint_instance_min_capacity       = 1
#   endpoint_instance_max_capacity       = 2
#   scale_in_cooldown                    = 300
#   scale_out_cooldown                   = 60
#   sagemaker_variant_name               = "AllTraffic"
#   invocations_target_value             = 100
# }


# ==============================================================================
# USER TOWER — SageMaker Model (from Model Package Registry)
# ==============================================================================

module "aws_ml_compute_model_serving_hymmrec_user_tower_model_layer_model" {
  providers = {
    aws.main = aws.account1
  }
  source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-compute-model-serving-deployment-sagemaker"

  enable_sagemaker_model              = true
  project                             = var.project
  sagemaker_model_name                = var.user_tower_model_sagemaker_name
  sagemaker_model_execution_role_arn  = aws_iam_role.sagemaker_endpoint_role.arn
  sagemaker_model_enable_network_isolation = false

  sagemaker_model_primary_container = [
    {
      image          = var.pytorch_inference_image
      model_data_url = "s3://${var.sagemaker_assets_bucket}/hymmrec/packaged-models/user-tower/user_tower.tar.gz"
    }
  ]

  sagemaker_model_container  = []
  sagemaker_model_vpc_config = []
}

# ==============================================================================
# USER TOWER — Endpoint
# ==============================================================================

# module "aws_sagemaker_gov_model_serving_user_tower_endpoint_layer_module" {
#   providers = {
#     aws.main = aws.account1
#   }
#   source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-governance-model-serving-endpoint-sagemaker"

#   enable_sagemaker_endpoint = true
#   project                   = var.project
#   endpoint-name             = var.user_tower_endpoint_name

#   enable_sagemaker_endpoint_configuration = true
#   endpoint_configuration_name             = var.user_tower_endpoint_config_name
#   sagemaker_endpoint_configuration_kms_key_arn = var.storage_kms_key_id
#   sagemaker_endpoint_configuration_production_variants = [
#     {
#       variant_name           = "AllTraffic"
#       model_name             = module.aws_ml_compute_model_serving_hymmrec_user_tower_model_layer_model.sagemaker_model_id
#       initial_instance_count = 1
#       instance_type          = var.endpoint_instance_type
#     }
#   ]

#   # Autoscaling
#   enable_sagemaker_default_autoscaling = true
#   endpoint_instance_min_capacity       = 1
#   endpoint_instance_max_capacity       = 2
#   scale_in_cooldown                    = 300
#   scale_out_cooldown                   = 60
#   sagemaker_variant_name               = "AllTraffic"
#   invocations_target_value             = 100
# }


# ==============================================================================
# ITEM TOWER — SageMaker Model (para Batch Transform en Inference Pipeline)
# ==============================================================================
# Este modelo es consumido por el Batch Transform del proyecto hymm-inf-exp-arch
# para generar item embeddings 64D + attention_weights.
# Input: JSONL con {item_idx, genres_multihot[20D], text_emb[1024D], img_emb[1024D]}
# Output: {item_embedding[64D], attention_weights: {category, text, image}}

module "aws_ml_compute_model_serving_hymmrec_item_tower_model_layer_model" {
  providers = {
    aws.main = aws.account1
  }
  source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-compute-model-serving-deployment-sagemaker"

  enable_sagemaker_model              = true
  project                             = var.project
  sagemaker_model_name                = var.item_tower_model_sagemaker_name
  sagemaker_model_execution_role_arn  = aws_iam_role.sagemaker_endpoint_role.arn
  sagemaker_model_enable_network_isolation = false

  sagemaker_model_primary_container = [
    {
      image          = var.pytorch_inference_image
      model_data_url = "s3://${var.sagemaker_assets_bucket}/hymmrec/packaged-models/item-tower/item_tower.tar.gz"
    }
  ]

  sagemaker_model_container  = []
  sagemaker_model_vpc_config = []
}
