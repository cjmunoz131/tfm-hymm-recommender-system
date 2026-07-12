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

module "aws_networking_base_vpc_layer_module" {
  providers = {
    aws.main = aws.account1
    aws.dns  = aws.dns
  }
  source               = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-networking-base-vpc-tgw"
  cidr_block           = var.vpc_base_cidr_block
  vpc_name             = var.vpc_name
  availability_zones   = var.availability_zones
  private_subnets      = var.subnet_private_cidr_blocks
  public_subnets       = var.subnet_public_cidr_blocks
  database_subnets     = var.subnet_database_cidr_blocks
  create_nat_gateway   = true
  enable_dns_hostnames = true
}

module "aws_networking_integration_vpc_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source           = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-networking-integration-vpc"
  vpc_workloads_id = module.aws_networking_base_vpc_layer_module.id
  gw_endpoints_services = [
    { type = "s3", route_tables = module.aws_networking_base_vpc_layer_module.private_route_table_id_list },
    { type = "dynamodb", route_tables = module.aws_networking_base_vpc_layer_module.private_route_table_id_list }
  ]
}

module "aws_security_keys_integration_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source   = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-security-keys-kms"
  key_name = var.integration_kms_key_name
}

module "aws_security_keys_kms_storage_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source   = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-security-keys-kms"
  key_name = var.storage_kms_key_name
}

############## SAGEMAKER INSTANCE AND ASSETS ###########
module "aws_storage_ml_objects_s3_bucket_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-storage-objects-s3"
  bucket_name           = "${var.project}-sagemaker-assets"
  force_destroy         = true
  versioning            = "Disabled"
  object_lock_enabled   = false
  lifecycle_rules       = []
  is_kms_used           = false
  bucket_policy_enabled = false
  confidentiality       = "internal"
  integrity             = "tolerable"
}

module "sagemaker-notebook-instance" {
  providers = {
    aws.main = aws.account1
  }
  source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-environment-model-dev-compute-notebook"
  project = "tfm-hymm-rec"
  vpc_id = module.aws_networking_base_vpc_layer_module.id
  private_subnet_id = module.aws_networking_base_vpc_layer_module.private_subnet_id_list[0]
  sagemaker_instance_name = "tfm-hymm-rec-ml"
  sagemaker_instance_type = "ml.t3.medium"
  platform_identifier = "notebook-al2-v3"
  direct_internet_access = "Disabled"
  volume_size = 50
  target_buckets = [module.aws_storage_ml_objects_s3_bucket_layer_module.bucket_id, module.aws_storage_gold_objects_s3_bucket_layer_module.bucket_id, module.aws_storage_silver_objects_s3_bucket_layer_module.bucket_id]
  repo_url = "https://github.com/cjmunoz131/tfm-hymm-recommender-system.git"
  env_name = "ml-dev"
  python_version = "3.11"
}

############################ DATALAKE ##############################
# Creacion del bronze zone del datalake
module "aws_storage_bronze_objects_s3_bucket_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-storage-objects-s3"
  bucket_name           = "${var.project}-dilkehousebronze01"
  force_destroy         = true
  versioning            = "Disabled"
  object_lock_enabled   = false
  lifecycle_rules       = []
  rule_sse_algorithm    = "aws:kms"
  arn_kms_key           = module.aws_security_keys_kms_storage_layer_module.kms_key_arn
  is_kms_used           = true
  bucket_policy_enabled = false
  confidentiality       = "internal"
  integrity             = "tolerable"
}

# Creacion del silver zone del datalake
module "aws_storage_silver_objects_s3_bucket_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-storage-objects-s3"
  bucket_name           = "${var.project}-dilkehousesilver01"
  force_destroy         = true
  versioning            = "Disabled"
  object_lock_enabled   = false
  lifecycle_rules       = []
  rule_sse_algorithm    = "aws:kms"
  arn_kms_key           = module.aws_security_keys_kms_storage_layer_module.kms_key_arn
  is_kms_used           = true
  bucket_policy_enabled = false
  confidentiality       = "internal"
  integrity             = "tolerable"
}

# Creacion del gold zone del datalake
module "aws_storage_gold_objects_s3_bucket_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-storage-objects-s3"
  bucket_name           = "${var.project}-dilkehousegold01"
  force_destroy         = true
  versioning            = "Disabled"
  object_lock_enabled   = false
  lifecycle_rules       = []
  rule_sse_algorithm    = "aws:kms"
  arn_kms_key           = module.aws_security_keys_kms_storage_layer_module.kms_key_arn
  is_kms_used           = true
  bucket_policy_enabled = false
  confidentiality       = "internal"
  integrity             = "tolerable"
}
### GLUE ARTEFACTS ###
module "aws_data_storage_glue_assests_bucket_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-storage-objects-s3"
  bucket_name           = "${var.project}-glue-assests-bucket"
  force_destroy         = true
  versioning            = "Disabled"
  object_lock_enabled   = false
  lifecycle_rules       = []
  rule_sse_algorithm    = "aws:kms"
  arn_kms_key           = module.aws_security_keys_kms_storage_layer_module.kms_key_arn
  is_kms_used           = false
  bucket_policy_enabled = false
  confidentiality       = "internal"
  integrity             = "tolerable"
}

#### CONFIGS SSM Parameter Store and Secrets Manager ######
resource "aws_ssm_parameter" "hymmrec_execution_plan_tmpl" {
  provider = aws.account1
  name     = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_trigger_functionality}/execution-plan-manifest.json"
  type     = "String"
  value    = file("${path.root}/config/execution-plan-manifest.json")
}

resource "aws_ssm_parameter" "hymmrec_domain_context" {
  provider = aws.account1
  name     = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_trigger_functionality}/domain-context.json"
  type     = "String"
  value    = file("${path.root}/config/domain-context.json")
}

resource "aws_ssm_parameter" "tmdb_baseline_params" {
  provider = aws.account1
  name     = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_master_functionality}/PRCS-EXT-011-BRNZ-002/TSKG0001/TSK0001"
  type     = "String"
  value    = file("${path.root}/config/tmdb-baseline-params.json")
}

resource "aws_ssm_parameter" "tmdb_posters_params" {
  provider = aws.account1
  name     = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_master_functionality}/PRCS-EXT-012-SLVR-003/TSKG0002/TSK0002"
  type     = "String"
  value    = file("${path.root}/config/tmdb-posters-params.json")
}

resource "aws_ssm_parameter" "obt_silver_params" {
  provider = aws.account1
  name     = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_master_functionality}/PRCS-BRNZ-002-SLVR-003/TSKG0003/TSK0003"
  type     = "String"
  value    = file("${path.root}/config/obt-silver-params.json")
}

module "aws_security_secrets_ssmstore_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                           = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-security-secret-secret-manager"
  secret_name                      = "${var.governance_domain}-${var.project}-${var.application-domain}"
  create_random_password           = true
  random_password_length           = 16
  random_password_override_special = "!&"
}
################# (TRIGGER EVENT) #################################################
################# (EVENT SCHEDULER CRON 1 time a DAY) #############################

module "aws_integration_event_bus_event_bridge_scheduler_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source              = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-integration-event-bus-event_bridge"
  name                = "${var.project}-ebr-scheduler-${terraform.workspace}"
  description         = "Eventbridge rule scheduler used to trigger the step function map processing"
  rule_type           = "schedule"
  project             = var.project
  schedule_expression = var.event_bridge_scheduler_expression
  state               = "ENABLED"
  targets = [{
    target_id     = "lambda-${var.project}-${var.data_orchestrator_trigger_functionality}-${terraform.workspace}"
    arn           = module.aws_app_compute_lambda_hymmrec-data-orchestrator-trigger_layer_module.lambda_arn
    required_role = true
  }]
  statements_policy = [{
    Effect   = "Allow"
    Action   = ["lambda:InvokeFunction"]
    Resource = [module.aws_app_compute_lambda_hymmrec-data-orchestrator-trigger_layer_module.lambda_arn]
  }]
}

################# (LAMBDA TRIGGER) #############################
module "aws_app_compute_lambda_hymmrec-data-orchestrator-trigger_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                    = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-app-compute-lambda"
  subnets_ids               = module.aws_networking_base_vpc_layer_module.private_subnet_id_list
  security_group_ids        = [aws_security_group.sg_lambda.id]
  vpc_attach                = true
  lambda_name               = "${var.project}-${var.data_orchestrator_trigger_functionality}"
  lambda_script             = ""
  lambda_runtime            = var.lambda_runtime
  description               = "functionality: omnichannel data orchestrator trigger for the ${var.project} project"
  source_code_path          = "./${path.root}/dev/lambdas"
  output_zip_path           = "./${path.root}/dev/artefacts/lambdas"
  create_layers             = false
  lambda_layers_definitions = {}
  lambda_layers             = null
  project                   = var.project
  use_existing_role         = false
  add_custom_policy         = true
  custom_policy_path        = "${path.root}/extra-policies/lambda"
  parameters_custom_policy_map = {
    region                  = data.aws_region.current.name
    account_id              = data.aws_caller_identity.current.account_id
    ssm_parameter_prefix    = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_trigger_functionality}"
    sfn_master_pipeline_arn = module.aws_integration_workflow_master_pipeline_step_function_layer_module.state_machine_arn # "*"
  }
  environment_variables = {
    EXECUTION_PLAN_MANIFEST = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_trigger_functionality}/execution-plan-manifest.json"
    DOMAIN_CONTEXT_PARAMETER = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_trigger_functionality}/domain-context.json"
    STATE_MACHINE_ARN       = module.aws_integration_workflow_master_pipeline_step_function_layer_module.state_machine_arn
    REGION                  = data.aws_region.current.name
    ID_DOMAIN_PARAMETER     = var.domain_id
  }
}
###################### GOVERNANCE GLUE DATA CATALOG ###################################
####### DATABASES #################
### BRONZE LAYER ###
module "aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                       = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-governance-catalog-database-glue"
  catalog_database_name        = format("%s_%s_%s_%s_%s", var.project, var.governance_domain, var.application-domain, "api_raw" , "bronze")
  catalog_database_description = "database in glue catalog for ${var.project}"
  parameters = {
    location             = "s3://${module.aws_storage_bronze_objects_s3_bucket_layer_module.bucket_id}/data/${var.application-domain}",
    created_by           = "terraform"
    environment          = terraform.workspace
    data_layer           = "bronze"
  }
}

module "aws_data_governance_catalog_bronze_ml_database_glue_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                       = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-governance-catalog-database-glue"
  catalog_database_name        = format("%s_%s_%s_%s_%s", var.project, var.governance_domain, var.ml-application-domain, "data_entry" , "bronze")
  catalog_database_description = "database in glue catalog for ${var.project}"
  parameters = {
    location             = "s3://${module.aws_storage_bronze_objects_s3_bucket_layer_module.bucket_id}/data/${var.ml-application-domain}",
    created_by           = "terraform"
    environment          = terraform.workspace
    data_layer           = "bronze"
  }
}
### SILVER LAYER ###
module "aws_data_governance_catalog_silver_database_glue_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                       = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-governance-catalog-database-glue"
  catalog_database_name        = format("%s_%s_%s_%s", var.project, var.governance_domain, var.movie-domain, "silver")
  catalog_database_description = "database in glue catalog for ${var.project}"
  parameters = {
    location             = "s3://${module.aws_storage_silver_objects_s3_bucket_layer_module.bucket_id}/data/${var.movie-domain}",
    created_by           = "terraform"
    environment          = terraform.workspace
    data_layer           = "silver"
  }
}

### GOLD LAYER ###
module "aws_data_governance_catalog_gold_database_glue_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                       = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-governance-catalog-database-glue"
  catalog_database_name        = format("%s_%s_%s_%s", var.project, var.governance_domain, var.ml_use_case ,"gold")
  catalog_database_description = "database in glue catalog for ${var.project}"
  parameters = {
    location             = "s3://${module.aws_storage_gold_objects_s3_bucket_layer_module.bucket_id}/data/${var.ml_use_case}",
    created_by           = "terraform"
    environment          = terraform.workspace
    data_layer           = "gold"
  }
}

### GOLD LAYER — ML Recommendations ###
module "aws_data_governance_catalog_gold_recommendations_database_glue_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                       = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-governance-catalog-database-glue"
  catalog_database_name        = format("%s_%s_%s_%s", var.project, var.governance_domain, "ml_recommendations", "gold")
  catalog_database_description = "Gold database for ML recommendations and explainability for ${var.project}"
  parameters = {
    location             = "s3://${module.aws_storage_gold_objects_s3_bucket_layer_module.bucket_id}/data/ml_recommendations",
    created_by           = "terraform"
    environment          = terraform.workspace
    data_layer           = "gold"
  }
}

### GLUE CRAWLER LAYER ###
module "aws-data-governance-metadata-tmdb-crawler-glue-layer-module" {
  providers = {
    aws.main = aws.account1
  }
  create                 = true
  source                 = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-governance-metadata-crawler-glue"
  crawler_name           = format("%s-%s-%s-%s", var.project, var.governance_domain, var.application-domain, "movies")
  crawler_description    = "crawler for movies content in tmdb database"
  database_name          = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
  security_configuration = module.aws_data_security_configuration_glue_layer_module.aws_glue_security_configuration_id
  create_role            = true
  role_name              = format("%s-crawler-%s", "tmdb", "bronze")
  s3_bucket_arn          = module.aws_storage_bronze_objects_s3_bucket_layer_module.bucket_arn
  kms_key_arn            = module.aws_security_keys_kms_storage_layer_module.kms_key_arn
  glue_connection_arn    = module.aws_data_security_configuration_glue_layer_module.glue_network_connection_arn
  recrawl_policy = {
    recrawl_behavior = "CRAWL_EVERYTHING"
  }
  schema_change_policy = {
    update_behavior = "UPDATE_IN_DATABASE"
    delete_behavior = "DEPRECATE_IN_DATABASE"
  }
  s3_target = [
    {
      path            = "s3://${module.aws_storage_bronze_objects_s3_bucket_layer_module.bucket_id}/data/${var.application-domain}/movies/"
      connection_name = module.aws_data_security_configuration_glue_layer_module.glue_network_connection_name
    }
  ]
  configuration = jsonencode({
    Version = 1.0
    CrawlerOutput = {
      Partitions = { AddOrUpdateBehavior = "InheritFromTable" }
    }
  })
}

module "aws-data-governance-metadata-ml-crawler-glue-layer-module" {
  providers = {
    aws.main = aws.account1
  }
  create                 = true
  source                 = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-governance-metadata-crawler-glue"
  crawler_name           = format("%s-%s-%s", var.project, var.governance_domain, var.ml-application-domain)
  crawler_description    = "crawler for movies content in ml database"
  database_name          = module.aws_data_governance_catalog_bronze_ml_database_glue_layer_module.name
  security_configuration = module.aws_data_security_configuration_glue_layer_module.aws_glue_security_configuration_id
  create_role            = true
  role_name              = format("%s-crawler-%s", var.ml-application-domain, "bronze")
  s3_bucket_arn          = module.aws_storage_bronze_objects_s3_bucket_layer_module.bucket_arn
  kms_key_arn            = module.aws_security_keys_kms_storage_layer_module.kms_key_arn
  glue_connection_arn    = module.aws_data_security_configuration_glue_layer_module.glue_network_connection_arn
  recrawl_policy = {
    recrawl_behavior = "CRAWL_EVERYTHING"
  }
  schema_change_policy = {
    update_behavior = "UPDATE_IN_DATABASE"
    delete_behavior = "DEPRECATE_IN_DATABASE"
  }
  s3_target = [
    {
      path            = "s3://${module.aws_storage_bronze_objects_s3_bucket_layer_module.bucket_id}/data/${var.ml-application-domain}/"
      connection_name = module.aws_data_security_configuration_glue_layer_module.glue_network_connection_name
    }
  ]
  configuration = jsonencode({
    Version = 1.0
    CrawlerOutput = {
      Partitions = { AddOrUpdateBehavior = "InheritFromTable" }
    }
  })
}

########### PROCESS METADATA EXECUTIONS HISTORY #############################
module "aws_db_dynamodb_executions_history_table_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                      = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-database-nosql-keyvalue-fastscale-dynamodb"
  dynamodb_table_name         = format("%s-%s", var.project, var.dynamodb_executions_history_table_name)
  kms_key_arn                 = module.aws_security_keys_kms_storage_layer_module.kms_key_arn
  attributes_index_list       = var.executions_history_attributes_list
  local_secondary_index_list  = var.executions_history_local_secondary_index_list
  global_secondary_index_list = var.global_secondary_index_list
}

#### GLUE SECURITY CONFIGURATIONS ########
module "aws_data_security_configuration_glue_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                    = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-security-configuration-glue"
  project                   = var.project
  kms_key_arn               = module.aws_security_keys_kms_storage_layer_module.kms_key_arn
  network_connection_enable = true
  vpc_workloads_id          = module.aws_networking_base_vpc_layer_module.id
  glue_connection_requirements = {
    availability_zone = "${var.region}${element(var.availability_zones, 0)}"
    subnet_id         = element(module.aws_networking_base_vpc_layer_module.private_subnet_id_list, 0)
  }
  iam_glue_default_role_enable = false
}

#### GLUE TMDB baseline #############
module "aws_data_processing_job_glue_tmdb_baseline_bronze_table_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source              = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-processing-job-glue"
  create              = true
  project             = var.project
  auto_scaling        = false
  job_name            = format("%s-%s-%s-%s-%s-%s",  var.governance_domain, var.project, "tmdb", "pull","api","glj-brz-${terraform.workspace}")
  job_connections     = [module.aws_data_security_configuration_glue_layer_module.glue_network_connection_name]
  glue_version        = "4.0"
  timeout             = 2880
  max_capacity        = "1.0"
  max_retries         = 1
  execution_property = {
    max_concurrent_runs = 4
  }
  security_configuration  = module.aws_data_security_configuration_glue_layer_module.aws_glue_security_configuration_id
  create_role             = true
  role_name               = format("%s-glj-brz-%s", "tmdb-bl", terraform.workspace)
  bucket_deployment       = module.aws_data_storage_glue_assests_bucket_layer_module.bucket_id
  repository_name         = var.glue_assets_repository_name
  glue_access_databases_tables ={
    "destination_database" = {
      database_name = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
      table_names   = ["*"]
    }
  }
  aws_glue_connection_arn = module.aws_data_security_configuration_glue_layer_module.glue_network_connection_arn
  add_iceberg_config         = false
  script_name = "tmdb_baseline_bronze_job"
  job_parameters = {
    "--source_database"            = module.aws_data_governance_catalog_bronze_ml_database_glue_layer_module.name,
    "--aws_region"                 = data.aws_region.current.name,
    "--source_table"               = "links",
    "--config_parameter"           = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_master_functionality}/PRCS-EXT-011-BRNZ-002/TSKG0001/TSK0001",
    "--secret_name"                = module.aws_security_secrets_ssmstore_layer_module.secret_id,
    "--library-set"                = "analytics",
    #"--encryption-type"            = "sse-s3",
    "--additional-python-modules"  = "aiohttp==3.8.5,Tenacity==8.2.3",
  }
  keys = [module.aws_security_keys_kms_storage_layer_module.kms_key_arn, "arn:aws:kms:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:alias/aws/glue"]
  additional_policies = [{
    name   = "AllowSSMParameterAccessForGlueJob1"
    sid    = "AllowSSMParameterAccessForGlueJob1"
    effect: "Allow"
    actions: [
      "ssm:GetParameter",
      "ssm:GetParameters"
    ]
    resources: ["arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:parameter/${var.project}/${terraform.workspace}/*"]
  },
  {
    name    = "AllowSecretsManagerReadAccess"
    sid     = "AllowSecretsManagerReadAccess"
    effect  = "Allow"
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret"
    ]
    resources = [module.aws_security_secrets_ssmstore_layer_module.secret_arn]
  }]
  command = {
    name           = "pythonshell"
    script_path    = "${var.glue_assets_repository_name}/scripts"
    python_version = "3.9"
  }
  main_source_path = "./${path.root}/dev/glue"
  source_buckets   = [module.aws_storage_bronze_objects_s3_bucket_layer_module.bucket_id]
  destiny_buckets  = [module.aws_storage_bronze_objects_s3_bucket_layer_module.bucket_id]
  sources_types    = ["s3"]
}

module "aws_data_processing_job_glue_tmdb_posters_silver_image_modal_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source              = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-processing-job-glue"
  create              = true
  project             = var.project
  auto_scaling        = false
  job_name            = format("%s-%s-%s-%s-%s-%s",  var.governance_domain, var.project, "tmdb", "pull","posters","glj-slvr-${terraform.workspace}")
  job_connections     = [module.aws_data_security_configuration_glue_layer_module.glue_network_connection_name]
  glue_version        = "4.0"
  timeout             = 2880
  max_capacity        = "1.0"
  max_retries         = 1
  execution_property = {
    max_concurrent_runs = 4
  }
  security_configuration  = module.aws_data_security_configuration_glue_layer_module.aws_glue_security_configuration_id
  create_role             = true
  role_name               = format("%s-glj-slvr-%s", "tmdb-posters", terraform.workspace)
  bucket_deployment       = module.aws_data_storage_glue_assests_bucket_layer_module.bucket_id
  repository_name         = var.glue_assets_repository_name
  glue_access_databases_tables ={
    "source_database" = {
      database_name = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
      table_names   = ["*"]
    }
  }
  aws_glue_connection_arn = module.aws_data_security_configuration_glue_layer_module.glue_network_connection_arn
  add_iceberg_config      = false
  script_name = "tmdb_posters_silver_job"
  job_parameters = {
    "--source_database"            = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name,
    "--source_table"               = "movies",
    "--config_parameter"           = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_master_functionality}/PRCS-EXT-012-SLVR-003/TSKG0002/TSK0002",
    "--aws_region"                 = data.aws_region.current.name,
    "--library-set"                = "analytics",
    #"--encryption-type"            = "sse-s3",
    "--additional-python-modules"  = "aiohttp==3.8.5,Tenacity==8.2.3"
  }
  keys = [module.aws_security_keys_kms_storage_layer_module.kms_key_arn, "arn:aws:kms:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:alias/aws/glue"]
  additional_policies = [{
    name   = "AllowSSMParameterAccessForGlueJob2"
    sid    = "AllowSSMParameterAccessForGlueJob2"
    effect: "Allow"
    actions: [
      "ssm:GetParameter",
      "ssm:GetParameters"
    ]
    resources: ["arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:parameter/${var.project}/${terraform.workspace}/*"]
  }]
  command = {
    name           = "pythonshell"
    script_path    = "${var.glue_assets_repository_name}/scripts"
    python_version = "3.9"
  }
  main_source_path = "./${path.root}/dev/glue"
  source_buckets   = [module.aws_storage_bronze_objects_s3_bucket_layer_module.bucket_id]
  destiny_buckets  = [module.aws_storage_silver_objects_s3_bucket_layer_module.bucket_id]
  sources_types    = ["s3"]
}


module "aws_data_processing_job_glue_create_obt_tables_silver_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source              = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-processing-job-glue"
  create              = true
  project             = var.project
  auto_scaling        = false
  job_name            = format("%s-%s-%s-%s", var.governance_domain, var.project, var.movie-domain,"glj-slvr-${terraform.workspace}")#each.value.raw_api_id, each.value.raw_app_id, "glj-brz") #format("%s-%s", var.project, "glue-payments-merge-job")
  job_connections     = [module.aws_data_security_configuration_glue_layer_module.glue_network_connection_name]
  glue_version        = "4.0"
  timeout             = 2880
  number_of_workers   = 2
  worker_type         = "G.1X"
  max_retries         = 1
  execution_property = {
    max_concurrent_runs = 4
  }
  security_configuration = module.aws_data_security_configuration_glue_layer_module.aws_glue_security_configuration_id
  create_role            = true
  role_name               = format("%s-obt-glj-slvr-%s", "hymmrec", terraform.workspace)
  bucket_deployment       = module.aws_data_storage_glue_assests_bucket_layer_module.bucket_id
  repository_name         = var.glue_assets_repository_name
  datalake_formats        = "iceberg"
  glue_access_databases_tables ={
    "source_tmdb_database" = {
      database_name = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
      table_names   = ["*"]
    },
    "source_ml_database" = {
      database_name = module.aws_data_governance_catalog_bronze_ml_database_glue_layer_module.name
      table_names   = ["*"]
    },
    "destination_database" = {
      database_name = module.aws_data_governance_catalog_silver_database_glue_layer_module.name
      table_names   = ["*"]
    }
  }
  aws_glue_connection_arn = module.aws_data_security_configuration_glue_layer_module.glue_network_connection_arn
  iceberg_datawarehouse_path = "data/${var.movie-domain}/"
  add_iceberg_config         = true
  script_name = "create_obt_silver_job"
  job_parameters = {
    "--enable-spark-ui"            = "true",
    "--enable-job-insights"        = "false",
    "--spark-event-logs-path"      = "s3://${module.aws_data_storage_glue_assests_bucket_layer_module.bucket_id}/sparkHistoryLogs/",
    "--source_ml_database"         = module.aws_data_governance_catalog_bronze_ml_database_glue_layer_module.name,
    "--source_ml_ratings_table"    = "ratings",
    "--source_ml_links_table"      = "links",
    "--source_tmdb_database"       = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name,
    "--source_tmdb_movies_table"   = "movies",
    "--target_database"            = module.aws_data_governance_catalog_silver_database_glue_layer_module.name,
    "--target_ratings_table"       = "cleansed_ratings",
    "--target_movies_table"        = "cleansed_movies",
    "--encryption-type"            = "sse-s3",
    "--obt_config_parameter"       = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_master_functionality}/PRCS-BRNZ-002-SLVR-003/TSKG0003/TSK0003",
    "--aws_region"                 = data.aws_region.current.name,
  }
  keys = [module.aws_security_keys_kms_storage_layer_module.kms_key_arn, "arn:aws:kms:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:alias/aws/glue"]
  additional_policies = [{
    name   = "AllowSSMParameterAccessForGlueJob"
    sid    = "AllowSSMParameterAccessForGlueJob"
    effect: "Allow"
    actions: [
      "ssm:GetParameter",
      "ssm:GetParameters"
    ]
    resources: ["arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:parameter/${var.project}/${terraform.workspace}/*"]
  }]
  command = {
    name           = "glueetl"
    script_path    = "${var.glue_assets_repository_name}/scripts"
    python_version = "3"
  }
  main_source_path = "./${path.root}/dev/glue"
  source_buckets   = [module.aws_storage_bronze_objects_s3_bucket_layer_module.bucket_id]
  destiny_buckets  = [module.aws_storage_silver_objects_s3_bucket_layer_module.bucket_id]
  sources_types    = ["s3"]
}

##### ORCHESTRATION MASTER CONTROL LAMBDAS #########
module "aws_app_compute_lambda_register_master_orchestration_execution_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source             = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-app-compute-lambda"
  subnets_ids        = module.aws_networking_base_vpc_layer_module.private_subnet_id_list
  security_group_ids = [aws_security_group.sg_lambda.id]
  vpc_attach         = true
  lambda_name        = "${var.project}-${var.register_master_orchestration_execution_functionality}"
  lambda_script      = ""
  lambda_runtime     = var.lambda_runtime
  description        = "functionality: omnichannel register master for the ${var.project} project"
  source_code_path   = "./${path.root}/dev/lambdas"
  output_zip_path    = "./${path.root}/dev/artefacts/lambdas"
  lambda_layers      = null
  project            = var.project
  use_existing_role  = false
  add_custom_policy  = true
  custom_policy_path = "${path.root}/extra-policies/lambda"
  parameters_custom_policy_map = {
    region               = data.aws_region.current.name
    account_id           = data.aws_caller_identity.current.account_id
    kms_database_key_arn = module.aws_security_keys_kms_storage_layer_module.kms_key_arn
    dynamodb_table_arn   = module.aws_db_dynamodb_executions_history_table_layer_module.ssmp_dynamodb_table_arn
    ssm_parameter_prefix = "/${var.project}/${terraform.workspace}/${var.register_master_orchestration_execution_functionality}"
  }
  environment_variables = {
    DDB_TABLE_NAME = module.aws_db_dynamodb_executions_history_table_layer_module.ssmp_dynamodb_table_name
  }
}

module "aws_app_compute_lambda_update_master_orchestration_status_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source             = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-app-compute-lambda"
  subnets_ids        = module.aws_networking_base_vpc_layer_module.private_subnet_id_list
  security_group_ids = [aws_security_group.sg_lambda.id]
  vpc_attach         = true
  lambda_name        = "${var.project}-${var.update_master_orchestration_status_functionality}"
  lambda_script      = ""
  lambda_runtime     = var.lambda_runtime
  description        = "functionality: omnichannel register master for the ${var.project} project"
  source_code_path   = "./${path.root}/dev/lambdas"
  output_zip_path    = "./${path.root}/dev/artefacts/lambdas"
  lambda_layers      = null
  project            = var.project
  use_existing_role  = false
  add_custom_policy  = true
  custom_policy_path = "${path.root}/extra-policies/lambda"
  parameters_custom_policy_map = {
    region               = data.aws_region.current.name
    account_id           = data.aws_caller_identity.current.account_id
    kms_database_key_arn = module.aws_security_keys_kms_storage_layer_module.kms_key_arn
    dynamodb_table_arn   = module.aws_db_dynamodb_executions_history_table_layer_module.ssmp_dynamodb_table_arn
    ssm_parameter_prefix = "/${var.project}/${terraform.workspace}/${var.update_master_orchestration_status_functionality}"
  }
  environment_variables = {
    DDB_TABLE_NAME = module.aws_db_dynamodb_executions_history_table_layer_module.ssmp_dynamodb_table_name
  }
}

module "aws_integration_workflow_master_pipeline_step_function_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                 = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-integration-workflow-process-step-function"
  create                 = true
  sfn_publish            = true
  source_definition_path = "${path.root}/state-machine-asls"
  sfn_state_machine_name = "sfn-${var.project}-${var.data_orchestrator_master_functionality}-${terraform.workspace}"
  type                   = "STANDARD"
  vars_map               = {
    registerMasterOrchestrationARN = module.aws_app_compute_lambda_register_master_orchestration_execution_layer_module.lambda_arn
    tmdb_baseline_bronze_glj = module.aws_data_processing_job_glue_tmdb_baseline_bronze_table_layer_module.job_name
    tmdb_posters_silver_glj = module.aws_data_processing_job_glue_tmdb_posters_silver_image_modal_layer_module.job_name
    create_OBT_silver_glj = module.aws_data_processing_job_glue_create_obt_tables_silver_layer_module.job_name
    updateMasterOrchStatusARN = module.aws_app_compute_lambda_update_master_orchestration_status_layer_module.lambda_arn
    tmdb-bl-crawler-glue-name = module.aws-data-governance-metadata-tmdb-crawler-glue-layer-module.name
  }
  tracing_enabled        = true
  custom_policy_path     = "${path.root}/extra-policies/step-function"
  create_role            = true
  create_terraform_style = false # quiero externa y formato json
  logging_configuration = {
    level                  = "ALL"
    include_execution_data = true
  }
  parameters_custom_policy_map = {
    registerMasterOrchestrationARN = module.aws_app_compute_lambda_register_master_orchestration_execution_layer_module.lambda_arn
    updateMasterOrchStatusARN = module.aws_app_compute_lambda_update_master_orchestration_status_layer_module.lambda_arn
    create_obt_silver_job_ARN = module.aws_data_processing_job_glue_create_obt_tables_silver_layer_module.job_arn
    tmdb_baseline_bronze_job_ARN = module.aws_data_processing_job_glue_tmdb_baseline_bronze_table_layer_module.job_arn
    tmdb_posters_silver_job_ARN = module.aws_data_processing_job_glue_tmdb_posters_silver_image_modal_layer_module.job_arn
    region = data.aws_region.current.name
    account_id = data.aws_caller_identity.current.account_id
    prefix = "${var.project}-${var.governance_domain}-*"
  }
  cloudwatch_log_group_name              = "${var.project}-${var.data_orchestrator_master_functionality}-${terraform.workspace}-SMLG"
  cloudwatch_log_group_retention_in_days = 7
  cloudwatch_log_group_kms_key_id        = module.aws_security_keys_integration_layer_module.kms_key_arn
  role_name                              = "${var.project}-${var.data_orchestrator_master_functionality}-sfn-iar-${terraform.workspace}"
  depends_on                             = [aws_kms_key_policy.kms_key_access]
}