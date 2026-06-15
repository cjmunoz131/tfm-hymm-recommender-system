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
  default     = "tfm"
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

variable "domain_id" {
  type    = string
  default = "01"
}

########################

variable "vpc_base_cidr_block" {
  description = "VPC base CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "vpc_name" {
  description = "VPC name"
  type        = string
  default     = "hymm-rec-vpc"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["a", "b"]
}

variable "subnet_private_cidr_blocks" {
  description = "List of private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "subnet_public_cidr_blocks" {
  description = "List of public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.3.0/24", "10.0.4.0/24"]
}

variable "subnet_database_cidr_blocks" {
  type        = list(string)
  description = "cidr block for vpc"
  default     = ["10.0.5.0/24", "10.0.6.0/24"]
}

variable "lambda_runtime" {
  type = string
  default = "python3.11"
}

variable "integration_kms_key_name" {
  description = "value"
  type        = string
  default     = "kms-int-hymm-rec"
}

variable "storage_kms_key_name" {
  description = "value"
  type        = string
  default     = "kms-stg-hymm-rec"
}

### Event bridge scheduler expression ######
variable "event_bridge_scheduler_expression" {
  type    = string
  default = "rate(1 day)"
}

variable "data_orchestrator_trigger_functionality" {
  type    = string
  default = "data-orch-trigger"
}

variable "data_orchestrator_master_functionality" {
  type    = string
  default = "master-data-pipeline"
}
###### DynamoDB executions history ################
variable "dynamodb_executions_history_table_name" {
  description = "DynamoDB table name for process images"
  type        = string
  default     = "pipelines-executions-history"
}

variable "executions_history_attributes_list" {
  description = "List of attributes for the executions_history table"
  type = list(object({
    key_name      = string,
    key_data_type = string
  }))
  default = [
    {
      key_data_type = "S"
      key_name      = "pipeline_id"
    },
    {
      key_data_type = "N"
      key_name      = "execution_time"
    },
    {
      key_data_type = "S"
      key_name      = "status"
    },
    {
      key_data_type = "S"
      key_name      = "domain_project"
    },
    {
      key_data_type = "S"
      key_name      = "correlation-id"
    }
  ]
}

variable "executions_history_local_secondary_index_list" {
  description = "List of local secondary indexes for the microservice"
  type = set(object({
    range_key          = string,
    lsi_projection     = string
    name               = string
    non_key_attributes = optional(set(string))
  }))
  default = [{
    range_key      = "status",
    lsi_projection = "ALL",
    name           = "search_by_pipelineid_and_status"
  }]
}

variable "global_secondary_index_list" {
  description = "List of global secondary indexes for the microservice"
  type = set(object({
    hash_key           = string,
    range_key          = optional(string),
    gsi_projection     = string
    name               = string
    non_key_attributes = optional(set(string))
  }))
  default = [{
    hash_key       = "correlation-id",
    range_key      = "pipeline_id",
    gsi_projection = "ALL",
    name           = "correlation_id_sk"
    },
    {
      hash_key       = "domain_project",
      range_key      = "status",
      gsi_projection = "ALL",
      name           = "domain_project_sk"
  }]
}


##### DATA CATALOG
variable "governance_domain" {
  type = string
  description = "governance domain"
  default = "tfm"
}

variable "application-domain" {
  type = string
  description = "application domain"
  default = "tmdb"
}

variable "ml-application-domain" {
  type = string
  description = "application domain"
  default = "ml"
}

variable "movie-domain" {
  type = string
  description = "movie data domain in silver"
  default = "obt_movie_affinity"
}

#### GLUE JOBS #######
variable "glue_assets_repository_name" {
  description = "Glue assets repository name"
  type        = string
  default     = "glue-repository"
}

### LAMBDAS ORCH CONTROL ####
variable "register_master_orchestration_execution_functionality" {
  type    = string
  default = "register-master-orch-execution"
}

variable "update_master_orchestration_status_functionality" {
  type    = string
  default = "update-master-orch-status"
}