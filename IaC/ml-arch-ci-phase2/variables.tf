variable "project" {
  type        = string
  description = "Deployment project"
  default     = "udacity-cjmm"
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

variable "vpc_base_cidr_block" {
  description = "VPC base CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "vpc_name" {
  description = "VPC name"
  type        = string
  default     = "olimpica-vpc"
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

variable "execute_inference_breed_dog_classification" {
  type        = string
  description = "Lambda runtime"
  default     = "exec-inf-breeddog-cls"
}

variable "lambda_runtime" {
  type = string
  default = "python3.11"
}

variable "integration_kms_key_name" {
  description = "value"
  type        = string
  default     = "integration-udacity"
}
