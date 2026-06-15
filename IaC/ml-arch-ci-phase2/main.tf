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
  ]
}

module "aws_security_keys_integration_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source   = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-security-keys-kms"
  key_name = var.integration_kms_key_name
}

# Storage creation
module "aws_storage_landing_objects_s3_bucket_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-storage-objects-s3"
  bucket_name           = "${var.project}-machine-learning-tests"
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
  project = "udacity-nanodegree"
  vpc_id = module.aws_networking_base_vpc_layer_module.id
  private_subnet_id = module.aws_networking_base_vpc_layer_module.private_subnet_id_list[0]
  sagemaker_instance_name = "udacity-ml"
  sagemaker_instance_type = "ml.t3.medium"
  platform_identifier = "notebook-al2-v3"
  direct_internet_access = "Disabled"
  volume_size = 50
  target_buckets = [module.aws_storage_landing_objects_s3_bucket_layer_module.bucket_id]
  repo_url = "https://github.com/cjmunoz131/udacity-aws-ml-nanodegree.git"
  env_name = "ml-dev"
  python_version = "3.11"
}

resource "aws_iam_role" "client" {
  provider = aws.account1
  name     = "iar-udacity-ml-training-for-instance-profile"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_instance_profile" "this" {
  provider = aws.account1
  name     = "udacity-ml-training-ip"
  role     = aws_iam_role.client.name
}

resource "aws_iam_role_policy_attachment" "client" {
  provider   = aws.account1
  role       = aws_iam_role.client.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_security_group" "client" {
  provider = aws.account1
  name     = "udacity-ml-training-sg"
  vpc_id   = module.aws_networking_base_vpc_layer_module.id
}

resource "aws_security_group_rule" "client_ingress" {
  provider          = aws.account1
  security_group_id = aws_security_group.client.id

  type        = "ingress"
  from_port   = 22
  to_port     = 22
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
}

resource "aws_security_group_rule" "client_egress" {
  provider          = aws.account1
  security_group_id = aws_security_group.client.id

  type        = "egress"
  from_port   = 0
  to_port     = 0
  protocol    = -1
  cidr_blocks = ["0.0.0.0/0"]
}

resource "tls_private_key" "emr_private_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "ec2_key_pair" {
  provider   = aws.account1
  key_name   = "s2smulticloud-kp"
  public_key = tls_private_key.emr_private_key.public_key_openssh

  tags = {
    category = "ec2",
    resource = "keypair"
  }

  provisioner "local-exec" {
    command = <<EOT
        echo "${tls_private_key.emr_private_key.private_key_pem}" > ./s2smulticloud-kp.pem
        chmod 400 ./s2smulticloud-kp.pem
        mv ./s2smulticloud-kp.pem $HOME/.ssh/
        EOT
  }
}

module "machine-learning-training-instance-ec2-module" {
  providers = {
    aws.main = aws.account1
  }
  source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-app-compute-virtual-machine-ec2"
  
  project     = var.project
  environment = terraform.workspace
  
  instances_config = {
    # Servidor de aplicación
    app_server = {
      iam_instance_profile  = aws_iam_instance_profile.this.name
      associate_public_ip_address = true
      ami           = "ami-09570605cb6ed4f72"
      instance_type = "m5.2xlarge"
      subnet_id     = module.aws_networking_base_vpc_layer_module.public_subnet_id_list[0]
      
      security = {
        security_group_ids = [aws_security_group.client.id]
        key_name           = aws_key_pair.ec2_key_pair.key_name
      }
      
      root_block_device = {
        volume_size = 100
        volume_type = "gp3"
        encrypted   = true
      }
      
      additional_volumes = {
        data = {
          device_name = "/dev/sdf"
          size        = 20
          type        = "gp3"
          encrypted   = true
        }
      }
    }
  }
}

resource "aws_security_group" "sg_lambda" {
  provider               = aws.account1
  description            = "sg para la integracion de las lambdas con rds proxy"
  name                   = "lambda-${var.project}-integration-sg"
  revoke_rules_on_delete = true
  vpc_id                 = module.aws_networking_base_vpc_layer_module.id

  lifecycle {
    ignore_changes = [
      vpc_id
    ]
  }
}

resource "aws_vpc_security_group_egress_rule" "sg_egress_default_lambda" {
  provider          = aws.account1
  security_group_id = aws_security_group.sg_lambda.id
  description       = "default egress in sg of the lambdas"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

module "aws_sagemaker_breed_dog_classification_endpoint_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source   = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-governance-model-serving-endpoint-sagemaker"
  enable_sagemaker_endpoint = true
  project                  = var.project
  endpoint-name = "breed-dog-class-endpoint"
  enable_sagemaker_endpoint_configuration = true
  sagemaker_endpoint_configuration_kms_key_arn = module.aws_security_keys_integration_layer_module.kms_key_arn
  sagemaker_endpoint_configuration_production_variants = [
    {
      variant_name          = "AllTraffic"
      model_name            = "pytorch-inference-2026-04-17-23-33-53-301"
      initial_instance_count = 1
      instance_type         = "ml.m5.large"
    }
  ]
  endpoint_configuration_name = "breed-dog-class-ep-config"
  endpoint_instance_min_capacity = 1
  endpoint_instance_max_capacity = 2
  scale_in_cooldown = 300
  scale_out_cooldown = 60
  sagemaker_variant_name = "AllTraffic"
  invocations_target_value = 100
  enable_sagemaker_default_autoscaling = true
}

module "aws_app_compute_lambda_execute_inference_breed_dog_classification_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                    = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-app-compute-lambda"
  subnets_ids               = module.aws_networking_base_vpc_layer_module.private_subnet_id_list
  vpc_attach                = true
  security_group_ids        = [aws_security_group.sg_lambda.id]
  lambda_name               = "${var.project}-${var.execute_inference_breed_dog_classification}"
  lambda_script             = ""
  lambda_runtime            = var.lambda_runtime
  description               = "functionality: omnichannel generator for the master execution plan in the ${var.project} project"
  source_code_path          = "./${path.root}/dev/lambdas"
  output_zip_path           = "./${path.root}/dev/artefacts/lambdas"
  project                   = var.project
  use_existing_role         = false
  add_custom_policy         = true
  custom_policy_path        = "${path.root}/extra-policies/lambda"
  create_layers             = false
  lambda_layers             = null
  parameters_custom_policy_map = {
    region               = data.aws_region.current.name
    account_id           = data.aws_caller_identity.current.account_id
    endpoint_name        = module.aws_sagemaker_breed_dog_classification_endpoint_layer_module.endpoint_name
    kms_key_arn          = module.aws_security_keys_integration_layer_module.kms_key_arn
  }
  environment_variables = {
    endpoint_Name = module.aws_sagemaker_breed_dog_classification_endpoint_layer_module.endpoint_name
  }
  enable_lambda_autoscaling = true
  enable_lambda_alias = true
  lambda_min_capacity = 1
  lambda_max_capacity = 3
  capacity_target_value = 0.7
  scale_in_cooldown = 300
  scale_out_cooldown = 60
  publish_new_version = true
}