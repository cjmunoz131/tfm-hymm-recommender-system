####################################################################################
# Lambda Security Group
####################################################################################
resource "aws_security_group" "sg_lambda" {
  provider               = aws.account1
  description            = "sg para la integracion de las lambdas con rds proxy"
  name                   = "lambda-${var.project}-rds-proxy-integration-sg"
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
  description       = "egress para el sg general de las lambdas"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

################################################################################
# EC2 Client Security Group
################################################################################

# resource "aws_security_group" "client" {
#   provider = aws.account1
#   name     = "${var.project}-client"
#   vpc_id   = module.aws_networking_base_vpc_layer_module.id
# }

# resource "aws_security_group_rule" "client_egress" {
#   provider          = aws.account1
#   security_group_id = aws_security_group.client.id

#   type        = "egress"
#   from_port   = 0
#   to_port     = 0
#   protocol    = -1
#   cidr_blocks = ["0.0.0.0/0"]
# }

# resource "aws_security_group_rule" "client_ingress" {
#   provider          = aws.account1
#   security_group_id = aws_security_group.client.id

#   type        = "ingress"
#   from_port   = 22
#   to_port     = 22
#   protocol    = "tcp"
#   cidr_blocks = ["0.0.0.0/0"]
# }

# ################################################################################
# # EC2 SSH Key Pair
# ################################################################################

# resource "tls_private_key" "emr_private_key" {
#   algorithm = "RSA"
#   rsa_bits  = 4096
# }

# resource "aws_key_pair" "ec2_key_pair" {
#   provider   = aws.account1
#   key_name   = "rds-metadata-kp"
#   public_key = tls_private_key.emr_private_key.public_key_openssh

#   tags = {
#     category = "ec2",
#     resource = "keypair"
#   }

#   provisioner "local-exec" {
#     command = <<EOT
#         echo "${tls_private_key.emr_private_key.private_key_pem}" > ./rds-metadata.pem
#         chmod 400 ./rds-metadata.pem
#         mv ./rds-metadata.pem $HOME/.ssh/
#         EOT
#   }
# }