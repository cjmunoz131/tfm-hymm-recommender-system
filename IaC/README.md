# Infrastructure

Directorio reservado para código de infraestructura como código (IaC).

## Opciones recomendadas

- **AWS CDK (Python)**: Para definir los recursos de forma programática
- **CloudFormation**: Templates YAML/JSON para los servicios
- **Terraform**: Si se prefiere un enfoque multi-cloud

## Recursos a provisionar

| Recurso | Descripción |
|---------|-------------|
| S3 Bucket | Almacenamiento de datos, modelos y artefactos |
| SageMaker Role | Rol IAM con permisos para S3, Bedrock, ECR |
| OpenSearch Domain | Clúster con k-NN habilitado |
| VPC / Subnets | Red privada para SageMaker y OpenSearch |
| SageMaker Endpoints | Configuración de endpoints de inferencia |
| Glue Jobs | Jobs de ETL para ingesta de datos TMDB |
| ECR Repository | Imágenes Docker custom (si se requieren) |

## Configuración mínima requerida

```bash
# Variables de entorno necesarias
export AWS_DEFAULT_REGION=us-east-1
export SAGEMAKER_ROLE=arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole
export S3_BUCKET=hymm-rec-artifacts
export OPENSEARCH_DOMAIN=https://search-hymm-rec-xxx.us-east-1.es.amazonaws.com
```
