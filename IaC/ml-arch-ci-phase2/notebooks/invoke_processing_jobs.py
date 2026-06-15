"""
==============================================================================
HYMM-REC MLOps Pipeline: Invocación de Processing Jobs desde SageMaker Notebook
==============================================================================
Este script simula las celdas de un notebook para ejecutar los processing jobs
del flujo MLOps del sistema recomendador híbrido.

Flujo:
  1. Processing Job 1: Feature Engineering + Embeddings Generation
  2. Processing Job 2: Dataset Preparation (K-Core + Splits)

Pre-requisitos:
  - SageMaker Execution Role con permisos a S3, Bedrock, Feature Store
  - Buckets: silver (input), gold (feature store), platinum (training data)
  - Tablas Iceberg en Silver ya materializadas por el pipeline de datos
  - Posters descargados en Silver
"""

# ==============================================================================
# CELDA 1: CONFIGURACIÓN
# ==============================================================================
import boto3
import sagemaker
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.spark.processing import PySparkProcessor
from sagemaker import get_execution_role

# Configuración base
REGION = "us-east-1"
ROLE = get_execution_role()
SESSION = sagemaker.Session()

# Buckets
BRONZE_BUCKET = "hymmrec-dilkehousebronze01"
SILVER_BUCKET = "hymmrec-dilkehousesilver01"
GOLD_BUCKET = "hymmrec-dilkehousegold01"
PLATINUM_BUCKET = "hymmrec-dilkhouseplatinum01"  # Para datos de entrenamiento

# Paths S3 Silver (input)
S3_SILVER_RATINGS = f"s3://{SILVER_BUCKET}/data/obt_movie_affinity/cleansed_ratings/"
S3_SILVER_MOVIES = f"s3://{SILVER_BUCKET}/data/obt_movie_affinity/cleansed_movies/"
S3_SILVER_POSTERS = f"s3://{SILVER_BUCKET}/data/imv_movie_affinity/movie_posters/"

# Paths S3 Gold (feature store / embeddings)
S3_GOLD_EMBEDDINGS = f"s3://{GOLD_BUCKET}/feature-store/embeddings/"
S3_GOLD_ENCODERS = f"s3://{GOLD_BUCKET}/feature-store/encoders/"
S3_GOLD_FEATURES = f"s3://{GOLD_BUCKET}/feature-store/interactions/"

# Paths S3 Platinum (training datasets)
S3_PLATINUM_DATASETS = f"s3://{PLATINUM_BUCKET}/datasets/"

# Scripts
PROCESSING_JOB_1_SCRIPT = "processing-feature-eng-job.py"
PROCESSING_JOB_2_SCRIPT = "processing-prepare-data-splits.py"
SCRIPTS_S3_PREFIX = f"s3://{GOLD_BUCKET}/sagemaker-scripts/"

print(f"Role: {ROLE}")
print(f"Region: {REGION}")
print(f"Silver: {SILVER_BUCKET}")
print(f"Gold: {GOLD_BUCKET}")
print(f"Platinum: {PLATINUM_BUCKET}")


# ==============================================================================
# CELDA 2: SUBIR SCRIPTS A S3
# ==============================================================================
# Subir los scripts de processing a S3 (o usar directamente desde local)
import os

local_scripts_path = "../dev/"
s3_client = boto3.client("s3")

for script in [PROCESSING_JOB_1_SCRIPT, PROCESSING_JOB_2_SCRIPT]:
    local_path = os.path.join(local_scripts_path, script)
    s3_key = f"sagemaker-scripts/{script}"
    s3_client.upload_file(local_path, GOLD_BUCKET, s3_key)
    print(f"Uploaded: {local_path} → s3://{GOLD_BUCKET}/{s3_key}")


# ==============================================================================
# CELDA 3: PROCESSING JOB 1 — Feature Engineering + Embeddings
# ==============================================================================
# Este job:
#   - Lee ratings + movies de Silver (Iceberg/Parquet)
#   - Aplica Label Encoding, Multi-Hot, Rating Scaling (PySpark + sklearn)
#   - Genera embeddings multimodales con Amazon Bedrock Nova (82K películas)
#   - Ingesta features en Feature Store offline
#   - Guarda embeddings_catalog.pkl + encoders.pkl en Gold

print("=" * 60)
print("PROCESSING JOB 1: Feature Engineering + Embeddings")
print("=" * 60)

# Usamos PySparkProcessor para la parte de Spark
# La generación de embeddings (Bedrock) se ejecuta en el driver
pyspark_processor = PySparkProcessor(
    role=ROLE,
    instance_type="ml.m5.xlarge",
    instance_count=2,  # 1 driver + 1 worker
    framework_version="3.3",
    py_version="py39",
    sagemaker_session=SESSION,
    base_job_name="hymmrec-feature-engineering",
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "feature-engineering"},
    ],
)

# Ejecutar Processing Job 1
pyspark_processor.run(
    submit_app=f"{local_scripts_path}/{PROCESSING_JOB_1_SCRIPT}",
    submit_py_files=[],
    arguments=[
        "--region", REGION,
        "--feature-group-name", "hymmrec-feature-interactions",
    ],
    inputs=[
        ProcessingInput(
            source=S3_SILVER_RATINGS,
            destination="/opt/ml/processing/input/ratings",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
        ProcessingInput(
            source=S3_SILVER_MOVIES,
            destination="/opt/ml/processing/input/movies",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
        ProcessingInput(
            source=S3_SILVER_POSTERS,
            destination="/opt/ml/processing/input/posters",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/embeddings",
            destination=S3_GOLD_EMBEDDINGS,
            output_name="embeddings",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/encoders",
            destination=S3_GOLD_ENCODERS,
            output_name="encoders",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/feature_interactions",
            destination=S3_GOLD_FEATURES,
            output_name="feature_interactions",
        ),
    ],
    spark_event_logs_s3_uri=f"s3://{GOLD_BUCKET}/spark-logs/job1/",
    logs=True,
    wait=True,
)

print("✅ Processing Job 1 completado.")


# ==============================================================================
# CELDA 4: VERIFICACIÓN INTERMEDIA (OPCIONAL)
# ==============================================================================
# Verificar que los artefactos se generaron correctamente

import pickle

# Verificar encoders
s3 = boto3.resource("s3")
obj = s3.Object(GOLD_BUCKET, "feature-store/encoders/encoders.pkl")
encoders = pickle.loads(obj.get()["Body"].read())
print(f"Encoders verificados:")
print(f"  le_user: {len(encoders['le_user'].classes_):,} usuarios")
print(f"  le_item: {len(encoders['le_item'].classes_):,} items")
print(f"  mlb: {len(encoders['mlb'].classes_)} géneros")

# Verificar embeddings (solo tamaño)
obj_emb = s3.Object(GOLD_BUCKET, "feature-store/embeddings/embeddings_catalog.pkl")
emb_size_mb = obj_emb.content_length / (1024 * 1024)
print(f"\nEmbeddings catalog: {emb_size_mb:.1f} MB")


# ==============================================================================
# CELDA 5: PROCESSING JOB 2 — Dataset Preparation (K-Core + Splits)
# ==============================================================================
# Este job:
#   - Lee features del Feature Store offline (o fallback Parquet de Gold)
#   - Aplica K-Core filtering (users≥20, items≥10) para eliminar ruido
#   - Split temporal-estratificado: train 80% / val 10% / test 10%
#   - Persiste cold-starts (datos descartados por k-core)
#   - Copia encoders + embeddings al Platinum bucket

print("\n" + "=" * 60)
print("PROCESSING JOB 2: Dataset Preparation (K-Core + Splits)")
print("=" * 60)

pyspark_processor_2 = PySparkProcessor(
    role=ROLE,
    instance_type="ml.m5.xlarge",
    instance_count=2,
    framework_version="3.3",
    py_version="py39",
    sagemaker_session=SESSION,
    base_job_name="hymmrec-dataset-splits",
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "dataset-preparation"},
    ],
)

pyspark_processor_2.run(
    submit_app=f"{local_scripts_path}/{PROCESSING_JOB_2_SCRIPT}",
    submit_py_files=[],
    arguments=[
        "--min-user-interactions", "20",
        "--min-item-interactions", "10",
        "--kcore-iterations", "5",
    ],
    inputs=[
        ProcessingInput(
            source=S3_GOLD_FEATURES,
            destination="/opt/ml/processing/input/features",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
        ProcessingInput(
            source=S3_GOLD_ENCODERS,
            destination="/opt/ml/processing/input/encoders",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
        ProcessingInput(
            source=S3_GOLD_EMBEDDINGS,
            destination="/opt/ml/processing/input/embeddings",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/platinum",
            destination=S3_PLATINUM_DATASETS,
            output_name="platinum_datasets",
        ),
    ],
    spark_event_logs_s3_uri=f"s3://{GOLD_BUCKET}/spark-logs/job2/",
    logs=True,
    wait=True,
)

print("✅ Processing Job 2 completado.")


# ==============================================================================
# CELDA 6: VERIFICACIÓN FINAL
# ==============================================================================
# Listar lo que quedó en Platinum (listo para Training Job)

import subprocess

print("\n📦 Contenido del Platinum bucket (datasets de entrenamiento):")
print("-" * 60)

for prefix in ["train/", "val/", "test/", "cold-starts/", "encoders/", "embeddings/"]:
    result = subprocess.run(
        ["aws", "s3", "ls", f"{S3_PLATINUM_DATASETS}{prefix}", "--summarize"],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")
    if lines and lines[-1].strip():
        # Mostrar solo el resumen (última línea con Total Size)
        summary = [l for l in lines if "Total Size" in l or "Total Objects" in l]
        print(f"  {prefix:<20} → {' | '.join(summary)}")
    else:
        print(f"  {prefix:<20} → (vacío)")

print("\n" + "=" * 60)
print("🎯 PRÓXIMO PASO: SageMaker Training Job")
print("   Input: s3://platinum-bucket/datasets/")
print("   Script: train_neumf.py o train_explainable.py")
print("=" * 60)
