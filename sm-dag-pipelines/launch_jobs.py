"""
Script de lanzamiento de jobs individuales en SageMaker.
Permite ejecutar cada paso del pipeline de forma independiente:
- Processing Job (Feature Engineering)
- Processing Job (Generación de Embeddings)
- Training Job (NeuMF / Explainable GMF)
- Hyperparameter Tuning Job
- Batch Transform Job
- Endpoint Deployment

Uso:
    python -m pipelines.launch_jobs --job processing-features
    python -m pipelines.launch_jobs --job training-neumf
    python -m pipelines.launch_jobs --job hpo
    python -m pipelines.launch_jobs --job deploy-endpoint
"""

import argparse
import logging
import os

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.transformer import Transformer
from sagemaker.tuner import (
    ContinuousParameter,
    HyperparameterTuner,
    IntegerParameter,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración base
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ROLE = os.environ.get("SAGEMAKER_ROLE", "")
BUCKET = os.environ.get("S3_BUCKET", "hymm-rec-artifacts")
PREFIX = "hymm-rec"


def get_session():
    return sagemaker.Session(boto_session=boto3.Session(region_name=REGION))


# =========================================================
# 1. PROCESSING: Feature Engineering
# =========================================================
def launch_processing_features():
    """Lanza el Processing Job de feature engineering."""
    session = get_session()

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=ROLE,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    processor.run(
        code="src/processing/feature_engineering.py",
        inputs=[
            ProcessingInput(
                source=f"s3://{BUCKET}/{PREFIX}/data/raw/",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed_data",
                source="/opt/ml/processing/output",
                destination=f"s3://{BUCKET}/{PREFIX}/data/processed/",
            )
        ],
        job_name="hymm-rec-feature-engineering",
    )
    logger.info("Processing Job (Feature Engineering) lanzado.")


# =========================================================
# 2. PROCESSING: Generación de Embeddings (Bedrock)
# =========================================================
def launch_processing_embeddings():
    """Lanza el Processing Job de generación de embeddings con Bedrock."""
    session = get_session()

    # Necesitamos una imagen con boto3 (SKLearn incluye boto3)
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=ROLE,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=session,
    )

    processor.run(
        code="src/processing/generate_embeddings.py",
        inputs=[
            ProcessingInput(
                source=f"s3://{BUCKET}/{PREFIX}/data/raw/",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="embeddings",
                source="/opt/ml/processing/output",
                destination=f"s3://{BUCKET}/{PREFIX}/feature-store/embeddings/",
            )
        ],
        arguments=["--region", REGION, "--catalog-file", "catalog.parquet", "--images-dir", "posters"],
        job_name="hymm-rec-generate-embeddings",
    )
    logger.info("Processing Job (Embeddings Bedrock) lanzado.")


# =========================================================
# 3. TRAINING: NeuMF Two-Tower
# =========================================================
def launch_training_neumf():
    """Lanza el Training Job para el modelo NeuMF."""
    session = get_session()

    estimator = PyTorch(
        entry_point="train_neumf.py",
        source_dir="src/training",
        role=ROLE,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        hyperparameters={
            "epochs": 20,
            "batch-size": 256,
            "lr": 0.001,
            "weight-decay": 1e-3,
            "emb-dim": 64,
            "aws-dim": 1024,
            "patience": 3,
        },
        output_path=f"s3://{BUCKET}/{PREFIX}/training-output/neumf/",
        metric_definitions=[
            {"Name": "train:mse", "Regex": "Train MSE: ([0-9\\.]+)"},
            {"Name": "valid:mse", "Regex": "Valid MSE: ([0-9\\.]+)"},
            {"Name": "valid:rmse_stars", "Regex": "RMSE estrellas: ([0-9\\.]+)"},
        ],
    )

    estimator.fit(
        inputs={"training": f"s3://{BUCKET}/{PREFIX}/data/processed/"},
        job_name="hymm-rec-train-neumf",
    )
    logger.info("Training Job (NeuMF) lanzado.")
    return estimator


# =========================================================
# 4. TRAINING: Explainable GMF
# =========================================================
def launch_training_explainable():
    """Lanza el Training Job para el modelo Explainable GMF."""
    session = get_session()

    estimator = PyTorch(
        entry_point="train_explainable.py",
        source_dir="src/training",
        role=ROLE,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        hyperparameters={
            "epochs": 20,
            "batch-size": 256,
            "lr": 0.001,
            "weight-decay": 1e-3,
            "emb-dim": 64,
            "aws-dim": 1024,
            "patience": 3,
        },
        output_path=f"s3://{BUCKET}/{PREFIX}/training-output/explainable/",
        metric_definitions=[
            {"Name": "train:mse", "Regex": "Train MSE: ([0-9\\.]+)"},
            {"Name": "valid:mse", "Regex": "Valid MSE: ([0-9\\.]+)"},
            {"Name": "valid:rmse_stars", "Regex": "RMSE estrellas: ([0-9\\.]+)"},
        ],
    )

    estimator.fit(
        inputs={"training": f"s3://{BUCKET}/{PREFIX}/data/processed/"},
        job_name="hymm-rec-train-explainable",
    )
    logger.info("Training Job (Explainable GMF) lanzado.")
    return estimator


# =========================================================
# 5. HYPERPARAMETER TUNING JOB
# =========================================================
def launch_hpo():
    """Lanza un Hyperparameter Tuning Job."""
    session = get_session()

    estimator = PyTorch(
        entry_point="train_neumf.py",
        source_dir="src/training",
        role=ROLE,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        hyperparameters={
            "epochs": 15,
            "aws-dim": 1024,
            "patience": 3,
        },
        output_path=f"s3://{BUCKET}/{PREFIX}/training-output/hpo/",
        metric_definitions=[
            {"Name": "valid:mse", "Regex": "Valid MSE: ([0-9\\.]+)"},
            {"Name": "valid:rmse_stars", "Regex": "RMSE estrellas: ([0-9\\.]+)"},
        ],
    )

    hyperparameter_ranges = {
        "lr": ContinuousParameter(1e-4, 1e-2, scaling_type="Logarithmic"),
        "batch-size": IntegerParameter(128, 512),
        "emb-dim": IntegerParameter(32, 128),
        "weight-decay": ContinuousParameter(1e-5, 1e-2, scaling_type="Logarithmic"),
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="valid:mse",
        objective_type="Minimize",
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=20,
        max_parallel_jobs=3,
        strategy="Bayesian",
    )

    tuner.fit(
        inputs={"training": f"s3://{BUCKET}/{PREFIX}/data/processed/"},
        job_name="hymm-rec-hpo",
    )
    logger.info("HPO Job lanzado (20 trials, 3 en paralelo).")
    return tuner


# =========================================================
# 6. DEPLOY ENDPOINT
# =========================================================
def launch_deploy_endpoint(model_data_s3: str = None):
    """Despliega un endpoint de inferencia en tiempo real."""
    session = get_session()

    if not model_data_s3:
        model_data_s3 = f"s3://{BUCKET}/{PREFIX}/training-output/explainable/model.tar.gz"

    model = PyTorchModel(
        model_data=model_data_s3,
        role=ROLE,
        entry_point="inference.py",
        source_dir="src/inference",
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g4dn.xlarge",
        endpoint_name="hymm-rec-explainable-endpoint",
    )
    logger.info(f"Endpoint desplegado: {predictor.endpoint_name}")
    return predictor


# =========================================================
# 7. BATCH TRANSFORM
# =========================================================
def launch_batch_transform(model_data_s3: str = None):
    """Lanza un Batch Transform Job para predicciones offline masivas."""
    session = get_session()

    if not model_data_s3:
        model_data_s3 = f"s3://{BUCKET}/{PREFIX}/training-output/explainable/model.tar.gz"

    model = PyTorchModel(
        model_data=model_data_s3,
        role=ROLE,
        entry_point="inference.py",
        source_dir="src/inference",
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
    )

    transformer = model.transformer(
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path=f"s3://{BUCKET}/{PREFIX}/batch-predictions/",
        strategy="MultiRecord",
        max_payload=6,
    )

    transformer.transform(
        data=f"s3://{BUCKET}/{PREFIX}/data/batch-input/",
        content_type="application/json",
        split_type="Line",
        job_name="hymm-rec-batch-transform",
    )
    logger.info("Batch Transform Job lanzado.")
    return transformer


# =========================================================
# 8. LLM FINE-TUNING (LLaMA 3.1)
# =========================================================
def launch_llm_finetuning():
    """Lanza el Training Job de fine-tuning de LLaMA 3.1 con QLoRA."""
    from sagemaker.huggingface import HuggingFace

    session = get_session()

    hf_estimator = HuggingFace(
        entry_point="train_llama_qlora.py",
        source_dir="src/explainability",
        role=ROLE,
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        transformers_version="4.44",
        pytorch_version="2.3",
        py_version="py310",
        sagemaker_session=session,
        hyperparameters={
            "model-id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "epochs": 3,
            "lora-r": 16,
            "lora-alpha": 32,
            "lora-dropout": 0.05,
            "learning-rate": 2e-4,
            "batch-size": 1,
            "gradient-accumulation": 8,
            "seed": 42,
        },
        output_path=f"s3://{BUCKET}/{PREFIX}/training-output/llm/",
        environment={
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
        },
    )

    hf_estimator.fit(
        inputs={"training": f"s3://{BUCKET}/{PREFIX}/data/llm/"},
        job_name="hymm-rec-llama-finetuning",
    )
    logger.info("LLM Fine-Tuning Job lanzado.")
    return hf_estimator


# =========================================================
# MAIN
# =========================================================
JOB_MAP = {
    "processing-features": launch_processing_features,
    "processing-embeddings": launch_processing_embeddings,
    "training-neumf": launch_training_neumf,
    "training-explainable": launch_training_explainable,
    "hpo": launch_hpo,
    "deploy-endpoint": launch_deploy_endpoint,
    "batch-transform": launch_batch_transform,
    "llm-finetuning": launch_llm_finetuning,
}


def main():
    parser = argparse.ArgumentParser(description="Lanzador de jobs AWS SageMaker")
    parser.add_argument(
        "--job",
        type=str,
        required=True,
        choices=list(JOB_MAP.keys()),
        help="Tipo de job a lanzar",
    )
    parser.add_argument("--model-data", type=str, default=None, help="S3 URI del model.tar.gz (para deploy/batch)")
    args = parser.parse_args()

    logger.info(f"Lanzando job: {args.job}")

    if args.job in ("deploy-endpoint", "batch-transform") and args.model_data:
        JOB_MAP[args.job](args.model_data)
    else:
        JOB_MAP[args.job]()


if __name__ == "__main__":
    main()
