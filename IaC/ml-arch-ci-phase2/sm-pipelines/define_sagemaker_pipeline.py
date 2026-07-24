"""
==============================================================================
HYMM-REC MLOps: SageMaker Pipeline con HPO (DAG Completo - CI/CD Ready)
==============================================================================
Script que genera la definición del SageMaker Pipeline para despliegue
via Terraform + Azure DevOps CI/CD.

DAG del Pipeline:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  FeatureEngineering (PySpark) ──┐                                        │
  │                                 ├──► DatasetPreparation                  │
  │  EmbeddingsGeneration (SKLearn) ┘         │                              │
  │                                           ▼                              │
  │                          ┌── HPO Regression ──► Training Regression ──┐  │
  │                          │                                            │  │
  │                          └── HPO TwoHeads  ──► Training TwoHeads  ───┘  │
  │                                                       │                  │
  │                                                       ▼                  │
  │                                              ModelEvaluation              │
  │                                                       │                  │
  │                                                       ▼                  │
  │                                            QualityGateCheck              │
  │                                              ┌────┴────┐                 │
  │                                         Pass │         │ Fail            │
  │                                              ▼         ▼                 │
  │                                     RegisterModel   FailStep             │
  │                                              │                           │
  │                                              ▼                           │
  │                                      ModelPackaging                      │
  └──────────────────────────────────────────────────────────────────────────┘

Despliegue (CI/CD):
  1. Terraform sube scripts a S3 (aws_s3_object)
  2. Terraform aplica infra (Feature Store, Model Registry, etc.)
  3. Este script se ejecuta: python define_sagemaker_pipeline.py --upsert
  4. Se inicia el pipeline: python define_sagemaker_pipeline.py --execute

NOTA: Todos los scripts se referencian desde S3 (no paths locales).
      Terraform se encarga de sincronizar dev/ → S3.
"""

import argparse
import json
import logging
import os
from typing import Optional

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.tuner import (
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep, TuningStep

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ==============================================================================

PIPELINE_NAME = "hymmrec-principal-model-sm-pipeline-default"

PIPELINE_DESCRIPTION = (
    "Pipeline MLOps end-to-end para HYMM-REC. Incluye Feature Engineering, "
    "Embeddings, Data Splits, HPO (Regresión + Two-Heads), Training Final, "
    "Evaluation, Quality Gate, Model Registry y Packaging."
)

# --- Buckets ---
DEFAULT_SILVER_BUCKET = "hymmrec-dilkehousesilver01"
DEFAULT_GOLD_BUCKET = "hymmrec-dilkehousegold01"
DEFAULT_PLATINUM_BUCKET = "hymmrec-sagemaker-assets"

# --- Paths locales para scripts (el SDK los empaqueta y sube automáticamente) ---
# Al hacer --upsert, el SDK crea sourcedir.tar.gz y lo sube a S3 por nosotros.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSING_SCRIPTS_LOCAL = os.path.join(SCRIPT_DIR, "..", "dev", "feng-data-preparing")
TRAINING_SCRIPTS_LOCAL = os.path.join(SCRIPT_DIR, "..", "dev", "training")
EVALUATION_SCRIPTS_LOCAL = os.path.join(SCRIPT_DIR, "..", "dev", "evaluation")
INFERENCE_SCRIPTS_LOCAL = os.path.join(SCRIPT_DIR, "..", "dev", "inference")

# S3 paths para referencia (Processing Jobs que usan 'code=' con rutas S3 directas)
SCRIPTS_S3_BASE = f"s3://{DEFAULT_PLATINUM_BUCKET}/sagemaker-scripts"
PROCESSING_SCRIPTS_S3 = f"{SCRIPTS_S3_BASE}/feng-data-preparing"
EVALUATION_SCRIPTS_S3 = f"{SCRIPTS_S3_BASE}/evaluation"
INFERENCE_SCRIPTS_S3 = f"{SCRIPTS_S3_BASE}/inference"

# --- Instancias por defecto ---
DEFAULT_PROCESSING_INSTANCE = "ml.m5.xlarge"
DEFAULT_EMBEDDINGS_INSTANCE = "ml.t3.medium"
DEFAULT_SPLITS_INSTANCE = "ml.m5.large"
DEFAULT_TRAINING_INSTANCE = "ml.m5.2xlarge"#"ml.g4dn.xlarge"
DEFAULT_HPO_INSTANCE = "ml.m5.2xlarge"#"ml.g4dn.xlarge"
DEFAULT_EVAL_INSTANCE = "ml.m5.2xlarge"#"ml.g4dn.xlarge"
DEFAULT_PACKAGING_INSTANCE = "ml.m5.large"

# --- HPO Config ---
HPO_MAX_JOBS = 12
HPO_MAX_PARALLEL_JOBS = 3
HPO_STRATEGY = "Bayesian"

# --- Cache Config ---
# Permite reusar resultados de steps anteriores si inputs/params no cambiaron.
# expire_after: tiempo máximo de cache (30 días). Si los datos no cambian,
# el pipeline salta steps ya ejecutados y va directo a los que necesitan re-run.
CACHE_CONFIG = CacheConfig(enable_caching=True, expire_after="P30D")

# --- Model Registry ---
MODEL_PACKAGE_GROUP_NAME = "hymmrec-multimodal-recommender"


# ==============================================================================
# PIPELINE PARAMETERS
# ==============================================================================


def define_pipeline_parameters():
    """Define parámetros configurables en runtime del pipeline."""

    # Infraestructura
    region = ParameterString(name="Region", default_value="us-east-1")
    role_arn = ParameterString(name="RoleArn", default_value="")

    # Buckets
    silver_bucket = ParameterString(name="SilverBucket", default_value=DEFAULT_SILVER_BUCKET)
    gold_bucket = ParameterString(name="GoldBucket", default_value=DEFAULT_GOLD_BUCKET)
    platinum_bucket = ParameterString(name="PlatinumBucket", default_value=DEFAULT_PLATINUM_BUCKET)

    # Instancias
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value=DEFAULT_PROCESSING_INSTANCE
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value=DEFAULT_TRAINING_INSTANCE
    )
    eval_instance_type = ParameterString(
        name="EvalInstanceType", default_value=DEFAULT_EVAL_INSTANCE
    )

    # Training Hyperparams (usados en Training Final post-HPO)
    epochs = ParameterInteger(name="Epochs", default_value=30)
    batch_size = ParameterInteger(name="BatchSize", default_value=256)
    learning_rate = ParameterFloat(name="LearningRate", default_value=0.001)
    emb_dim = ParameterInteger(name="EmbeddingDim", default_value=64)
    dropout = ParameterFloat(name="Dropout", default_value=0.3)

    # HPO
    hpo_max_jobs = ParameterInteger(name="HPOMaxJobs", default_value=HPO_MAX_JOBS)
    hpo_max_parallel = ParameterInteger(name="HPOMaxParallel", default_value=HPO_MAX_PARALLEL_JOBS)

    # Data Splits
    min_user_interactions = ParameterInteger(name="MinUserInteractions", default_value=5)
    min_item_interactions = ParameterInteger(name="MinItemInteractions", default_value=5)
    train_ratio = ParameterFloat(name="TrainRatio", default_value=0.80)
    val_ratio = ParameterFloat(name="ValRatio", default_value=0.10)

    # Quality Gate
    max_rmse_threshold = ParameterFloat(name="MaxRMSEThreshold", default_value=2.0)

    return {
        "region": region,
        "role_arn": role_arn,
        "silver_bucket": silver_bucket,
        "gold_bucket": gold_bucket,
        "platinum_bucket": platinum_bucket,
        "processing_instance_type": processing_instance_type,
        "training_instance_type": training_instance_type,
        "eval_instance_type": eval_instance_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "emb_dim": emb_dim,
        "dropout": dropout,
        "hpo_max_jobs": hpo_max_jobs,
        "hpo_max_parallel": hpo_max_parallel,
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "max_rmse_threshold": max_rmse_threshold,
    }


# ==============================================================================
# STEP 1: Feature Engineering (PySpark Processing)
# ==============================================================================


def create_step_feature_engineering(params, role, session):
    """
    Processing Job 1: Feature Engineering + Feature Store Ingestion.
    - Lee ratings + movies de Silver (Iceberg/Parquet)
    - Label Encoding, Multi-Hot, Rating Scaling (PySpark + sklearn)
    - Ingesta features en Feature Store offline
    - Guarda encoders.pkl en Platinum
    """
    pyspark_processor = PySparkProcessor(
        role=role,
        instance_type=DEFAULT_PROCESSING_INSTANCE,
        instance_count=1,
        framework_version="3.3",
        sagemaker_session=session,
        base_job_name="hymmrec-feature-engineering",
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "feature-engineering"},
        ],
    )

    step_feng = ProcessingStep(
        name="FeatureEngineering",
        processor=pyspark_processor,
        cache_config=CACHE_CONFIG,
        inputs=[
            ProcessingInput(
                source=f"s3://{DEFAULT_SILVER_BUCKET}/data/obt_movie_affinity/cleansed_ratings/",
                destination="/opt/ml/processing/input/ratings",
                input_name="ratings",
            ),
            ProcessingInput(
                source=f"s3://{DEFAULT_SILVER_BUCKET}/data/obt_movie_affinity/cleansed_movies/",
                destination="/opt/ml/processing/input/movies",
                input_name="movies",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/encoders",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/encoders/",
                output_name="encoders",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/feature_interactions",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/interactions/",
                output_name="feature_interactions",
            ),
        ],
        code=f"{PROCESSING_SCRIPTS_S3}/processing-feature-eng-job.py",
        job_arguments=["--region", "us-east-1", "--feature-group-name", "hymmrec-interactions-sm-fg"],
    )

    return step_feng


# ==============================================================================
# STEP 2: Embeddings Generation (SKLearn + Bedrock Nova)
# ==============================================================================


def create_step_embeddings(params, role, session):
    """
    Processing Job 2: Generación de Embeddings Multimodales.
    - Lee catálogo de películas + posters desde Silver
    - Genera embeddings multimodales (texto + imagen) con Amazon Bedrock Nova
    - Guarda embeddings_catalog.pkl en Platinum

    Puede correr EN PARALELO con Feature Engineering.
    """
    sklearn_processor = SKLearnProcessor(
        role=role,
        instance_type=DEFAULT_EMBEDDINGS_INSTANCE,
        instance_count=1,
        framework_version="1.2-1",
        sagemaker_session=session,
        base_job_name="hymmrec-embeddings-gen",
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "embeddings-generation"},
        ],
    )

    step_embeddings = ProcessingStep(
        name="EmbeddingsGeneration",
        processor=sklearn_processor,
        cache_config=CACHE_CONFIG,
        inputs=[
            ProcessingInput(
                source=f"s3://{DEFAULT_SILVER_BUCKET}/data/obt_movie_affinity/cleansed_movies/",
                destination="/opt/ml/processing/input/movies",
                input_name="movies",
            ),
            ProcessingInput(
                source=f"s3://{DEFAULT_SILVER_BUCKET}/data/imv_movie_affinity/movie_posters/",
                destination="/opt/ml/processing/input/posters",
                input_name="posters",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/embeddings",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/embeddings/",
                output_name="embeddings",
            ),
        ],
        code=f"{PROCESSING_SCRIPTS_S3}/processing-embeddings-job.py",
        job_arguments=["--region", "us-east-1", "--max-workers", "10"],
    )

    return step_embeddings


# ==============================================================================
# STEP 3: Dataset Preparation (K-Core + Splits)
# ==============================================================================


def create_step_data_splits(params, role, session, step_feng, step_embeddings):
    """
    Processing Job 3: Preparación de Datasets para Training.
    - Lee features de Platinum (output de Feature Engineering)
    - Filtrado K-Core (usuarios/items con pocas interacciones)
    - Split temporal-estratificado: train 80% / val 10% / test 10%
    - Persiste cold-starts

    DEPENDENCIAS: Feature Engineering + Embeddings completados.
    """
    sklearn_splits_processor = SKLearnProcessor(
        role=role,
        instance_type=DEFAULT_SPLITS_INSTANCE,
        instance_count=1,
        framework_version="1.2-1",
        sagemaker_session=session,
        base_job_name="hymmrec-dataset-splits",
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "dataset-preparation"},
        ],
    )

    step_splits = ProcessingStep(
        name="DatasetPreparation",
        processor=sklearn_splits_processor,
        cache_config=CACHE_CONFIG,
        inputs=[
            ProcessingInput(
                source=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/interactions/",
                destination="/opt/ml/processing/input/features",
                input_name="features",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/train",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/datasets/train/",
                output_name="train",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/val",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/datasets/val/",
                output_name="val",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/test",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/datasets/test/",
                output_name="test",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/cold-starts",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/datasets/cold-starts/",
                output_name="cold_starts",
            ),
        ],
        code=f"{PROCESSING_SCRIPTS_S3}/processing-prepare-data-splits.py",
        job_arguments=[
            "--min-user-interactions", "5",
            "--min-item-interactions", "5",
            "--train-ratio", "0.80",
            "--val-ratio", "0.10",
        ],
    )

    step_splits.add_depends_on([step_feng, step_embeddings])
    return step_splits


# ==============================================================================
# STEP 4: HPO Regression (Hyperparameter Tuning)
# ==============================================================================


def create_step_hpo_regression(params, role, session, step_splits):
    """
    HPO Job: Busca mejores hiperparámetros para modelo de regresión.
    - Métrica objetivo: val_rmse_stars (Minimize)
    - HP tunables: lr, batch_size, emb_dim, dropout, weight_decay
    - Estrategia: Bayesian, 12 jobs, 3 en paralelo

    DEPENDENCIA: Data Splits completado.
    """
    regression_estimator = PyTorch(
        entry_point="hpo_hymmrec_regression.py",
        source_dir=TRAINING_SCRIPTS_LOCAL,
        role=role,
        instance_type=DEFAULT_HPO_INSTANCE,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="hymmrec-hpo-regression",
        hyperparameters={
            "epochs": 15,
            "patience": 2,
            "num_workers": 2,
        },
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "hpo-regression"},
        ],
        metric_definitions=[
            {"Name": "val_rmse_stars", "Regex": r"val_rmse_stars=(.*?);"},
            {"Name": "train_mse", "Regex": r"train_mse=(.*?);"},
            {"Name": "val_mse", "Regex": r"val_mse=(.*?);"},
            {"Name": "test_rmse_stars", "Regex": r"test_rmse_stars=(.*?);"},
        ],
    )

    regression_hp_ranges = {
        "lr": ContinuousParameter(0.0001, 0.01, scaling_type="Logarithmic"),
        "batch_size": CategoricalParameter([128, 256, 512]),
        "emb_dim": CategoricalParameter([32, 64, 128]),
        "dropout": ContinuousParameter(0.1, 0.5),
        "weight_decay": ContinuousParameter(1e-6, 1e-3, scaling_type="Logarithmic"),
    }

    regression_tuner = HyperparameterTuner(
        estimator=regression_estimator,
        objective_metric_name="val_rmse_stars",
        hyperparameter_ranges=regression_hp_ranges,
        metric_definitions=[
            {"Name": "val_rmse_stars", "Regex": r"val_rmse_stars=(.*?);"},
            {"Name": "train_mse", "Regex": r"train_mse=(.*?);"},
            {"Name": "val_mse", "Regex": r"val_mse=(.*?);"},
            {"Name": "test_rmse_stars", "Regex": r"test_rmse_stars=(.*?);"},
        ],
        objective_type="Minimize",
        max_jobs=HPO_MAX_JOBS,
        max_parallel_jobs=HPO_MAX_PARALLEL_JOBS,
        strategy=HPO_STRATEGY,
        base_tuning_job_name="hymmrec-hpo-reg",
    )

    step_hpo_regression = TuningStep(
        name="HPORegression",
        tuner=regression_tuner,
        cache_config=CACHE_CONFIG,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/datasets/",
                content_type="application/x-parquet",
            ),
            "embeddings": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/embeddings/",
                content_type="application/octet-stream",
            ),
            "encoders": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/encoders/",
                content_type="application/octet-stream",
            ),
        },
    )

    step_hpo_regression.add_depends_on([step_splits])
    return step_hpo_regression


# ==============================================================================
# STEP 5: HPO Two-Heads (Hyperparameter Tuning)
# ==============================================================================


def create_step_hpo_twoheads(params, role, session, step_splits):
    """
    HPO Job: Busca mejores hiperparámetros para modelo Two-Heads.
    - Métrica objetivo: val_bce (Minimize) — discriminación positivos vs negativos
    - HP tunables: lr, batch_size, emb_dim, dropout, weight_decay
    - neg_ratio=4 fijo (no tunable)
    - Estrategia: Bayesian, 12 jobs, 3 en paralelo

    DEPENDENCIA: Data Splits completado. Paralelo con HPO Regression.
    """
    twoheads_estimator = PyTorch(
        entry_point="hpo_hymmrec_twoheads.py",
        source_dir=TRAINING_SCRIPTS_LOCAL,
        role=role,
        instance_type=DEFAULT_HPO_INSTANCE,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="hymmrec-hpo-twoheads",
        hyperparameters={
            "epochs": 15,
            "patience": 3,
            "num_workers": 2,
            "neg_ratio": 4,
        },
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "hpo-twoheads"},
        ],
        metric_definitions=[
            {"Name": "val_bce", "Regex": r"val_bce=(.*?);"},
            {"Name": "train_total_loss", "Regex": r"train_total_loss=(.*?);"},
            {"Name": "val_total_loss", "Regex": r"val_total_loss=(.*?);"},
            {"Name": "val_mse", "Regex": r"val_mse=(.*?);"},
            {"Name": "val_rmse_stars", "Regex": r"val_rmse_stars=(.*?);"},
            {"Name": "test_total_loss", "Regex": r"test_total_loss=(.*?);"},
            {"Name": "test_rmse_stars", "Regex": r"test_rmse_stars=(.*?);"},
        ],
    )

    twoheads_hp_ranges = {
        "lr": ContinuousParameter(0.0001, 0.005, scaling_type="Logarithmic"),
        "batch_size": CategoricalParameter([128, 256, 512]),
        "emb_dim": CategoricalParameter([64, 128]),
        "dropout": ContinuousParameter(0.1, 0.5),
        "weight_decay": ContinuousParameter(1e-6, 1e-3, scaling_type="Logarithmic"),
    }

    twoheads_tuner = HyperparameterTuner(
        estimator=twoheads_estimator,
        objective_metric_name="val_bce",
        hyperparameter_ranges=twoheads_hp_ranges,
        metric_definitions=[
            {"Name": "val_bce", "Regex": r"val_bce=(.*?);"},
            {"Name": "train_total_loss", "Regex": r"train_total_loss=(.*?);"},
            {"Name": "val_total_loss", "Regex": r"val_total_loss=(.*?);"},
            {"Name": "val_mse", "Regex": r"val_mse=(.*?);"},
            {"Name": "val_rmse_stars", "Regex": r"val_rmse_stars=(.*?);"},
            {"Name": "test_total_loss", "Regex": r"test_total_loss=(.*?);"},
            {"Name": "test_rmse_stars", "Regex": r"test_rmse_stars=(.*?);"},
        ],
        objective_type="Minimize",
        max_jobs=HPO_MAX_JOBS,
        max_parallel_jobs=HPO_MAX_PARALLEL_JOBS,
        strategy=HPO_STRATEGY,
        base_tuning_job_name="hymmrec-hpo-th",
    )

    step_hpo_twoheads = TuningStep(
        name="HPOTwoHeads",
        tuner=twoheads_tuner,
        cache_config=CACHE_CONFIG,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/datasets/",
                content_type="application/x-parquet",
            ),
            "embeddings": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/embeddings/",
                content_type="application/octet-stream",
            ),
            "encoders": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/encoders/",
                content_type="application/octet-stream",
            ),
        },
    )

    step_hpo_twoheads.add_depends_on([step_splits])
    return step_hpo_twoheads


# ==============================================================================
# STEP 6: Training Final — Regression (con mejores HPs del HPO)
# ==============================================================================


def create_step_training_regression(params, role, session, step_hpo_regression):
    """
    Training Job Final: Modelo de Regresión con mejores HPs del HPO.
    - Usa get_top_model_s3_uri() del TuningStep para obtener el mejor modelo
    - Más épocas (30) y más paciencia (5) que en HPO
    - Output: model.tar.gz con model.pth + model_metadata.json

    DEPENDENCIA: HPO Regression completado.
    """
    regression_estimator = PyTorch(
        entry_point="train_hymmrec_regression.py",
        source_dir=TRAINING_SCRIPTS_LOCAL,
        role=role,
        instance_type=DEFAULT_TRAINING_INSTANCE,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="hymmrec-train-regression",
        output_path=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/models/regression/",
        hyperparameters={
            "epochs": 30,
            "patience": 5,
            "num_workers": 2,
            "scheduler_patience": 2,
            "scheduler_factor": 0.5,
            "min_lr": 1e-6,
            # Best HPs del HPO (hymmrec-hpo-regression-2026-07-20)
            "lr": 0.0002838743621574962,
            "batch_size": 256,
            "emb_dim": 128,
            "dropout": 0.41655263256088393,
            "weight_decay": 1.1497633801683068e-06,
        },
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "training-regression-final"},
        ],
        metric_definitions=[
            {"Name": "train_mse", "Regex": r"train_mse=(.*?);"},
            {"Name": "val_mse", "Regex": r"val_mse=(.*?);"},
            {"Name": "val_rmse_stars", "Regex": r"val_rmse_stars=(.*?);"},
            {"Name": "test_rmse_stars", "Regex": r"test_rmse_stars=(.*?);"},
        ],
    )

    step_train_regression = TrainingStep(
        name="TrainingRegression",
        estimator=regression_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/datasets/",
                content_type="application/x-parquet",
            ),
            "embeddings": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/embeddings/",
                content_type="application/octet-stream",
            ),
            "encoders": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/encoders/",
                content_type="application/octet-stream",
            ),
        },
    )

    step_train_regression.add_depends_on([step_hpo_regression])
    return step_train_regression


# ==============================================================================
# STEP 7: Training Final — Two-Heads (con mejores HPs del HPO)
# ==============================================================================


def create_step_training_twoheads(params, role, session, step_hpo_twoheads):
    """
    Training Job Final: Modelo Multi-Task Two-Heads con mejores HPs del HPO.
    - Cabeza 1: BCE (ranking/retrieval) sobre todos los datos
    - Cabeza 2: MSE (calidad) enmascarado solo sobre positivos
    - neg_ratio=4 fijo
    - Más épocas (30) y más paciencia (5) que en HPO

    DEPENDENCIA: HPO Two-Heads completado.
    """
    twoheads_estimator = PyTorch(
        entry_point="train_hymmrec_twoheads.py",
        source_dir=TRAINING_SCRIPTS_LOCAL,
        role=role,
        instance_type=DEFAULT_TRAINING_INSTANCE,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="hymmrec-train-twoheads",
        output_path=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/models/twoheads/",
        hyperparameters={
            "epochs": 30,
            "patience": 5,
            "neg_ratio": 4,
            "num_workers": 2,
            "scheduler_patience": 2,
            "scheduler_factor": 0.5,
            "min_lr": 1e-6,
            # Best HPs del HPO (hymmrec-hpo-twoheads-2026-07-20)
            "lr": 0.00018431337415071667,
            "batch_size": 128,
            "emb_dim": 128,
            "dropout": 0.23663261628280174,
            "weight_decay": 0.0008727185963231925,
        },
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "training-twoheads-final"},
        ],
        metric_definitions=[
            {"Name": "train_total_loss", "Regex": r"train_total_loss=(.*?);"},
            {"Name": "val_total_loss", "Regex": r"val_total_loss=(.*?);"},
            {"Name": "val_bce", "Regex": r"val_bce=(.*?);"},
            {"Name": "val_mse", "Regex": r"val_mse=(.*?);"},
            {"Name": "val_rmse_stars", "Regex": r"val_rmse_stars=(.*?);"},
            {"Name": "test_total_loss", "Regex": r"test_total_loss=(.*?);"},
            {"Name": "test_rmse_stars", "Regex": r"test_rmse_stars=(.*?);"},
        ],
    )

    step_train_twoheads = TrainingStep(
        name="TrainingTwoHeads",
        estimator=twoheads_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/datasets/",
                content_type="application/x-parquet",
            ),
            "embeddings": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/embeddings/",
                content_type="application/octet-stream",
            ),
            "encoders": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/encoders/",
                content_type="application/octet-stream",
            ),
        },
    )

    step_train_twoheads.add_depends_on([step_hpo_twoheads])
    return step_train_twoheads


# ==============================================================================
# STEP 8: Model Evaluation (Processing Job)
# ==============================================================================


def create_step_evaluation(params, role, session, step_train_regression, step_train_twoheads):
    """
    Evaluation Job: Evalúa ambos modelos sobre test + cold-start.
    - Métricas: RMSE, NDCG@K, HR@K, Precision@K, Recall@K, F1
    - Selecciona modelo ganador automáticamente (hybrid score)
    - Output: evaluation_report.json con winner_rmse_stars

    DEPENDENCIA: Ambos Training Jobs completados.
    """
    eval_processor = PyTorchProcessor(
        role=role,
        instance_type=DEFAULT_EVAL_INSTANCE,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="hymmrec-evaluation",
        command=["python3"],
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "evaluation"},
        ],
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="reports",
        path="evaluation_report.json",
    )

    step_eval = ProcessingStep(
        name="ModelEvaluation",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train_regression.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/models/regression",
                input_name="model_regression",
            ),
            ProcessingInput(
                source=step_train_twoheads.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/models/twoheads",
                input_name="model_twoheads",
            ),
            ProcessingInput(
                source=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/datasets/",
                destination="/opt/ml/processing/input/datasets",
                input_name="datasets",
            ),
            ProcessingInput(
                source=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/embeddings/",
                destination="/opt/ml/processing/input/embeddings",
                input_name="embeddings",
            ),
            ProcessingInput(
                source=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/model_artefacts/encoders/",
                destination="/opt/ml/processing/input/encoders",
                input_name="encoders",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/reports",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/evaluation/reports/",
                output_name="reports",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/winner",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/evaluation/winner/",
                output_name="winner",
            ),
        ],
        code=f"{EVALUATION_SCRIPTS_S3}/evaluation-job.py",
        job_arguments=["--k", "10", "--num-decoys", "99"],
        property_files=[evaluation_report],
    )

    return step_eval, evaluation_report


# ==============================================================================
# STEP 9: Model Registry
# ==============================================================================


def create_step_register_model(params, role, session, step_train_regression, step_train_twoheads):
    """
    Model Registry: Registra ambos modelos en SageMaker Model Registry.
    - Status: PendingManualApproval
    - El evaluation job determina el ganador en metadata
    """
    # Registrar modelo Two-Heads
    step_register_twoheads = RegisterModel(
        name="RegisterModelTwoHeads",
        estimator=PyTorch(
            entry_point="train_hymmrec_twoheads.py",
            source_dir=TRAINING_SCRIPTS_LOCAL,
            role=role,
            instance_type=DEFAULT_TRAINING_INSTANCE,
            instance_count=1,
            framework_version="2.1",
            py_version="py310",
            sagemaker_session=session,
        ),
        model_data=step_train_twoheads.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.c5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large", "ml.m5.xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="PendingManualApproval",
        description=(
            "HYMM-REC Two-Heads: Recomendador híbrido multimodal. "
            "Two-Tower con atención multimodal explicable. "
            "BCE(ranking) + MSE(calidad enmascarado)."
        ),
    )

    # Registrar modelo Regresión
    step_register_regression = RegisterModel(
        name="RegisterModelRegression",
        estimator=PyTorch(
            entry_point="train_hymmrec_regression.py",
            source_dir=TRAINING_SCRIPTS_LOCAL,
            role=role,
            instance_type=DEFAULT_TRAINING_INSTANCE,
            instance_count=1,
            framework_version="2.1",
            py_version="py310",
            sagemaker_session=session,
        ),
        model_data=step_train_regression.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.c5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large", "ml.m5.xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="PendingManualApproval",
        description=(
            "HYMM-REC Regression: Predicción de rating puro. "
            "Single-Head MSELoss sobre rating escalado [0,1]."
        ),
    )

    return step_register_twoheads, step_register_regression


# ==============================================================================
# STEP 10: Model Packaging
# ==============================================================================


def create_step_model_packaging(params, role, session, step_register_twoheads):
    """
    Model Packaging: Extrae UserTower, ItemTower y FullModel como artefactos
    separados con inference.py para deploy independiente.

    DEPENDENCIA: Model Registry completado.
    """
    packaging_processor = PyTorchProcessor(
        role=role,
        instance_type=DEFAULT_PACKAGING_INSTANCE,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="hymmrec-model-packaging",
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "model-packaging"},
        ],
    )

    step_packaging = ProcessingStep(
        name="ModelPackaging",
        processor=packaging_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/evaluation/winner/",
                destination="/opt/ml/processing/input/winner",
                input_name="winner",
            ),
            ProcessingInput(
                source=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/evaluation/models/twoheads/",
                destination="/opt/ml/processing/input/model",
                input_name="model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/full-model",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/packaged-models/full-model/",
                output_name="full_model",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/user-tower",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/packaged-models/user-tower/",
                output_name="user_tower",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/item-tower",
                destination=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/packaged-models/item-tower/",
                output_name="item_tower",
            ),
        ],
        code=f"{EVALUATION_SCRIPTS_S3}/model-packaging-job.py",
    )

    step_packaging.add_depends_on([step_register_twoheads])
    return step_packaging


# ==============================================================================
# STEP 11: Quality Gate (Condition)
# ==============================================================================


def create_condition_quality_gate(
    params, evaluation_report, step_eval,
    step_register_twoheads, step_register_regression, step_packaging,
):
    """
    Quality Gate: RMSE del modelo ganador <= threshold.
    - PASS → Registra modelos + Packaging
    - FAIL → FailStep con mensaje descriptivo
    """
    cond_rmse = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="winner_rmse_stars",
        ),
        right=params["max_rmse_threshold"],
    )

    step_fail = FailStep(
        name="QualityGateFailed",
        error_message=(
            "El modelo ganador NO supera el umbral de calidad. "
            "RMSE en estrellas supera el máximo permitido. "
            "Revisa hiperparámetros, datos, o ajusta el threshold."
        ),
    )

    step_condition = ConditionStep(
        name="QualityGateCheck",
        conditions=[cond_rmse],
        if_steps=[step_register_twoheads, step_register_regression, step_packaging],
        else_steps=[step_fail],
    )

    return step_condition


# ==============================================================================
# BUILD PIPELINE (ensambla el DAG completo)
# ==============================================================================


def build_pipeline(
    region: str = "us-east-1",
    role_arn: Optional[str] = None,
    pipeline_name: str = PIPELINE_NAME,
) -> Pipeline:
    """
    Construye el SageMaker Pipeline completo.

    DAG:
      FeatureEngineering ──┐
                           ├──► DatasetPreparation ──┬──► HPORegression ──► TrainingRegression ──┐
      EmbeddingsGeneration ┘                         └──► HPOTwoHeads  ──► TrainingTwoHeads  ───┤
                                                                                                ▼
                                                                                        ModelEvaluation
                                                                                                │
                                                                                                ▼
                                                                                      QualityGateCheck
                                                                                       ┌────┴────┐
                                                                                  Pass │         │ Fail
                                                                                       ▼         ▼
                                                                              RegisterModel   FailStep
                                                                              + Packaging
    """
    boto_session = boto3.Session(region_name=region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    if role_arn:
        role = role_arn
    else:
        try:
            role = get_execution_role()
        except ValueError:
            role = os.environ.get(
                "SAGEMAKER_ROLE_ARN",
                "arn:aws:iam::697682206292:role/sgmkr-notebook-tfm-hymm-rec-ml-iar-dev",
            )

    logger.info(f"Building pipeline: {pipeline_name} | Region: {region}")
    logger.info(f"Role: {role}")
    logger.info(f"Scripts S3 base: {SCRIPTS_S3_BASE}")

    # --- Parameters ---
    params = define_pipeline_parameters()

    # --- Step 1: Feature Engineering ---
    step_feng = create_step_feature_engineering(params, role, sm_session)
    logger.info("  [1/11] FeatureEngineering")

    # --- Step 2: Embeddings ---
    step_embeddings = create_step_embeddings(params, role, sm_session)
    logger.info("  [2/11] EmbeddingsGeneration")

    # --- Step 3: Data Splits ---
    step_splits = create_step_data_splits(params, role, sm_session, step_feng, step_embeddings)
    logger.info("  [3/11] DatasetPreparation")

    # --- Step 4: HPO Regression ---
    step_hpo_regression = create_step_hpo_regression(params, role, sm_session, step_splits)
    logger.info("  [4/11] HPORegression")

    # --- Step 5: HPO Two-Heads ---
    step_hpo_twoheads = create_step_hpo_twoheads(params, role, sm_session, step_splits)
    logger.info("  [5/11] HPOTwoHeads")

    # --- Step 6: Training Regression ---
    step_train_regression = create_step_training_regression(
        params, role, sm_session, step_hpo_regression
    )
    logger.info("  [6/11] TrainingRegression")

    # --- Step 7: Training Two-Heads ---
    step_train_twoheads = create_step_training_twoheads(
        params, role, sm_session, step_hpo_twoheads
    )
    logger.info("  [7/11] TrainingTwoHeads")

    # --- Step 8: Evaluation ---
    step_eval, evaluation_report = create_step_evaluation(
        params, role, sm_session, step_train_regression, step_train_twoheads
    )
    logger.info("  [8/11] ModelEvaluation")

    # --- Step 9: Model Registry ---
    step_register_twoheads, step_register_regression = create_step_register_model(
        params, role, sm_session, step_train_regression, step_train_twoheads
    )
    logger.info("  [9/11] RegisterModel (TwoHeads + Regression)")

    # --- Step 10: Model Packaging ---
    step_packaging = create_step_model_packaging(
        params, role, sm_session, step_register_twoheads
    )
    logger.info("  [10/11] ModelPackaging")

    # --- Step 11: Quality Gate ---
    step_condition = create_condition_quality_gate(
        params, evaluation_report, step_eval,
        step_register_twoheads, step_register_regression, step_packaging,
    )
    logger.info("  [11/11] QualityGateCheck")

    # --- Assemble Pipeline ---
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            params["region"],
            params["role_arn"],
            params["silver_bucket"],
            params["gold_bucket"],
            params["platinum_bucket"],
            params["processing_instance_type"],
            params["training_instance_type"],
            params["eval_instance_type"],
            params["epochs"],
            params["batch_size"],
            params["learning_rate"],
            params["emb_dim"],
            params["dropout"],
            params["hpo_max_jobs"],
            params["hpo_max_parallel"],
            params["min_user_interactions"],
            params["min_item_interactions"],
            params["train_ratio"],
            params["val_ratio"],
            params["max_rmse_threshold"],
        ],
        steps=[
            step_feng,
            step_embeddings,
            step_splits,
            step_hpo_regression,
            step_hpo_twoheads,
            step_train_regression,
            step_train_twoheads,
            step_eval,
            step_condition,
        ],
        sagemaker_session=sm_session,
    )

    logger.info(f"Pipeline '{pipeline_name}' built with {len(pipeline.steps)} steps.")
    return pipeline


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def export_pipeline_json(pipeline: Pipeline, output_path: str) -> dict:
    """Exporta la definición del pipeline como JSON."""
    definition = json.loads(pipeline.definition())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(definition, f, indent=2, ensure_ascii=False)
    logger.info(f"Pipeline JSON exported: {output_path}")
    return definition


def upsert_pipeline(pipeline: Pipeline, role_arn: str) -> str:
    """Crea o actualiza el pipeline en SageMaker."""
    response = pipeline.upsert(role_arn=role_arn)
    pipeline_arn = response["PipelineArn"]
    logger.info(f"Pipeline upserted: {pipeline_arn}")
    return pipeline_arn


def start_pipeline(pipeline: Pipeline, parameters: Optional[dict] = None) -> str:
    """Inicia una ejecución del pipeline."""
    execution = pipeline.start(parameters=parameters or {})
    logger.info(f"Pipeline execution started: {execution.arn}")
    return execution.arn


# ==============================================================================
# MAIN — CLI
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="HYMM-REC SageMaker Pipeline (HPO + CI/CD Ready)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Exportar JSON de definición (para inspección):
  python define_sagemaker_pipeline.py --export-json pipeline_definition.json

  # Crear/actualizar pipeline en SageMaker:
  python define_sagemaker_pipeline.py --upsert

  # Crear y ejecutar inmediatamente:
  python define_sagemaker_pipeline.py --execute

  # Desde CI/CD (Azure DevOps) con role explícito:
  python define_sagemaker_pipeline.py --upsert --execute \\
      --role-arn arn:aws:iam::697682206292:role/sgmkr-notebook-tfm-hymm-rec-ml-iar-dev
        """,
    )

    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--role-arn", type=str, default=None)
    parser.add_argument("--pipeline-name", type=str, default=PIPELINE_NAME)
    parser.add_argument("--export-json", type=str, default=None, metavar="PATH")
    parser.add_argument("--upsert", action="store_true")
    parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()

    # Build pipeline
    pipeline = build_pipeline(
        region=args.region,
        role_arn=args.role_arn,
        pipeline_name=args.pipeline_name,
    )

    # Determine role
    role = args.role_arn or os.environ.get(
        "SAGEMAKER_ROLE_ARN",
        "arn:aws:iam::697682206292:role/sgmkr-notebook-tfm-hymm-rec-ml-iar-dev",
    )

    # Export JSON
    if args.export_json:
        export_pipeline_json(pipeline, args.export_json)

    # Default: export JSON if no action specified
    if not args.export_json and not args.upsert and not args.execute:
        default_path = "pipeline_definition.json"
        definition = export_pipeline_json(pipeline, default_path)
        print(f"\n{'='*70}")
        print("HYMM-REC SAGEMAKER PIPELINE DEFINITION (con HPO)")
        print(f"{'='*70}")
        print(f"  Archivo:     {default_path}")
        print(f"  Pipeline:    {args.pipeline_name}")
        print(f"  Steps:       {len(definition.get('Steps', []))}")
        print(f"  Parameters:  {len(definition.get('Parameters', []))}")
        print(f"\nDAG del Pipeline:")
        print(f"  1. FeatureEngineering (PySpark)")
        print(f"  2. EmbeddingsGeneration (SKLearn + Bedrock Nova)")
        print(f"  3. DatasetPreparation (SKLearn) [depends: 1, 2]")
        print(f"  4. HPORegression (Bayesian, {HPO_MAX_JOBS} jobs) [depends: 3]")
        print(f"  5. HPOTwoHeads (Bayesian, {HPO_MAX_JOBS} jobs) [depends: 3]")
        print(f"  6. TrainingRegression (PyTorch GPU) [depends: 4]")
        print(f"  7. TrainingTwoHeads (PyTorch GPU) [depends: 5]")
        print(f"  8. ModelEvaluation (PyTorch GPU) [depends: 6, 7]")
        print(f"  9. QualityGateCheck (Condition: RMSE <= threshold)")
        print(f"     +-- PASS: RegisterModel (TwoHeads + Regression) -> ModelPackaging")
        print(f"     +-- FAIL: QualityGateFailed (FailStep)")
        print(f"\nScripts S3: {SCRIPTS_S3_BASE}/")
        print(f"\nPara crear en SageMaker:  python define_sagemaker_pipeline.py --upsert")
        print(f"Para ejecutar:            python define_sagemaker_pipeline.py --execute")
        return

    # Upsert
    if args.upsert or args.execute:
        pipeline_arn = upsert_pipeline(pipeline, role)
        print(f"\n Pipeline registrado: {pipeline_arn}")

    # Execute
    if args.execute:
        execution_arn = start_pipeline(pipeline)
        print(f"\n Pipeline en ejecucion: {execution_arn}")
        print(f"   Monitor: SageMaker Studio -> Pipelines -> {args.pipeline_name}")


if __name__ == "__main__":
    main()
