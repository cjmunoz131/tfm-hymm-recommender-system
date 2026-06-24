"""
==============================================================================
HYMM-REC MLOps: Definición del SageMaker Pipeline (DAG Completo)
==============================================================================
Este script genera la definición JSON del SageMaker Pipeline que orquesta
el flujo MLOps end-to-end del sistema recomendador híbrido multimodal.

DAG del Pipeline:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Processing (Feature Eng)  ──┐                                      │
  │                              ├──► Processing (Data Splits)          │
  │  Processing (Embeddings)  ───┘         │                            │
  │                                        ▼                            │
  │                          ┌─── Training (Regression) ───┐            │
  │                          │                             │            │
  │                          └─── Training (Two-Heads)  ───┘            │
  │                                        │                            │
  │                                        ▼                            │
  │                              Evaluation Job                         │
  │                                        │                            │
  │                                        ▼                            │
  │                           Condition (Quality Gate)                   │
  │                              ┌────┴────┐                            │
  │                         Pass │         │ Fail                       │
  │                              ▼         ▼                            │
  │                     Model Registry   FailStep                       │
  │                              │                                      │
  │                              ▼                                      │
  │                      Model Packaging                                │
  └─────────────────────────────────────────────────────────────────────┘

Uso:
  python define_sagemaker_pipeline.py [--region us-east-1] [--role-arn <arn>]

  # Solo generar JSON sin ejecutar:
  python define_sagemaker_pipeline.py --export-json pipeline_definition.json

  # Crear/actualizar y ejecutar:
  python define_sagemaker_pipeline.py --execute
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
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTES Y CONFIGURACIÓN POR DEFECTO
# ==============================================================================

PIPELINE_NAME = "hymmrec-mlops-pipeline"
PIPELINE_DESCRIPTION = (
    "Pipeline MLOps end-to-end para el sistema recomendador híbrido multimodal "
    "HYMM-REC. Incluye Feature Engineering, Embeddings, Data Splits, Training "
    "(Regresión + Two-Heads), Evaluation, Model Registry y Packaging."
)

# Buckets por defecto
DEFAULT_SILVER_BUCKET = "hymmrec-dilkehousesilver01"
DEFAULT_GOLD_BUCKET = "hymmrec-dilkehousegold01"
DEFAULT_PLATINUM_BUCKET = "hymmrec-sagemaker-assets"

# Instancias por defecto
DEFAULT_PROCESSING_INSTANCE = "ml.m5.xlarge"
DEFAULT_EMBEDDINGS_INSTANCE = "ml.t3.medium"
DEFAULT_SPLITS_INSTANCE = "ml.m5.large"
DEFAULT_TRAINING_INSTANCE = "ml.g4dn.xlarge"
DEFAULT_EVAL_INSTANCE = "ml.g4dn.xlarge"
DEFAULT_PACKAGING_INSTANCE = "ml.m5.large"

# Scripts relativos al directorio dev/
SCRIPTS_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "dev")
PROCESSING_SCRIPTS_DIR = os.path.join(SCRIPTS_BASE_DIR, "feng-data-preparing")
TRAINING_SCRIPTS_DIR = os.path.join(SCRIPTS_BASE_DIR, "training")
EVALUATION_SCRIPTS_DIR = os.path.join(SCRIPTS_BASE_DIR, "evaluation")

# Model Registry
MODEL_PACKAGE_GROUP_NAME = "hymmrec-model-sm-pg"


# ==============================================================================
# PARÁMETROS DEL PIPELINE (configurables en tiempo de ejecución)
# ==============================================================================


def define_pipeline_parameters():
    """Define los parámetros parametrizables del pipeline."""

    # --- Infraestructura ---
    region = ParameterString(name="Region", default_value="us-east-1")
    role_arn = ParameterString(name="RoleArn", default_value="")

    # --- Buckets ---
    silver_bucket = ParameterString(
        name="SilverBucket", default_value=DEFAULT_SILVER_BUCKET
    )
    gold_bucket = ParameterString(
        name="GoldBucket", default_value=DEFAULT_GOLD_BUCKET
    )
    platinum_bucket = ParameterString(
        name="PlatinumBucket", default_value=DEFAULT_PLATINUM_BUCKET
    )

    # --- Instancias ---
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value=DEFAULT_PROCESSING_INSTANCE
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value=DEFAULT_TRAINING_INSTANCE
    )
    eval_instance_type = ParameterString(
        name="EvalInstanceType", default_value=DEFAULT_EVAL_INSTANCE
    )

    # --- Hiperparámetros de Training ---
    epochs = ParameterInteger(name="Epochs", default_value=30)
    batch_size = ParameterInteger(name="BatchSize", default_value=256)
    learning_rate = ParameterFloat(name="LearningRate", default_value=0.001)
    emb_dim = ParameterInteger(name="EmbeddingDim", default_value=64)
    dropout = ParameterFloat(name="Dropout", default_value=0.3)

    # --- Data Splits ---
    min_user_interactions = ParameterInteger(
        name="MinUserInteractions", default_value=5
    )
    min_item_interactions = ParameterInteger(
        name="MinItemInteractions", default_value=5
    )
    train_ratio = ParameterFloat(name="TrainRatio", default_value=0.80)
    val_ratio = ParameterFloat(name="ValRatio", default_value=0.10)

    # --- Quality Gate ---
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
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "max_rmse_threshold": max_rmse_threshold,
    }


# ==============================================================================
# STEP 1: PROCESSING — Feature Engineering (PySpark)
# ==============================================================================


def create_step_feature_engineering(params, role, session):
    """
    Processing Job 1: Feature Engineering + Feature Store Ingestion.
    - Lee ratings + movies de Silver (Iceberg/Parquet)
    - Aplica Label Encoding, Multi-Hot, Rating Scaling (PySpark + sklearn)
    - Ingesta features en Feature Store offline
    - Guarda encoders.pkl en Platinum
    """
    pyspark_processor = PySparkProcessor(
        role=role,
        instance_type=params["processing_instance_type"],
        instance_count=1,
        framework_version="3.3",
        sagemaker_session=session,
        base_job_name="hymmrec-feature-engineering",
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "feature-engineering"},
            {"Key": "pipeline", "Value": PIPELINE_NAME},
        ],
    )

    step_feng = ProcessingStep(
        name="FeatureEngineering",
        processor=pyspark_processor,
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
        code=os.path.join(PROCESSING_SCRIPTS_DIR, "processing-feature-eng-job.py"),
        job_arguments=[
            "--region", "us-east-1",
            "--feature-group-name", "hymmrec-interactions-sm-fg",
        ],
    )

    return step_feng


# ==============================================================================
# STEP 2: PROCESSING — Multimodal Embeddings (Bedrock Nova)
# ==============================================================================


def create_step_embeddings(params, role, session):
    """
    Processing Job 2: Generación de Embeddings Multimodales.
    - Lee catálogo de películas desde Silver
    - Descarga posters desde Silver
    - Genera embeddings multimodales (texto + imagen) con Amazon Bedrock Nova
    - Guarda embeddings_catalog.pkl en Gold/Platinum

    NOTA: Este step puede correr EN PARALELO con Feature Engineering
    ya que no depende de sus outputs.
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
            {"Key": "pipeline", "Value": PIPELINE_NAME},
        ],
    )

    step_embeddings = ProcessingStep(
        name="EmbeddingsGeneration",
        processor=sklearn_processor,
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
        code=os.path.join(PROCESSING_SCRIPTS_DIR, "processing-embeddings-job.py"),
        job_arguments=[
            "--region", "us-east-1",
            "--max-workers", "10",
        ],
    )

    return step_embeddings


# ==============================================================================
# STEP 3: PROCESSING — Dataset Preparation (K-Core + Splits)
# ==============================================================================


def create_step_data_splits(params, role, session, step_feng, step_embeddings):
    """
    Processing Job 3: Preparación de Datasets para Training.
    - Lee features de Platinum (output de Feature Engineering)
    - Aplica filtrado K-Core (usuarios/items con pocas interacciones)
    - Split temporal-estratificado: train 80% / val 10% / test 10%
    - Persiste cold-starts (datos descartados)

    DEPENDENCIAS: Requiere que Feature Engineering Y Embeddings hayan completado.
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
            {"Key": "pipeline", "Value": PIPELINE_NAME},
        ],
    )

    step_splits = ProcessingStep(
        name="DatasetPreparation",
        processor=sklearn_splits_processor,
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
        code=os.path.join(PROCESSING_SCRIPTS_DIR, "processing-prepare-data-splits.py"),
        job_arguments=[
            "--min-user-interactions", "5",
            "--min-item-interactions", "5",
            "--train-ratio", "0.80",
            "--val-ratio", "0.10",
        ],
    )

    # Dependencias: esperar a que Feature Engineering y Embeddings terminen
    step_splits.add_depends_on([step_feng, step_embeddings])

    return step_splits


# ==============================================================================
# STEP 4: TRAINING — Regresión (Single-Head: Rating Prediction)
# ==============================================================================


def create_step_training_regression(params, role, session, step_splits):
    """
    Training Job: Modelo de Regresión (Single-Head).
    - Predice rating escalado [0,1] con MSELoss
    - Usa embeddings multimodales + features estructurales
    - Métricas: RMSE en estrellas (1-5)

    DEPENDENCIA: Requiere que Data Splits haya completado.
    """
    regression_estimator = PyTorch(
        entry_point="train_hymmrec_regression.py",
        source_dir=TRAINING_SCRIPTS_DIR,
        role=role,
        instance_type=params["training_instance_type"],
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="hymmrec-train-regression",
        output_path=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/models/regression/",
        hyperparameters={
            "epochs": params["epochs"],
            "patience": 5,
            "batch_size": params["batch_size"],
            "lr": params["learning_rate"],
            "emb_dim": params["emb_dim"],
            "dropout": params["dropout"],
            "weight_decay": 1e-5,
            "num_workers": 2,
            "scheduler_patience": 2,
            "scheduler_factor": 0.5,
            "min_lr": 1e-6,
        },
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "training-regression"},
            {"Key": "pipeline", "Value": PIPELINE_NAME},
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

    # Dependencia: requiere Data Splits completado
    step_train_regression.add_depends_on([step_splits])

    return step_train_regression


# ==============================================================================
# STEP 5: TRAINING — Multi-Task Two-Heads (Retrieval + Calidad)
# ==============================================================================


def create_step_training_twoheads(params, role, session, step_splits):
    """
    Training Job: Modelo Multi-Task Two-Heads.
    - Cabeza 1: BCE (ranking/retrieval) sobre TODOS los datos
    - Cabeza 2: MSE (calidad) ENMASCARADO solo sobre interacciones positivas
    - Loss Total = BCE + MSE(positivos)
    - Arquitectura Two-Tower con atención multimodal explicable

    DEPENDENCIA: Requiere que Data Splits haya completado.
    Puede correr EN PARALELO con Training Regression.
    """
    twoheads_estimator = PyTorch(
        entry_point="train_hymmrec_twoheads.py",
        source_dir=TRAINING_SCRIPTS_DIR,
        role=role,
        instance_type=params["training_instance_type"],
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="hymmrec-train-twoheads",
        output_path=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/models/twoheads/",
        hyperparameters={
            "epochs": params["epochs"],
            "patience": 5,
            "batch_size": params["batch_size"],
            "lr": params["learning_rate"],
            "emb_dim": params["emb_dim"],
            "dropout": params["dropout"],
            "weight_decay": 1e-5,
            "neg_ratio": 4,
            "num_workers": 2,
            "scheduler_patience": 2,
            "scheduler_factor": 0.5,
            "min_lr": 1e-6,
        },
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "training-twoheads"},
            {"Key": "pipeline", "Value": PIPELINE_NAME},
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

    # Dependencia: requiere Data Splits completado
    step_train_twoheads.add_depends_on([step_splits])

    return step_train_twoheads


# ==============================================================================
# STEP 6: EVALUATION — Evaluación Comparativa de Modelos
# ==============================================================================


def create_step_evaluation(
    params, role, session, step_train_regression, step_train_twoheads
):
    """
    Evaluation Job: Evalúa ambos modelos sobre test + cold-start.
    - Métricas: RMSE, NDCG@K, HR@K, Precision@K, Recall@K, F1
    - Genera reportes JSON con métricas completas
    - Selecciona modelo ganador automáticamente (hybrid score)
    - Output: evaluation_report.json con rmse_stars del ganador

    DEPENDENCIA: Requiere que AMBOS Training Jobs hayan completado.
    """
    eval_processor = PyTorchProcessor(
        role=role,
        instance_type=params["eval_instance_type"],
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="hymmrec-evaluation",
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "evaluation"},
            {"Key": "pipeline", "Value": PIPELINE_NAME},
        ],
    )

    # PropertyFile para capturar métricas y usarlas en el ConditionStep
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
        code=os.path.join(EVALUATION_SCRIPTS_DIR, "evaluation-job.py"),
        job_arguments=["--k", "10", "--num-decoys", "99"],
        property_files=[evaluation_report],
    )

    return step_eval, evaluation_report


# ==============================================================================
# STEP 7: MODEL REGISTRY — Registro del Modelo Ganador
# ==============================================================================


def create_step_register_model(
    params, role, session, step_eval, step_train_regression, step_train_twoheads
):
    """
    Model Registry: Registra el modelo ganador en SageMaker Model Registry.
    - Adjunta métricas de evaluación como metadata
    - Status: PendingManualApproval (requiere aprobación humana)
    - Soporte para deploy en ml.m5.large / ml.c5.large

    NOTA: Se registran AMBOS modelos (regresión y two-heads) como candidatos.
    El evaluation job determina cuál es el ganador en los metadatos.
    """
    # Imagen de inferencia PyTorch
    inference_image_uri = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region="us-east-1",
        version="2.1",
        py_version="py310",
        instance_type="ml.m5.large",
        image_scope="inference",
    )

    # Registrar modelo Two-Heads (modelo principal para retrieval + calidad)
    step_register_twoheads = RegisterModel(
        name="RegisterModelTwoHeads",
        estimator=PyTorch(
            entry_point="train_hymmrec_twoheads.py",
            source_dir=TRAINING_SCRIPTS_DIR,
            role=role,
            instance_type=params["training_instance_type"],
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
            "HYMM-REC Two-Heads: Sistema recomendador híbrido multimodal. "
            "Arquitectura Two-Tower con atención multimodal explicable. "
            "Cabeza ranking (BCE) + Cabeza calidad (MSE enmascarado)."
        ),
    )

    # Registrar modelo Regresión (alternativa para rating prediction puro)
    step_register_regression = RegisterModel(
        name="RegisterModelRegression",
        estimator=PyTorch(
            entry_point="train_hymmrec_regression.py",
            source_dir=TRAINING_SCRIPTS_DIR,
            role=role,
            instance_type=params["training_instance_type"],
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
            "HYMM-REC Regression: Modelo de predicción de rating puro. "
            "Single-Head con MSELoss sobre rating escalado [0,1]."
        ),
    )

    return step_register_twoheads, step_register_regression


# ==============================================================================
# STEP 8: MODEL PACKAGING — Extracción de Torres Independientes
# ==============================================================================


def create_step_model_packaging(
    params, role, session, step_register_twoheads, step_eval
):
    """
    Model Packaging: Extrae UserTower, ItemTower y FullModel como artefactos
    separados, cada uno con su inference.py para deploy independiente.
    - Full Model → Endpoint real-time (predicción completa)
    - User Tower → Endpoint real-time (embedding de usuario para OpenSearch)
    - Item Tower → Batch Transform offline (índice de embeddings)

    DEPENDENCIA: Requiere que Model Registry haya completado.
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
            {"Key": "pipeline", "Value": PIPELINE_NAME},
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
                source=f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/models/twoheads/",
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
        code=os.path.join(EVALUATION_SCRIPTS_DIR, "model-packaging-job.py"),
    )

    # Dependencia: requiere que el registro del modelo haya completado
    step_packaging.add_depends_on([step_register_twoheads])

    return step_packaging


# ==============================================================================
# STEP 9: CONDITION — Quality Gate (RMSE Threshold)
# ==============================================================================


def create_condition_quality_gate(
    params,
    evaluation_report,
    step_eval,
    step_register_twoheads,
    step_register_regression,
    step_packaging,
):
    """
    Quality Gate: Condición que verifica si el modelo ganador cumple con
    el umbral de calidad mínimo (RMSE en estrellas <= threshold).

    Si PASA → Registra ambos modelos en Model Registry + Packaging
    Si FALLA → FailStep con mensaje descriptivo
    """
    # Condición: RMSE del modelo ganador <= umbral máximo
    cond_rmse = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="winner_rmse_stars",
        ),
        right=params["max_rmse_threshold"],
    )

    # Paso de fallo si no supera el quality gate
    step_fail = FailStep(
        name="QualityGateFailed",
        error_message=(
            "El modelo ganador NO supera el umbral de calidad. "
            "RMSE en estrellas supera el máximo permitido. "
            "Revisa los hiperparámetros, datos de entrenamiento, o ajusta el threshold."
        ),
    )

    # ConditionStep: bifurcación del DAG
    step_condition = ConditionStep(
        name="QualityGateCheck",
        conditions=[cond_rmse],
        if_steps=[step_register_twoheads, step_register_regression, step_packaging],
        else_steps=[step_fail],
    )

    return step_condition


# ==============================================================================
# DEFINICIÓN DEL PIPELINE COMPLETO
# ==============================================================================


def build_pipeline(
    region: str = "us-east-1",
    role_arn: Optional[str] = None,
    pipeline_name: str = PIPELINE_NAME,
) -> Pipeline:
    """
    Construye el SageMaker Pipeline completo con el DAG:

      FeatureEngineering ──┐
                           ├──► DatasetPreparation ──► TrainingRegression  ──┐
      EmbeddingsGeneration ┘                      └──► TrainingTwoHeads  ───┤
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

    Returns:
        Pipeline: Objeto SageMaker Pipeline listo para upsert/start.
    """
    # --- Sesión y Role ---
    boto_session = boto3.Session(region_name=region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    if role_arn:
        role = role_arn
    else:
        try:
            role = get_execution_role()
        except ValueError:
            # Fuera de SageMaker notebook, intentar obtener del env
            role = os.environ.get(
                "SAGEMAKER_ROLE_ARN",
                "arn:aws:iam::697682206292:role/sgmkr-notebook-tfm-hymm-rec-ml-iar-dev",
            )

    logger.info(f"Construyendo pipeline: {pipeline_name}")
    logger.info(f"Region: {region}")
    logger.info(f"Role: {role}")

    # --- Parámetros ---
    params = define_pipeline_parameters()

    # --- Step 1: Feature Engineering (PySpark) ---
    step_feng = create_step_feature_engineering(params, role, sm_session)
    logger.info("✓ Step definido: FeatureEngineering")

    # --- Step 2: Embeddings Generation (SKLearn + Bedrock) ---
    step_embeddings = create_step_embeddings(params, role, sm_session)
    logger.info("✓ Step definido: EmbeddingsGeneration")

    # --- Step 3: Dataset Preparation (SKLearn) ---
    step_splits = create_step_data_splits(params, role, sm_session, step_feng, step_embeddings)
    logger.info("✓ Step definido: DatasetPreparation")

    # --- Step 4: Training Regression ---
    step_train_regression = create_step_training_regression(
        params, role, sm_session, step_splits
    )
    logger.info("✓ Step definido: TrainingRegression")

    # --- Step 5: Training Two-Heads ---
    step_train_twoheads = create_step_training_twoheads(
        params, role, sm_session, step_splits
    )
    logger.info("✓ Step definido: TrainingTwoHeads")

    # --- Step 6: Evaluation ---
    step_eval, evaluation_report = create_step_evaluation(
        params, role, sm_session, step_train_regression, step_train_twoheads
    )
    logger.info("✓ Step definido: ModelEvaluation")

    # --- Step 7: Model Registry ---
    step_register_twoheads, step_register_regression = create_step_register_model(
        params, role, sm_session, step_eval, step_train_regression, step_train_twoheads
    )
    logger.info("✓ Step definido: RegisterModel (TwoHeads + Regression)")

    # --- Step 8: Model Packaging ---
    step_packaging = create_step_model_packaging(
        params, role, sm_session, step_register_twoheads, step_eval
    )
    logger.info("✓ Step definido: ModelPackaging")

    # --- Step 9: Quality Gate (Condition) ---
    step_condition = create_condition_quality_gate(
        params,
        evaluation_report,
        step_eval,
        step_register_twoheads,
        step_register_regression,
        step_packaging,
    )
    logger.info("✓ Step definido: QualityGateCheck")

    # --- Ensamblar Pipeline ---
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
            step_train_regression,
            step_train_twoheads,
            step_eval,
            step_condition,
        ],
        sagemaker_session=sm_session,
    )

    logger.info(f"Pipeline '{pipeline_name}' construido con {len(pipeline.steps)} steps.")
    return pipeline


# ==============================================================================
# EXPORTAR JSON / UPSERT / EJECUTAR
# ==============================================================================


def export_pipeline_json(pipeline: Pipeline, output_path: str) -> dict:
    """Exporta la definición del pipeline como JSON."""
    definition = json.loads(pipeline.definition())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(definition, f, indent=2, ensure_ascii=False)
    logger.info(f"Pipeline JSON exportado a: {output_path}")
    return definition


def upsert_pipeline(pipeline: Pipeline, role_arn: str) -> str:
    """Crea o actualiza el pipeline en SageMaker."""
    response = pipeline.upsert(role_arn=role_arn)
    pipeline_arn = response["PipelineArn"]
    logger.info(f"Pipeline creado/actualizado: {pipeline_arn}")
    return pipeline_arn


def start_pipeline(pipeline: Pipeline, parameters: Optional[dict] = None) -> str:
    """Inicia una ejecución del pipeline."""
    execution = pipeline.start(parameters=parameters or {})
    execution_arn = execution.arn
    logger.info(f"Pipeline ejecutándose: {execution_arn}")
    logger.info(f"Estado: {execution.describe()['PipelineExecutionStatus']}")
    return execution_arn


# ==============================================================================
# MAIN — CLI
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Define y gestiona el SageMaker Pipeline de HYMM-REC MLOps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Exportar JSON de definición:
  python define_sagemaker_pipeline.py --export-json pipeline_definition.json

  # Crear/actualizar pipeline en SageMaker:
  python define_sagemaker_pipeline.py --upsert

  # Crear y ejecutar inmediatamente:
  python define_sagemaker_pipeline.py --execute

  # Con parámetros personalizados:
  python define_sagemaker_pipeline.py --execute --region us-east-1 \\
      --role-arn arn:aws:iam::697682206292:role/sgmkr-notebook-tfm-hymm-rec-ml-iar-dev
        """,
    )

    parser.add_argument(
        "--region", type=str, default="us-east-1", help="AWS Region (default: us-east-1)"
    )
    parser.add_argument(
        "--role-arn", type=str, default=None, help="SageMaker Execution Role ARN"
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default=PIPELINE_NAME,
        help=f"Nombre del pipeline (default: {PIPELINE_NAME})",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Exportar definición JSON del pipeline a un archivo",
    )
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Crear o actualizar el pipeline en SageMaker",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Crear/actualizar Y ejecutar el pipeline",
    )

    args = parser.parse_args()

    # Construir pipeline
    pipeline = build_pipeline(
        region=args.region,
        role_arn=args.role_arn,
        pipeline_name=args.pipeline_name,
    )

    # Determinar role para upsert
    role = args.role_arn or os.environ.get(
        "SAGEMAKER_ROLE_ARN",
        "arn:aws:iam::697682206292:role/sgmkr-notebook-tfm-hymm-rec-ml-iar-dev",
    )

    # Exportar JSON
    if args.export_json:
        export_pipeline_json(pipeline, args.export_json)

    # Si no se especificó ninguna acción, exportar JSON por defecto
    if not args.export_json and not args.upsert and not args.execute:
        default_path = "pipeline_definition.json"
        definition = export_pipeline_json(pipeline, default_path)
        print(f"\n{'='*60}")
        print(f"PIPELINE DEFINITION GENERADA")
        print(f"{'='*60}")
        print(f"  Archivo: {default_path}")
        print(f"  Pipeline: {args.pipeline_name}")
        print(f"  Steps: {len(definition.get('Steps', []))}")
        print(f"  Parameters: {len(definition.get('Parameters', []))}")
        print(f"\nDAG del Pipeline:")
        print(f"  1. FeatureEngineering (PySpark)")
        print(f"  2. EmbeddingsGeneration (SKLearn + Bedrock Nova)")
        print(f"  3. DatasetPreparation (SKLearn) [depends: 1, 2]")
        print(f"  4. TrainingRegression (PyTorch GPU) [depends: 3]")
        print(f"  5. TrainingTwoHeads (PyTorch GPU) [depends: 3]")
        print(f"  6. ModelEvaluation (PyTorch GPU) [depends: 4, 5]")
        print(f"  7. QualityGateCheck (Condition: RMSE <= threshold)")
        print(f"     ├─ PASS: RegisterModel (TwoHeads + Regression) → ModelPackaging")
        print(f"     └─ FAIL: QualityGateFailed (FailStep)")
        print(f"\nPara crear el pipeline en SageMaker:")
        print(f"  python define_sagemaker_pipeline.py --upsert")
        print(f"\nPara ejecutar:")
        print(f"  python define_sagemaker_pipeline.py --execute")
        return

    # Upsert
    if args.upsert or args.execute:
        pipeline_arn = upsert_pipeline(pipeline, role)
        print(f"\n✅ Pipeline registrado: {pipeline_arn}")

    # Ejecutar
    if args.execute:
        execution_arn = start_pipeline(pipeline)
        print(f"\n🚀 Pipeline en ejecución: {execution_arn}")
        print(f"   Monitor en SageMaker Studio → Pipelines → {args.pipeline_name}")


if __name__ == "__main__":
    main()
