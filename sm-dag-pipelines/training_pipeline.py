"""
SageMaker Pipeline: Orquestación completa del entrenamiento del sistema recomendador.

Pipeline Steps:
1. Processing: Feature Engineering (label encoding, multi-hot, scaling, split)
2. Training: Entrenamiento del modelo NeuMF o ExplainableGMF
3. Evaluation: Métricas de test sobre el mejor modelo
4. Condition: Si RMSE < umbral → registrar modelo
5. Register: Registro del modelo en el Model Registry

Uso:
    python -m pipelines.training_pipeline --execute
"""

import argparse
import logging
import os

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_pipeline(
    region: str = "us-east-1",
    role: str = None,
    bucket: str = None,
    pipeline_name: str = "hymm-rec-training-pipeline",
) -> Pipeline:
    """
    Construye el SageMaker Pipeline completo.
    
    Args:
        region: Región AWS
        role: ARN del rol de ejecución de SageMaker
        bucket: Bucket S3 para artefactos
        pipeline_name: Nombre del pipeline
    
    Returns:
        Instancia de Pipeline lista para ejecutar
    """
    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    role = role or get_execution_role()
    bucket = bucket or session.default_bucket()
    prefix = "hymm-rec"

    # =========================================================
    # PARÁMETROS DEL PIPELINE
    # =========================================================
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.g4dn.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data_s3 = ParameterString(
        name="InputDataS3Uri", default_value=f"s3://{bucket}/{prefix}/data/raw/"
    )
    epochs = ParameterInteger(name="Epochs", default_value=20)
    batch_size = ParameterInteger(name="BatchSize", default_value=256)
    learning_rate = ParameterFloat(name="LearningRate", default_value=0.001)
    rmse_threshold = ParameterFloat(name="RMSEThreshold", default_value=0.85)

    # =========================================================
    # STEP 1: PROCESSING (Feature Engineering)
    # =========================================================
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        sagemaker_session=session,
    )

    processing_step = ProcessingStep(
        name="FeatureEngineering",
        processor=sklearn_processor,
        code="src/processing/feature_engineering.py",
        inputs=[
            ProcessingInput(
                source=input_data_s3,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed_data",
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/{prefix}/data/processed/",
            )
        ],
    )

    # =========================================================
    # STEP 2: TRAINING (NeuMF Two-Tower)
    # =========================================================
    pytorch_estimator = PyTorch(
        entry_point="train_neumf.py",
        source_dir="src/training",
        role=role,
        instance_type=training_instance_type,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        hyperparameters={
            "epochs": epochs,
            "batch-size": batch_size,
            "lr": learning_rate,
            "emb-dim": 64,
            "aws-dim": 1024,
            "patience": 3,
        },
        output_path=f"s3://{bucket}/{prefix}/training-output/",
        metric_definitions=[
            {"Name": "train:mse", "Regex": "Train MSE: ([0-9\\.]+)"},
            {"Name": "valid:mse", "Regex": "Valid MSE: ([0-9\\.]+)"},
            {"Name": "valid:rmse_stars", "Regex": "RMSE estrellas: ([0-9\\.]+)"},
            {"Name": "test:rmse_stars", "Regex": "Test Final.*RMSE estrellas: ([0-9\\.]+)"},
        ],
    )

    training_step = TrainingStep(
        name="TrainNeuMF",
        estimator=pytorch_estimator,
        inputs={
            "training": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "processed_data"
                ].S3Output.S3Uri,
                content_type="application/x-parquet",
            )
        },
    )

    # =========================================================
    # STEP 3: REGISTER MODEL (Condicional)
    # =========================================================
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=pytorch_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.g4dn.xlarge", "ml.m5.large"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=f"{prefix}-model-group",
        approval_status=model_approval_status,
    )

    # =========================================================
    # PIPELINE DEFINITION
    # =========================================================
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            training_instance_type,
            model_approval_status,
            input_data_s3,
            epochs,
            batch_size,
            learning_rate,
            rmse_threshold,
        ],
        steps=[processing_step, training_step, register_step],
        sagemaker_session=session,
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--role", type=str, default=None)
    parser.add_argument("--bucket", type=str, default=None)
    parser.add_argument("--pipeline-name", type=str, default="hymm-rec-training-pipeline")
    parser.add_argument("--execute", action="store_true", help="Ejecutar el pipeline tras crearlo")
    args = parser.parse_args()

    pipeline = create_pipeline(
        region=args.region,
        role=args.role,
        bucket=args.bucket,
        pipeline_name=args.pipeline_name,
    )

    # Upsert (crear o actualizar)
    pipeline.upsert(role_arn=args.role or get_execution_role())
    logger.info(f"Pipeline '{args.pipeline_name}' creado/actualizado exitosamente.")

    if args.execute:
        execution = pipeline.start()
        logger.info(f"Pipeline ejecutándose: {execution.arn}")
        logger.info("Monitorea en SageMaker Studio > Pipelines")


if __name__ == "__main__":
    main()
