"""
==============================================================================
HYMM-REC MLOps Pipeline: Invocación de Training Jobs desde SageMaker Notebook
==============================================================================
Este script simula las celdas de un notebook para ejecutar los training/HPO jobs
del sistema recomendador híbrido multimodal.

Modelos entrenados:
  1. Regresión (Single-Head): Predice rating escalado → MSELoss
  2. Multi-Task Two-Heads: Predice interacción + rating → BCE + MSE(enmascarado)

Flujo:
  1. Configuración y upload de scripts de training a S3
  2. HPO Job (Regresión): Busca mejores hyperparámetros para rating prediction
  3. Training Job (Regresión): Entrena modelo final con mejores HPs
  4. HPO Job (Two-Heads): Busca mejores HPs para multi-task (retrieval + calidad)
  5. Training Job (Two-Heads): Entrena modelo final con mejores HPs
  6. Comparación de resultados y selección del modelo ganador

Pre-requisitos:
  - Processing Jobs completados (datasets en Platinum bucket)
  - Embeddings generados (embeddings_catalog.pkl en Platinum)
  - SageMaker Execution Role con permisos a S3 y EC2 (GPU instances)
  - Datasets: train/, val/, test/ parquets en S3_PLATINUM_DATASETS
"""

# ==============================================================================
# CELDA 1: CONFIGURACIÓN GENERAL
# ==============================================================================
import boto3
import sagemaker
import os
import json
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import (
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
    IntegerParameter,
)

# Configuración base
REGION = "us-east-1"
ROLE = get_execution_role()
SESSION = sagemaker.Session()

# Buckets
PLATINUM_BUCKET = "hymmrec-sagemaker-assets"
GOLD_BUCKET = "hymmrec-dilkehousegold01"

# Paths S3 (Outputs de los Processing Jobs)
S3_PLATINUM_DATASETS = f"s3://{PLATINUM_BUCKET}/hymmrec/datasets/"
S3_PLATINUM_EMBEDDINGS = f"s3://{PLATINUM_BUCKET}/hymmrec/model_artefacts/embeddings/"
S3_PLATINUM_ENCODERS = f"s3://{PLATINUM_BUCKET}/hymmrec/model_artefacts/encoders/"

# Outputs de Training
S3_MODEL_OUTPUT = f"s3://{PLATINUM_BUCKET}/hymmrec/models/"
S3_TRAINING_SCRIPTS = f"s3://{PLATINUM_BUCKET}/sagemaker-scripts/training/"

# Instancia de Training (GPU)
TRAINING_INSTANCE_TYPE = "ml.g4dn.xlarge"  # 1x T4 GPU, 16GB RAM, 4 vCPU
HPO_INSTANCE_TYPE = "ml.g4dn.xlarge"

# Scripts locales (relativos al notebook)
LOCAL_TRAINING_DIR = "../dev/training/"

print(f"Role: {ROLE}")
print(f"Region: {REGION}")
print(f"Datasets: {S3_PLATINUM_DATASETS}")
print(f"Embeddings: {S3_PLATINUM_EMBEDDINGS}")
print(f"Model Output: {S3_MODEL_OUTPUT}")
print(f"Training Instance: {TRAINING_INSTANCE_TYPE}")


# ==============================================================================
# CELDA 2: UPLOAD SCRIPTS DE TRAINING A S3
# ==============================================================================
s3_client = boto3.client("s3")

training_scripts = [
    "nn_hymmrec.py",
    "dataloaders.py",
    "train_hymmrec_regression.py",
    "hpo_hymmrec_regression.py",
    "train_hymmrec_twoheads.py",
    "hpo_hymmrec_twoheads.py",
]

for script in training_scripts:
    local_path = os.path.join(LOCAL_TRAINING_DIR, script)
    s3_key = f"sagemaker-scripts/training/{script}"
    s3_client.upload_file(local_path, PLATINUM_BUCKET, s3_key)
    print(f"Uploaded: {local_path} → s3://{PLATINUM_BUCKET}/{s3_key}")

print("\nTodos los scripts subidos correctamente.")


# ==============================================================================
# CELDA 3: VERIFICAR QUE LOS DATASETS EXISTEN
# ==============================================================================
print("\nVerificando datasets en S3...")

s3 = boto3.resource("s3")
bucket = s3.Bucket(PLATINUM_BUCKET)

# Verificar splits
for split in ["train", "val", "test"]:
    prefix = f"hymmrec/datasets/{split}/"
    objects = list(bucket.objects.filter(Prefix=prefix).limit(3))
    if objects:
        total_size = sum(obj.size for obj in objects)
        print(f"  ✅ {split}/: {len(objects)} archivos ({total_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"  ❌ {split}/: NO ENCONTRADO - Ejecutar Processing Jobs primero")

# Verificar embeddings
emb_prefix = "hymmrec/model_artefacts/embeddings/"
emb_objects = list(bucket.objects.filter(Prefix=emb_prefix).limit(3))
if emb_objects:
    print(f"  ✅ embeddings/: {len(emb_objects)} archivos")
else:
    print(f"  ❌ embeddings/: NO ENCONTRADO")


# ==============================================================================
# CELDA 4: HPO JOB — REGRESIÓN (Rating Prediction)
# ==============================================================================
# Busca la mejor combinación de hyperparámetros para el modelo de regresión.
# Single-Head: una sola cabeza que predice rating escalado [0,1] con MSELoss.

print("\n" + "=" * 60)
print("HPO JOB: REGRESIÓN (Rating Prediction — Single-Head)")
print("=" * 60)

# Estimador base para HPO de regresión
regression_estimator = PyTorch(
    entry_point="hpo_hymmrec_regression.py",
    source_dir=LOCAL_TRAINING_DIR,
    role=ROLE,
    instance_type=HPO_INSTANCE_TYPE,
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    sagemaker_session=SESSION,
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
)

# Definir rangos de hyperparámetros
regression_hp_ranges = {
    "lr": ContinuousParameter(0.0001, 0.01, scaling_type="Logarithmic"),
    "batch_size": CategoricalParameter([128, 256, 512]),
    "emb_dim": CategoricalParameter([32, 64, 128]),
    "dropout": ContinuousParameter(0.1, 0.5),
    "weight_decay": ContinuousParameter(1e-6, 1e-3, scaling_type="Logarithmic"),
}

# Métrica objetivo
regression_objective_metric = {
    "Name": "val_rmse_stars",
    "Regex": r"val_rmse_stars=(.*?);",
}

# Crear Tuner
regression_tuner = HyperparameterTuner(
    estimator=regression_estimator,
    objective_metric_name=regression_objective_metric["Name"],
    hyperparameter_ranges=regression_hp_ranges,
    metric_definitions=[
        regression_objective_metric,
        {"Name": "train_mse", "Regex": r"train_mse=(.*?);"},
        {"Name": "val_mse", "Regex": r"val_mse=(.*?);"},
        {"Name": "test_rmse_stars", "Regex": r"test_rmse_stars=(.*?);"},
        {"Name": "lr", "Regex": r"lr=(.*?);"},
    ],
    objective_type="Minimize",
    max_jobs=12,
    max_parallel_jobs=3,
    strategy="Bayesian",
    base_tuning_job_name="hymmrec-hpo-reg",
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "hpo-regression"},
    ],
)

# Lanzar HPO
regression_tuner.fit(
    inputs={
        "train": S3_PLATINUM_DATASETS,
        "embeddings": S3_PLATINUM_EMBEDDINGS,
    },
    wait=True,
    logs="All",
)

print("✅ HPO Regresión completado.")

# Obtener mejores hyperparámetros
best_regression_hp = regression_tuner.best_training_job()
print(f"\nMejor job: {best_regression_hp}")

regression_tuner_analytics = regression_tuner.analytics()
best_regression_result = regression_tuner_analytics.dataframe().sort_values(
    "FinalObjectiveValue"
).iloc[0]
print(f"Mejor val_rmse_stars: {best_regression_result['FinalObjectiveValue']:.4f}")
print(f"Hyperparámetros óptimos:")
for col in regression_tuner_analytics.dataframe().columns:
    if col.startswith("lr") or col.startswith("batch") or col.startswith("emb") or col.startswith("dropout") or col.startswith("weight"):
        print(f"  {col}: {best_regression_result[col]}")


# ==============================================================================
# CELDA 5: TRAINING JOB FINAL — REGRESIÓN (con mejores HPs)
# ==============================================================================
print("\n" + "=" * 60)
print("TRAINING JOB FINAL: REGRESIÓN (Single-Head)")
print("=" * 60)

# Extraer mejores HPs del tuner
best_reg_hps = regression_tuner.best_estimator().hyperparameters()

regression_final_estimator = PyTorch(
    entry_point="train_hymmrec_regression.py",
    source_dir=LOCAL_TRAINING_DIR,
    role=ROLE,
    instance_type=TRAINING_INSTANCE_TYPE,
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    sagemaker_session=SESSION,
    base_job_name="hymmrec-train-regression",
    output_path=f"{S3_MODEL_OUTPUT}regression/",
    hyperparameters={
        "epochs": 30,  # Más épocas para entrenamiento final
        "patience": 5,  # Más paciencia
        "batch_size": int(best_reg_hps.get("batch_size", 256)),
        "lr": float(best_reg_hps.get("lr", 0.001)),
        "emb_dim": int(best_reg_hps.get("emb_dim", 64)),
        "dropout": float(best_reg_hps.get("dropout", 0.3)),
        "weight_decay": float(best_reg_hps.get("weight_decay", 1e-5)),
        "num_workers": 2,
        # LR Scheduler
        "scheduler_patience": 2,
        "scheduler_factor": 0.5,
        "min_lr": 1e-6,
    },
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "training-regression-final"},
    ],
)

regression_final_estimator.fit(
    inputs={
        "train": S3_PLATINUM_DATASETS,
        "embeddings": S3_PLATINUM_EMBEDDINGS,
    },
    wait=True,
    logs="All",
)

print("✅ Training Regresión completado.")
print(f"Modelo: {regression_final_estimator.model_data}")


# ==============================================================================
# CELDA 6: HPO JOB — MULTI-TASK TWO-HEADS (Retrieval + Calidad)
# ==============================================================================
# Busca la mejor combinación de hyperparámetros para el modelo de dos cabezas.
# Cabeza 1: BCE (ranking/retrieval) sobre TODOS los datos
# Cabeza 2: MSE (calidad) ENMASCARADO solo sobre interacciones positivas
# Loss Total = BCE + MSE(positivos)

print("\n" + "=" * 60)
print("HPO JOB: MULTI-TASK TWO-HEADS (Retrieval + Calidad)")
print("=" * 60)

twoheads_estimator = PyTorch(
    entry_point="hpo_hymmrec_twoheads.py",
    source_dir=LOCAL_TRAINING_DIR,
    role=ROLE,
    instance_type=HPO_INSTANCE_TYPE,
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    sagemaker_session=SESSION,
    base_job_name="hymmrec-hpo-twoheads",
    hyperparameters={
        "epochs": 15,
        "patience": 2,
        "num_workers": 2,
    },
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "hpo-twoheads"},
    ],
)

# Rangos de HPs para two-heads (incluye num_negatives)
twoheads_hp_ranges = {
    "lr": ContinuousParameter(0.0001, 0.01, scaling_type="Logarithmic"),
    "batch_size": CategoricalParameter([128, 256, 512]),
    "emb_dim": CategoricalParameter([32, 64, 128]),
    "dropout": ContinuousParameter(0.1, 0.5),
    "weight_decay": ContinuousParameter(1e-6, 1e-3, scaling_type="Logarithmic"),
    "num_negatives": CategoricalParameter([10, 20, 50]),
}

twoheads_objective_metric = {
    "Name": "val_total_loss",
    "Regex": r"val_total_loss=(.*?);",
}

twoheads_tuner = HyperparameterTuner(
    estimator=twoheads_estimator,
    objective_metric_name=twoheads_objective_metric["Name"],
    hyperparameter_ranges=twoheads_hp_ranges,
    metric_definitions=[
        twoheads_objective_metric,
        {"Name": "train_total_loss", "Regex": r"train_total_loss=(.*?);"},
        {"Name": "val_bce", "Regex": r"val_bce=(.*?);"},
        {"Name": "val_mse", "Regex": r"val_mse=(.*?);"},
        {"Name": "val_rmse_stars", "Regex": r"val_rmse_stars=(.*?);"},
        {"Name": "test_total_loss", "Regex": r"test_total_loss=(.*?);"},
        {"Name": "test_rmse_stars", "Regex": r"test_rmse_stars=(.*?);"},
        {"Name": "lr", "Regex": r"lr=(.*?);"},
    ],
    objective_type="Minimize",
    max_jobs=16,  # Más jobs porque tenemos un HP extra (num_negatives)
    max_parallel_jobs=4,
    strategy="Bayesian",
    base_tuning_job_name="hymmrec-hpo-th",
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "hpo-twoheads"},
    ],
)

twoheads_tuner.fit(
    inputs={
        "train": S3_PLATINUM_DATASETS,
        "embeddings": S3_PLATINUM_EMBEDDINGS,
    },
    wait=True,
    logs="All",
)

print("✅ HPO Two-Heads completado.")

best_twoheads_hp = twoheads_tuner.best_training_job()
print(f"\nMejor job: {best_twoheads_hp}")

twoheads_tuner_analytics = twoheads_tuner.analytics()
best_th_result = twoheads_tuner_analytics.dataframe().sort_values(
    "FinalObjectiveValue"
).iloc[0]
print(f"Mejor val_total_loss: {best_th_result['FinalObjectiveValue']:.6f}")
print(f"Hyperparámetros óptimos:")
for col in twoheads_tuner_analytics.dataframe().columns:
    if any(col.startswith(p) for p in ["lr", "batch", "emb", "dropout", "weight", "num_neg"]):
        print(f"  {col}: {best_th_result[col]}")


# ==============================================================================
# CELDA 7: TRAINING JOB FINAL — MULTI-TASK TWO-HEADS (con mejores HPs)
# ==============================================================================
print("\n" + "=" * 60)
print("TRAINING JOB FINAL: MULTI-TASK TWO-HEADS")
print("=" * 60)

best_th_hps = twoheads_tuner.best_estimator().hyperparameters()

twoheads_final_estimator = PyTorch(
    entry_point="train_hymmrec_twoheads.py",
    source_dir=LOCAL_TRAINING_DIR,
    role=ROLE,
    instance_type=TRAINING_INSTANCE_TYPE,
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    sagemaker_session=SESSION,
    base_job_name="hymmrec-train-twoheads",
    output_path=f"{S3_MODEL_OUTPUT}twoheads/",
    hyperparameters={
        "epochs": 30,
        "patience": 5,
        "batch_size": int(best_th_hps.get("batch_size", 256)),
        "lr": float(best_th_hps.get("lr", 0.001)),
        "emb_dim": int(best_th_hps.get("emb_dim", 64)),
        "dropout": float(best_th_hps.get("dropout", 0.3)),
        "weight_decay": float(best_th_hps.get("weight_decay", 1e-5)),
        "num_negatives": int(best_th_hps.get("num_negatives", 20)),
        "num_workers": 2,
        # LR Scheduler
        "scheduler_patience": 2,
        "scheduler_factor": 0.5,
        "min_lr": 1e-6,
    },
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "training-twoheads-final"},
    ],
)

twoheads_final_estimator.fit(
    inputs={
        "train": S3_PLATINUM_DATASETS,
        "embeddings": S3_PLATINUM_EMBEDDINGS,
    },
    wait=True,
    logs="All",
)

print("✅ Training Two-Heads completado.")
print(f"Modelo: {twoheads_final_estimator.model_data}")


# ==============================================================================
# CELDA 8: COMPARACIÓN DE RESULTADOS Y RESUMEN
# ==============================================================================
print("\n" + "=" * 60)
print("RESUMEN DE RESULTADOS")
print("=" * 60)

print("\n📊 Modelo 1: REGRESIÓN (Single-Head)")
print(f"   - Artifact: {regression_final_estimator.model_data}")
reg_hp_summary = regression_final_estimator.hyperparameters()
print(f"   - LR: {reg_hp_summary.get('lr')}")
print(f"   - Emb Dim: {reg_hp_summary.get('emb_dim')}")
print(f"   - Dropout: {reg_hp_summary.get('dropout')}")
print(f"   - Batch: {reg_hp_summary.get('batch_size')}")
print(f"   - Objetivo: Minimizar RMSE en estrellas (rating prediction)")

print("\n📊 Modelo 2: MULTI-TASK TWO-HEADS (Retrieval + Calidad)")
print(f"   - Artifact: {twoheads_final_estimator.model_data}")
th_hp_summary = twoheads_final_estimator.hyperparameters()
print(f"   - LR: {th_hp_summary.get('lr')}")
print(f"   - Emb Dim: {th_hp_summary.get('emb_dim')}")
print(f"   - Dropout: {th_hp_summary.get('dropout')}")
print(f"   - Batch: {th_hp_summary.get('batch_size')}")
print(f"   - Num Negatives: {th_hp_summary.get('num_negatives')}")
print(f"   - Objetivo: Minimizar BCE(ranking) + MSE(calidad)")

print("\n" + "=" * 60)
print("PIPELINE DE TRAINING COMPLETADO")
print("=" * 60)
print(f"\nPróximos pasos:")
print(f"  1. Evaluar métricas finales de ambos modelos")
print(f"  2. Comparar:")
print(f"     - Regresión: Mejor en RMSE puro (predicción de rating)")
print(f"     - Two-Heads: Mejor en ranking (retrieval) + calidad simultánea")
print(f"  3. Seleccionar modelo según caso de uso:")
print(f"     - Si el objetivo es SOLO predecir ratings → Regresión")
print(f"     - Si el objetivo es recomendar Y predecir calidad → Two-Heads")
print(f"  4. Registrar modelo ganador en SageMaker Model Registry")
print(f"  5. Deploy a endpoint de inferencia")


# ==============================================================================
# CELDA 9 (OPCIONAL): VISUALIZACIÓN DE CURVAS DE ENTRENAMIENTO
# ==============================================================================
# Descomenta esta celda para visualizar las curvas de entrenamiento
# después de descargar los artefactos del modelo.

# import matplotlib.pyplot as plt
#
# def plot_multitask_curves(metrics_dict, title):
#     """Plotea las curvas de entrenamiento para modelo multi-task."""
#     history = metrics_dict.get("training_history", {})
#
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#
#     # Total Loss
#     axes[0].plot(history["train_total"], label="Train Total", marker="o")
#     axes[0].plot(history["val_total"], label="Val Total", marker="s")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("Total Loss (BCE + MSE)")
#     axes[0].set_title(f"{title} - Total Loss")
#     axes[0].legend()
#     axes[0].grid(True)
#
#     # BCE (Ranking)
#     axes[1].plot(history["train_bce"], label="Train BCE", marker="o", color="red")
#     axes[1].plot(history["val_bce"], label="Val BCE", marker="s", color="darkred")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("BCE Loss")
#     axes[1].set_title(f"{title} - Cabeza Ranking (BCE)")
#     axes[1].legend()
#     axes[1].grid(True)
#
#     # MSE (Calidad) + RMSE Stars
#     ax3 = axes[2]
#     ax3.plot(history["train_mse"], label="Train MSE", marker="o", color="green")
#     ax3.plot(history["val_mse"], label="Val MSE", marker="s", color="darkgreen")
#     ax3_twin = ax3.twinx()
#     ax3_twin.plot(history["val_rmse_stars"], label="Val RMSE Stars", marker="^",
#                   color="orange", linestyle="--")
#     ax3.set_xlabel("Epoch")
#     ax3.set_ylabel("MSE Loss")
#     ax3_twin.set_ylabel("RMSE (Estrellas)")
#     ax3.set_title(f"{title} - Cabeza Calidad (MSE)")
#     ax3.legend(loc="upper left")
#     ax3_twin.legend(loc="upper right")
#     ax3.grid(True)
#
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_regression_curves(metrics_dict, title):
#     """Plotea las curvas de entrenamiento para modelo de regresión."""
#     history = metrics_dict.get("training_history", {})
#
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
#     # MSE
#     axes[0].plot(history["train_mse"], label="Train MSE", marker="o")
#     axes[0].plot(history["val_mse"], label="Val MSE", marker="s")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("MSE")
#     axes[0].set_title(f"{title} - Learning Curve (MSE)")
#     axes[0].legend()
#     axes[0].grid(True)
#
#     # RMSE Stars
#     axes[1].plot(history["val_rmse_stars"], label="Val RMSE Stars", marker="s", color="orange")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("RMSE (Estrellas)")
#     axes[1].set_title(f"{title} - RMSE en Escala Real")
#     axes[1].legend()
#     axes[1].grid(True)
#
#     plt.tight_layout()
#     plt.show()
