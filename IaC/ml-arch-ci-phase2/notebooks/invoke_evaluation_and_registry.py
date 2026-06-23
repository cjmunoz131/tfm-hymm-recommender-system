"""
==============================================================================
HYMM-REC MLOps Pipeline: Evaluación, Registry, Packaging y Deploy
==============================================================================
Orquestador para ejecutar desde SageMaker Notebook Instance los pasos
post-training del pipeline MLOps.

Flujo:
  Step 1: Evaluation Job (Processing Job)
    → Evalúa ambos modelos (regresión + two-heads) sobre test + cold-start
    → Genera reportes JSON con métricas completas
    → Selecciona modelo ganador automáticamente

  Step 2: Model Registry
    → Registra el modelo ganador en SageMaker Model Registry
    → Adjunta métricas de evaluación como metadata
    → Status: PendingManualApproval (tú decides si aprobarlo)

  Step 3: Model Packaging (Processing Job)
    → Extrae UserTower, ItemTower, FullModel como artefactos separados
    → Cada uno con su inference.py para deploy independiente

  Step 4: Deploy (Endpoints)
    → Full Model → Endpoint real-time (predicción completa)
    → User Tower → Endpoint real-time (embedding de usuario para OpenSearch)
    → Item Tower → No se despliega aquí (se usa en Batch Transform offline)

Pre-requisitos:
  - Training Jobs completados (modelos en S3_MODEL_OUTPUT)
  - Datasets en Platinum bucket (para evaluación)
  - Model Package Group creado (o se crea automáticamente)
"""

# ==============================================================================
# CELDA 1: CONFIGURACIÓN
# ==============================================================================
import boto3
import sagemaker
import os
import json
import time
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorchModel
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.pytorch.processing import PyTorchProcessor

REGION = "us-east-1"
ROLE = get_execution_role()
SESSION = sagemaker.Session()

# Buckets
PLATINUM_BUCKET = "hymmrec-sagemaker-assets"
GOLD_BUCKET = "hymmrec-dilkehousegold01"

# Paths S3
S3_PLATINUM_DATASETS = f"s3://{PLATINUM_BUCKET}/hymmrec/datasets/"
S3_PLATINUM_EMBEDDINGS = f"s3://{PLATINUM_BUCKET}/hymmrec/model_artefacts/embeddings/"
S3_MODEL_OUTPUT = f"s3://{PLATINUM_BUCKET}/hymmrec/models/"

# Paths de los modelos entrenados (ajustar según output del training)
# Estos paths apuntan al model.tar.gz generado por SageMaker Training Jobs
S3_REGRESSION_MODEL = f"{S3_MODEL_OUTPUT}regression/"
S3_TWOHEADS_MODEL = f"{S3_MODEL_OUTPUT}twoheads/"

# Paths de evaluación
S3_EVAL_REPORTS = f"s3://{PLATINUM_BUCKET}/hymmrec/evaluation/reports/"
S3_EVAL_WINNER = f"s3://{PLATINUM_BUCKET}/hymmrec/evaluation/winner/"
S3_PACKAGED_MODELS = f"s3://{PLATINUM_BUCKET}/hymmrec/packaged-models/"

# Scripts locales
LOCAL_EVAL_DIR = "../dev/evaluation/"

# Model Registry
MODEL_PACKAGE_GROUP_NAME = "hymmrec-multimodal-recommender"

# Instancias
EVAL_INSTANCE_TYPE = "ml.g4dn.xlarge"  # GPU para evaluación rápida
PACKAGING_INSTANCE_TYPE = "ml.m5.large"  # CPU suficiente para packaging
ENDPOINT_INSTANCE_TYPE = "ml.m5.large"  # CPU para inferencia (modelo liviano)

print(f"Role: {ROLE}")
print(f"Region: {REGION}")
print(f"Regression Model: {S3_REGRESSION_MODEL}")
print(f"Two-Heads Model: {S3_TWOHEADS_MODEL}")
print(f"Eval Reports: {S3_EVAL_REPORTS}")


# ==============================================================================
# CELDA 2: UPLOAD SCRIPTS DE EVALUACIÓN A S3
# ==============================================================================
s3_client = boto3.client("s3")

eval_scripts = ["evaluation-job.py", "model-packaging-job.py"]

for script in eval_scripts:
    local_path = os.path.join(LOCAL_EVAL_DIR, script)
    s3_key = f"sagemaker-scripts/evaluation/{script}"
    s3_client.upload_file(local_path, PLATINUM_BUCKET, s3_key)
    print(f"Uploaded: {local_path} → s3://{PLATINUM_BUCKET}/{s3_key}")


# ==============================================================================
# CELDA 3: EXTRAER MODELOS DE LOS TRAINING ARTIFACTS
# ==============================================================================
# Los Training Jobs de SageMaker guardan el modelo como model.tar.gz
# Necesitamos extraer model.pth y model_metadata.json para el evaluation job.
# Subimos los modelos descomprimidos a una ruta conocida.

import tarfile
from io import BytesIO

def extract_model_artifact(model_data_uri, dest_prefix):
    """Descarga model.tar.gz y extrae model.pth + model_metadata.json a S3."""
    # Encontrar el model.tar.gz en el output del training job
    parsed = model_data_uri.replace("s3://", "").split("/", 1)
    bucket_name, prefix = parsed[0], parsed[1]

    # Listar objetos para encontrar el tar.gz
    s3_resource = boto3.resource("s3")
    bucket_obj = s3_resource.Bucket(bucket_name)
    tar_key = None
    for obj in bucket_obj.objects.filter(Prefix=prefix):
        if obj.key.endswith("model.tar.gz"):
            tar_key = obj.key
            break

    if not tar_key:
        print(f"  ⚠️ No se encontró model.tar.gz en {model_data_uri}")
        print(f"  Buscando en prefix: {prefix}")
        # Intentar con el prefix directo como tar
        tar_key = prefix.rstrip("/") + "/output/model.tar.gz"

    print(f"  Descargando: s3://{bucket_name}/{tar_key}")
    obj = s3_client.get_object(Bucket=bucket_name, Key=tar_key)
    tar_bytes = obj["Body"].read()

    # Extraer y subir archivos individuales
    with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                f = tar.extractfile(member)
                if f:
                    dest_key = f"{dest_prefix}{member.name}"
                    s3_client.put_object(
                        Bucket=PLATINUM_BUCKET, Key=dest_key, Body=f.read()
                    )
                    print(f"  Extraído: {member.name} → s3://{PLATINUM_BUCKET}/{dest_key}")


print("\nExtrayendo modelo de Regresión...")
extract_model_artifact(S3_REGRESSION_MODEL, "hymmrec/evaluation/models/regression/")

print("\nExtrayendo modelo Two-Heads...")
extract_model_artifact(S3_TWOHEADS_MODEL, "hymmrec/evaluation/models/twoheads/")

S3_EVAL_MODELS_REG = f"s3://{PLATINUM_BUCKET}/hymmrec/evaluation/models/regression/"
S3_EVAL_MODELS_TH = f"s3://{PLATINUM_BUCKET}/hymmrec/evaluation/models/twoheads/"


# ==============================================================================
# CELDA 4: STEP 1 — EVALUATION JOB (Processing Job)
# ==============================================================================
# Evalúa ambos modelos sobre test + cold-start con métricas completas:
#   - Pointwise: MSE, RMSE, BCE, F1, Accuracy, Precision, Recall
#   - Ranking: HR@K, NDCG@K (con verdaderos negativos)
#   - Top-K: Precision@K, Recall@K
#   - Explicabilidad: Pesos de atención promedio por modalidad
#   - Genera reporte de comparación y selecciona ganador

print("\n" + "=" * 60)
print("STEP 1: EVALUATION JOB")
print("=" * 60)

eval_processor = PyTorchProcessor(
    role=ROLE,
    instance_type=EVAL_INSTANCE_TYPE,
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    sagemaker_session=SESSION,
    base_job_name="hymmrec-evaluation",
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "evaluation"},
    ],
)

eval_processor.run(
    code=f"{LOCAL_EVAL_DIR}evaluation-job.py",
    arguments=[
        "--k", "10",
        "--num-decoys", "99",
    ],
    inputs=[
        ProcessingInput(
            source=S3_EVAL_MODELS_REG,
            destination="/opt/ml/processing/input/models/regression",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
        ProcessingInput(
            source=S3_EVAL_MODELS_TH,
            destination="/opt/ml/processing/input/models/twoheads",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
        ProcessingInput(
            source=S3_PLATINUM_DATASETS,
            destination="/opt/ml/processing/input/datasets",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
        ProcessingInput(
            source=S3_PLATINUM_EMBEDDINGS,
            destination="/opt/ml/processing/input/embeddings",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/reports",
            destination=S3_EVAL_REPORTS,
            output_name="reports",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/winner",
            destination=S3_EVAL_WINNER,
            output_name="winner",
        ),
    ],
    logs=True,
    wait=True,
)

print("✅ Evaluation Job completado.")


# ==============================================================================
# CELDA 5: LEER RESULTADOS DE EVALUACIÓN
# ==============================================================================
print("\nDescargando reportes de evaluación...")

s3_resource = boto3.resource("s3")

# Leer comparación
comparison_obj = s3_resource.Object(
    PLATINUM_BUCKET, "hymmrec/evaluation/reports/model_comparison.json"
)
comparison = json.loads(comparison_obj.get()["Body"].read().decode())

# Leer winner metadata
winner_obj = s3_resource.Object(
    PLATINUM_BUCKET, "hymmrec/evaluation/winner/best_model_metadata.json"
)
winner_meta = json.loads(winner_obj.get()["Body"].read().decode())

print(f"\n{'='*60}")
print(f"RESULTADOS DE EVALUACIÓN")
print(f"{'='*60}")
print(f"\nScore Regresión: {comparison['regression_score']:.4f}")
print(f"Score Two-Heads: {comparison['twoheads_score']:.4f}")
print(f"\n🏆 MODELO GANADOR: {comparison['winner'].upper()}")
print(f"\nMétricas del ganador:")
for k, v in comparison["winner_summary"].items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for k2, v2 in v.items():
            print(f"    {k2}: {v2:.2f}%" if "pct" in k2 else f"    {k2}: {v2}")
    else:
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

WINNER_MODEL = comparison["winner"]
print(f"\nProcediendo con modelo: {WINNER_MODEL}")


# ==============================================================================
# CELDA 6: STEP 2 — MODEL REGISTRY
# ==============================================================================
# Registra el modelo ganador en SageMaker Model Registry con:
#   - Métricas de evaluación adjuntas
#   - Metadata de hyperparámetros y configuración
#   - Status: PendingManualApproval (requiere aprobación humana)

print("\n" + "=" * 60)
print("STEP 2: MODEL REGISTRY")
print("=" * 60)

from sagemaker.model_metrics import MetricsSource, ModelMetrics

# 1. Crear Model Package Group (si no existe)
sm_client = boto3.client("sagemaker")

try:
    sm_client.describe_model_package_group(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME
    )
    print(f"Model Package Group ya existe: {MODEL_PACKAGE_GROUP_NAME}")
except sm_client.exceptions.ClientError:
    sm_client.create_model_package_group(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        ModelPackageGroupDescription=(
            "HYMM-REC: Sistema recomendador híbrido multimodal con "
            "embeddings estructurales, textuales (Bedrock Nova) y visuales. "
            "Arquitectura Two-Tower con atención multimodal explicable."
        ),
        Tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "domain", "Value": "recommender-systems"},
        ],
    )
    print(f"Model Package Group creado: {MODEL_PACKAGE_GROUP_NAME}")

# 2. Determinar URI del modelo ganador
if WINNER_MODEL == "twoheads":
    winner_model_uri = S3_EVAL_MODELS_TH
    winner_report_key = "hymmrec/evaluation/reports/evaluation_twoheads.json"
else:
    winner_model_uri = S3_EVAL_MODELS_REG
    winner_report_key = "hymmrec/evaluation/reports/evaluation_regression.json"

# 3. Subir métricas en formato compatible con Model Registry
metrics_uri = f"s3://{PLATINUM_BUCKET}/{winner_report_key}"

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        content_type="application/json",
        s3_uri=metrics_uri,
    ),
)

# 4. Crear el PyTorchModel para registro
from sagemaker.pytorch import PyTorchModel

# Construir el model_data URI (apunta al tar.gz del modelo ganador)
if WINNER_MODEL == "twoheads":
    model_data_for_registry = f"{S3_TWOHEADS_MODEL}"
else:
    model_data_for_registry = f"{S3_REGRESSION_MODEL}"

# Buscar el model.tar.gz exacto
bucket_obj = s3_resource.Bucket(PLATINUM_BUCKET)
model_tar_key = None
prefix = model_data_for_registry.replace(f"s3://{PLATINUM_BUCKET}/", "")
for obj in bucket_obj.objects.filter(Prefix=prefix):
    if "model.tar.gz" in obj.key:
        model_tar_key = obj.key
        break

if model_tar_key:
    model_data_uri = f"s3://{PLATINUM_BUCKET}/{model_tar_key}"
else:
    # Fallback: usar el path directo
    model_data_uri = f"{model_data_for_registry}output/model.tar.gz"

print(f"Model artifact: {model_data_uri}")

# 5. Registrar en Model Registry
winner_metrics = comparison["winner_summary"]

model_package_response = sm_client.create_model_package(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
    ModelPackageDescription=(
        f"HYMM-REC {WINNER_MODEL} | "
        f"NDCG@10={winner_metrics.get('ndcg_at_k', 0):.4f} | "
        f"HR@10={winner_metrics.get('hit_rate_at_k', 0):.4f} | "
        f"RMSE*={winner_metrics.get('rmse_stars', 0):.4f}"
    ),
    InferenceSpecification={
        "Containers": [
            {
                "Image": sagemaker.image_uris.retrieve(
                    framework="pytorch",
                    region=REGION,
                    version="2.1",
                    py_version="py310",
                    instance_type=ENDPOINT_INSTANCE_TYPE,
                    image_scope="inference",
                ),
                "ModelDataUrl": model_data_uri,
                "Framework": "PYTORCH",
            }
        ],
        "SupportedTransformInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large", "ml.c5.large"],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
    ModelApprovalStatus="PendingManualApproval",
    ModelMetrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": metrics_uri,
            }
        }
    },
    CustomerMetadataProperties={
        "winner_model": WINNER_MODEL,
        "ndcg_at_10": str(round(winner_metrics.get("ndcg_at_k", 0), 4)),
        "hit_rate_at_10": str(round(winner_metrics.get("hit_rate_at_k", 0), 4)),
        "rmse_stars": str(round(winner_metrics.get("rmse_stars", 0), 4)),
        "evaluation_date": winner_meta.get("evaluation_timestamp", ""),
    },
    Tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "model_type", "Value": WINNER_MODEL},
    ],
)

model_package_arn = model_package_response["ModelPackageArn"]
print(f"\n✅ Modelo registrado en Model Registry:")
print(f"   ARN: {model_package_arn}")
print(f"   Status: PendingManualApproval")
print(f"   → Apruébalo en la consola de SageMaker para habilitar deploy")


# ==============================================================================
# CELDA 7: APROBAR MODELO (Manual o Automático)
# ==============================================================================
# Descomenta la siguiente línea para aprobarlo automáticamente.
# En producción esto lo haría un gate de calidad basado en umbrales.

APPROVAL_THRESHOLD_NDCG = 0.3  # Mínimo NDCG@10 para auto-aprobar
APPROVAL_THRESHOLD_RMSE = 2.0  # Máximo RMSE en estrellas

ndcg_ok = winner_metrics.get("ndcg_at_k", 0) >= APPROVAL_THRESHOLD_NDCG
rmse_ok = winner_metrics.get("rmse_stars", 99) <= APPROVAL_THRESHOLD_RMSE

if ndcg_ok and rmse_ok:
    print(f"\n✅ Métricas superan umbrales (NDCG>={APPROVAL_THRESHOLD_NDCG}, RMSE<={APPROVAL_THRESHOLD_RMSE})")
    print("   Aprobando modelo automáticamente...")

    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Approved",
    )
    print(f"   → Modelo APROBADO: {model_package_arn}")
else:
    print(f"\n⚠️ Métricas NO superan umbrales:")
    print(f"   NDCG@10={winner_metrics.get('ndcg_at_k', 0):.4f} (umbral: {APPROVAL_THRESHOLD_NDCG})")
    print(f"   RMSE*={winner_metrics.get('rmse_stars', 0):.4f} (umbral: {APPROVAL_THRESHOLD_RMSE})")
    print(f"   → Modelo queda como PendingManualApproval")
    print(f"   → Revisa en consola: SageMaker → Model Registry → {MODEL_PACKAGE_GROUP_NAME}")


# ==============================================================================
# CELDA 8: STEP 3 — MODEL PACKAGING (Processing Job)
# ==============================================================================
# Extrae UserTower, ItemTower y FullModel como artefactos separados.
# Cada uno empaquetado con inference.py para deploy independiente.

print("\n" + "=" * 60)
print("STEP 3: MODEL PACKAGING")
print("=" * 60)

# Usar la ruta del modelo ganador extraído
winner_model_s3 = S3_EVAL_MODELS_TH if WINNER_MODEL == "twoheads" else S3_EVAL_MODELS_REG

packaging_processor = SKLearnProcessor(
    role=ROLE,
    instance_type=PACKAGING_INSTANCE_TYPE,
    instance_count=1,
    framework_version="1.2-1",
    sagemaker_session=SESSION,
    base_job_name="hymmrec-model-packaging",
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "model-packaging"},
    ],
)

packaging_processor.run(
    code=f"{LOCAL_EVAL_DIR}model-packaging-job.py",
    arguments=[],
    inputs=[
        ProcessingInput(
            source=winner_model_s3,
            destination="/opt/ml/processing/input/model",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
        ProcessingInput(
            source=S3_EVAL_WINNER,
            destination="/opt/ml/processing/input/winner",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/full-model",
            destination=f"{S3_PACKAGED_MODELS}full-model/",
            output_name="full-model",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/user-tower",
            destination=f"{S3_PACKAGED_MODELS}user-tower/",
            output_name="user-tower",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/item-tower",
            destination=f"{S3_PACKAGED_MODELS}item-tower/",
            output_name="item-tower",
        ),
    ],
    logs=True,
    wait=True,
)

print("✅ Model Packaging completado.")
print(f"   Full Model: {S3_PACKAGED_MODELS}full-model/full_model.tar.gz")
print(f"   User Tower: {S3_PACKAGED_MODELS}user-tower/user_tower.tar.gz")
print(f"   Item Tower: {S3_PACKAGED_MODELS}item-tower/item_tower.tar.gz")


# ==============================================================================
# CELDA 9: STEP 4 — DEPLOY ENDPOINTS
# ==============================================================================
# Despliega los artefactos como endpoints de SageMaker:
#   - Full Model: predicción completa (user, item) → rating + interaction + atención
#   - User Tower: user_id → embedding 64D (para búsqueda ANN en OpenSearch)
#   - Item Tower: NO se despliega como endpoint (se usa offline con Batch Transform)

print("\n" + "=" * 60)
print("STEP 4: DEPLOY ENDPOINTS")
print("=" * 60)

# --- 4A: Full Model Endpoint ---
print("\n[4A] Desplegando Full Model Endpoint...")

full_model_data = f"{S3_PACKAGED_MODELS}full-model/full_model.tar.gz"

full_model = PyTorchModel(
    model_data=full_model_data,
    role=ROLE,
    entry_point="inference.py",
    framework_version="2.1",
    py_version="py310",
    sagemaker_session=SESSION,
    name=f"hymmrec-full-model-{int(time.time())}",
)

full_model_predictor = full_model.deploy(
    initial_instance_count=1,
    instance_type=ENDPOINT_INSTANCE_TYPE,
    endpoint_name="hymmrec-full-model-endpoint",
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "component", "Value": "full-model"},
    ],
    wait=True,
)

print(f"✅ Full Model Endpoint desplegado: hymmrec-full-model-endpoint")

# --- 4B: User Tower Endpoint ---
print("\n[4B] Desplegando User Tower Endpoint...")

user_tower_data = f"{S3_PACKAGED_MODELS}user-tower/user_tower.tar.gz"

user_tower_model = PyTorchModel(
    model_data=user_tower_data,
    role=ROLE,
    entry_point="inference.py",
    framework_version="2.1",
    py_version="py310",
    sagemaker_session=SESSION,
    name=f"hymmrec-user-tower-{int(time.time())}",
)

user_tower_predictor = user_tower_model.deploy(
    initial_instance_count=1,
    instance_type=ENDPOINT_INSTANCE_TYPE,
    endpoint_name="hymmrec-user-tower-endpoint",
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "component", "Value": "user-tower"},
    ],
    wait=True,
)

print(f"✅ User Tower Endpoint desplegado: hymmrec-user-tower-endpoint")


# ==============================================================================
# CELDA 10: VERIFICACIÓN DE ENDPOINTS (Smoke Test)
# ==============================================================================
print("\n" + "=" * 60)
print("VERIFICACIÓN: Smoke Test de Endpoints")
print("=" * 60)

import json
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Test Full Model
print("\n[Test] Full Model Endpoint...")
full_model_predictor.serializer = JSONSerializer()
full_model_predictor.deserializer = JSONDeserializer()

test_input_full = {
    "user_idx": 0,
    "item_idx": 0,
    "genres_multihot": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "text_emb": [0.0] * 1024,  # Placeholder
    "img_emb": [0.0] * 1024,   # Placeholder
}

try:
    response_full = full_model_predictor.predict(test_input_full)
    print(f"  ✅ Respuesta: {json.dumps(response_full, indent=2)}")
except Exception as e:
    print(f"  ⚠️ Error: {e}")

# Test User Tower
print("\n[Test] User Tower Endpoint...")
user_tower_predictor.serializer = JSONSerializer()
user_tower_predictor.deserializer = JSONDeserializer()

test_input_user = {"user_ids": [0, 1, 2]}

try:
    response_user = user_tower_predictor.predict(test_input_user)
    n_embeddings = len(response_user.get("user_embeddings", []))
    dim = len(response_user["user_embeddings"][0]) if n_embeddings > 0 else 0
    print(f"  ✅ {n_embeddings} embeddings generados, dimensión: {dim}")
except Exception as e:
    print(f"  ⚠️ Error: {e}")


# ==============================================================================
# CELDA 11: RESUMEN FINAL
# ==============================================================================
print("\n" + "=" * 60)
print("PIPELINE POST-TRAINING COMPLETADO")
print("=" * 60)

print(f"""
📊 Evaluación:
   - Modelo ganador: {WINNER_MODEL}
   - NDCG@10: {winner_metrics.get('ndcg_at_k', 'N/A')}
   - HR@10: {winner_metrics.get('hit_rate_at_k', 'N/A')}
   - RMSE Stars: {winner_metrics.get('rmse_stars', 'N/A')}

📦 Model Registry:
   - Package Group: {MODEL_PACKAGE_GROUP_NAME}
   - ARN: {model_package_arn}

🚀 Endpoints desplegados:
   - Full Model: hymmrec-full-model-endpoint ({ENDPOINT_INSTANCE_TYPE})
   - User Tower: hymmrec-user-tower-endpoint ({ENDPOINT_INSTANCE_TYPE})

📁 Artefactos empaquetados:
   - {S3_PACKAGED_MODELS}full-model/full_model.tar.gz
   - {S3_PACKAGED_MODELS}user-tower/user_tower.tar.gz
   - {S3_PACKAGED_MODELS}item-tower/item_tower.tar.gz

🔮 Próximos pasos:
   1. Item Tower: Batch Transform para generar embeddings de todos los ítems
   2. Indexar item embeddings en OpenSearch (ANN/kNN)
   3. Flujo de inferencia: User Tower → OpenSearch ANN → Full Model re-ranking
""")


# ==============================================================================
# CELDA 12 (OPCIONAL): CLEANUP — Eliminar Endpoints
# ==============================================================================
# Descomenta para eliminar endpoints y dejar de incurrir en costos.
# Útil si solo necesitabas los endpoints para la demo del TFM.

# print("Eliminando endpoints...")
# full_model_predictor.delete_endpoint(delete_endpoint_config=True)
# user_tower_predictor.delete_endpoint(delete_endpoint_config=True)
# print("✅ Endpoints eliminados.")
