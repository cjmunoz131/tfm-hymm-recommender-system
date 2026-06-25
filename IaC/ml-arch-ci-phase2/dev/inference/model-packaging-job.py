"""
HYMM-REC: SageMaker Processing Job — Model Packaging
=====================================================
Extrae las sub-redes del modelo ganador y empaqueta artefactos separados
para deploy independiente de cada componente.

Artefactos generados:
  1. full_model.tar.gz  → Modelo completo (UserTower + ItemTower + Heads)
     - Deploy: SageMaker Endpoint (real-time inference)
     - Input: (user_id, item_id, genres, text_emb, img_emb)
     - Output: P(interaction), rating, attention_weights

  2. user_tower.tar.gz  → Solo UserTower
     - Deploy: SageMaker Endpoint (real-time)
     - Input: user_id → Output: user_embedding (64D)
     - Uso: Generar vector de usuario para búsqueda ANN en OpenSearch

  3. item_tower.tar.gz  → Solo ItemTower
     - Deploy: Batch Transform (offline)
     - Input: (item_id, genres, text_emb, img_emb) → Output: item_embedding (64D)
     - Uso: Generar embeddings de ítems para indexar en OpenSearch

Cada artefacto incluye:
  - model.pth (pesos del modelo/sub-red)
  - model_metadata.json (dimensiones, modo, versión)
  - inference.py (script de serving para SageMaker)

Inputs:
  - /opt/ml/processing/input/model/        → model.pth + model_metadata.json (ganador)
  - /opt/ml/processing/input/winner/        → best_model_metadata.json (del eval job)

Outputs:
  - /opt/ml/processing/output/full-model/   → full_model.tar.gz
  - /opt/ml/processing/output/user-tower/   → user_tower.tar.gz
  - /opt/ml/processing/output/item-tower/   → item_tower.tar.gz

Processor: SKLearnProcessor (ml.m5.large — solo CPU, operación ligera)
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# ARQUITECTURA (inline — misma que evaluation-job.py)
# ============================================================
class ModalityAttention(nn.Module):
    def __init__(self, cat_dim=64, aws_dim=1024):
        super().__init__()
        self.cat_scorer = nn.Linear(cat_dim, 1)
        self.text_scorer = nn.Linear(aws_dim, 1)
        self.img_scorer = nn.Linear(aws_dim, 1)

    def forward(self, cat_vec, text_vec, img_vec):
        scores = torch.cat([
            self.cat_scorer(cat_vec), self.text_scorer(text_vec), self.img_scorer(img_vec)
        ], dim=1)
        weights = F.softmax(scores, dim=1)
        return (cat_vec * weights[:, 0:1], text_vec * weights[:, 1:2],
                img_vec * weights[:, 2:3], weights)


class UserTower(nn.Module):
    def __init__(self, num_users, emb_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )

    def forward(self, user_id):
        return self.mlp(self.user_embedding(user_id))


class ExplainableItemTower(nn.Module):
    def __init__(self, num_items, num_categories, emb_dim=64, aws_dim=1024, dropout=0.3):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )
        self.cat_mlp = nn.Sequential(
            nn.Linear(num_categories, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )
        self.attention_layer = ModalityAttention(cat_dim=emb_dim, aws_dim=aws_dim)
        content_dim = emb_dim + aws_dim + aws_dim
        self.content_mlp = nn.Sequential(
            nn.Linear(content_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout * 0.67),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout * 0.67),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(emb_dim + 256, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Dropout(dropout * 0.67),
        )

    def forward(self, item_id, cat_id, text_emb, img_emb):
        emb_i = self.item_mlp(self.item_embedding(item_id))
        emb_c = self.cat_mlp(cat_id)
        c_w, t_w, i_w, attn = self.attention_layer(emb_c, text_emb, img_emb)
        content = self.content_mlp(torch.cat([c_w, t_w, i_w], dim=1))
        return self.final_mlp(torch.cat([emb_i, content], dim=1)), attn


class MultimodalExplainableGMF(nn.Module):
    def __init__(self, num_users, num_items, num_categories, emb_dim=64, aws_dim=1024, dropout=0.3):
        super().__init__()
        self.user_tower = UserTower(num_users, emb_dim)
        self.item_tower = ExplainableItemTower(num_items, num_categories, emb_dim, aws_dim, dropout)
        self.gmf_layer = nn.Linear(emb_dim, 1)

    def forward(self, user_id, item_id, cat_id, text_emb, img_emb):
        u = self.user_tower(user_id)
        i, attn = self.item_tower(item_id, cat_id, text_emb, img_emb)
        return torch.sigmoid(self.gmf_layer(u * i)), attn


class MultimodalExplainableGMF_TwoHeads(nn.Module):
    def __init__(self, num_users, num_items, num_categories, emb_dim=64, aws_dim=1024, dropout=0.3):
        super().__init__()
        self.user_tower = UserTower(num_users, emb_dim)
        self.item_tower = ExplainableItemTower(num_items, num_categories, emb_dim, aws_dim, dropout)
        self.head_interaction = nn.Linear(emb_dim, 1)
        self.head_rating = nn.Linear(emb_dim, 1)

    def forward(self, user_id, item_id, cat_id, text_emb, img_emb):
        u = self.user_tower(user_id)
        i, attn = self.item_tower(item_id, cat_id, text_emb, img_emb)
        v = u * i
        return torch.sigmoid(self.head_interaction(v)), torch.sigmoid(self.head_rating(v)), attn


# ============================================================
# INFERENCE SCRIPTS (se incluyen en cada tar.gz)
# ============================================================
INFERENCE_FULL_MODEL = '''
import json
import os
import pickle
import torch
import numpy as np

def model_fn(model_dir):
    """Carga el modelo completo desde el artefacto."""
    import sys
    sys.path.insert(0, model_dir)

    with open(os.path.join(model_dir, "model_metadata.json")) as f:
        meta = json.load(f)

    # Importar arquitectura
    from model_architecture import get_model_class
    ModelClass = get_model_class(meta["mode"])

    model = ModelClass(
        num_users=meta["num_users"],
        num_items=meta["num_items"],
        num_categories=meta["num_categories"],
        emb_dim=meta["emb_dim"],
        aws_dim=meta.get("aws_dim", 1024),
        dropout=meta.get("dropout", 0.3),
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu"))
    model.eval()
    return {"model": model, "meta": meta}


def input_fn(request_body, content_type="application/json"):
    """Parsea el input JSON."""
    if content_type == "application/json":
        data = json.loads(request_body)
        return data
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_dict):
    """Genera predicción del modelo completo."""
    model = model_dict["model"]
    meta = model_dict["meta"]

    user = torch.tensor([input_data["user_idx"]], dtype=torch.long)
    item = torch.tensor([input_data["item_idx"]], dtype=torch.long)
    genres = torch.tensor([input_data["genres_multihot"]], dtype=torch.float32)
    text_emb = torch.tensor([input_data["text_emb"]], dtype=torch.float32)
    img_emb = torch.tensor([input_data["img_emb"]], dtype=torch.float32)

    with torch.no_grad():
        if meta["mode"] == "multitask_twoheads":
            prob_int, pred_rat, attn = model(user, item, genres, text_emb, img_emb)
            return {
                "prob_interaction": prob_int.item(),
                "pred_rating_scaled": pred_rat.item(),
                "pred_rating_stars": pred_rat.item() * 4.0 + 1.0,
                "hybrid_score": prob_int.item() * pred_rat.item(),
                "attention_weights": {
                    "category": attn[0][0].item(),
                    "text": attn[0][1].item(),
                    "image": attn[0][2].item(),
                },
            }
        else:
            pred, attn = model(user, item, genres, text_emb, img_emb)
            return {
                "pred_rating_scaled": pred.item(),
                "pred_rating_stars": pred.item() * 4.0 + 1.0,
                "attention_weights": {
                    "category": attn[0][0].item(),
                    "text": attn[0][1].item(),
                    "image": attn[0][2].item(),
                },
            }


def output_fn(prediction, accept="application/json"):
    """Serializa la respuesta."""
    return json.dumps(prediction), accept
'''

INFERENCE_USER_TOWER = '''
import json
import os
import torch

def model_fn(model_dir):
    """Carga solo la UserTower."""
    import sys
    sys.path.insert(0, model_dir)
    from model_architecture import UserTower

    with open(os.path.join(model_dir, "model_metadata.json")) as f:
        meta = json.load(f)

    tower = UserTower(num_users=meta["num_users"], emb_dim=meta["emb_dim"])
    tower.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu"))
    tower.eval()
    return {"model": tower, "meta": meta}


def input_fn(request_body, content_type="application/json"):
    return json.loads(request_body)


def predict_fn(input_data, model_dict):
    """Genera embedding de usuario (64D)."""
    tower = model_dict["model"]
    user_ids = input_data.get("user_ids", [input_data.get("user_idx")])
    user_tensor = torch.tensor(user_ids, dtype=torch.long)

    with torch.no_grad():
        embeddings = tower(user_tensor)

    return {"user_embeddings": embeddings.numpy().tolist()}


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), accept
'''

INFERENCE_ITEM_TOWER = '''
import json
import os
import torch
import numpy as np

def model_fn(model_dir):
    """Carga solo la ItemTower."""
    import sys
    sys.path.insert(0, model_dir)
    from model_architecture import ExplainableItemTower

    with open(os.path.join(model_dir, "model_metadata.json")) as f:
        meta = json.load(f)

    tower = ExplainableItemTower(
        num_items=meta["num_items"],
        num_categories=meta["num_categories"],
        emb_dim=meta["emb_dim"],
        aws_dim=meta.get("aws_dim", 1024),
        dropout=meta.get("dropout", 0.3),
    )
    tower.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu"))
    tower.eval()
    return {"model": tower, "meta": meta}


def input_fn(request_body, content_type="application/json"):
    return json.loads(request_body)


def predict_fn(input_data, model_dict):
    """Genera embedding de item (64D) + pesos de atención."""
    tower = model_dict["model"]

    item = torch.tensor([input_data["item_idx"]], dtype=torch.long)
    genres = torch.tensor([input_data["genres_multihot"]], dtype=torch.float32)
    text_emb = torch.tensor([input_data["text_emb"]], dtype=torch.float32)
    img_emb = torch.tensor([input_data["img_emb"]], dtype=torch.float32)

    with torch.no_grad():
        embedding, attn = tower(item, genres, text_emb, img_emb)

    return {
        "item_embedding": embedding[0].numpy().tolist(),
        "attention_weights": {
            "category": attn[0][0].item(),
            "text": attn[0][1].item(),
            "image": attn[0][2].item(),
        },
    }


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), accept
'''


# Módulo de arquitectura que se incluye en cada tar.gz
MODEL_ARCHITECTURE_PY = '''
"""Módulo de arquitectura portable para SageMaker inference."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAttention(nn.Module):
    def __init__(self, cat_dim=64, aws_dim=1024):
        super().__init__()
        self.cat_scorer = nn.Linear(cat_dim, 1)
        self.text_scorer = nn.Linear(aws_dim, 1)
        self.img_scorer = nn.Linear(aws_dim, 1)

    def forward(self, cat_vec, text_vec, img_vec):
        scores = torch.cat([self.cat_scorer(cat_vec), self.text_scorer(text_vec), self.img_scorer(img_vec)], dim=1)
        weights = F.softmax(scores, dim=1)
        return cat_vec * weights[:, 0:1], text_vec * weights[:, 1:2], img_vec * weights[:, 2:3], weights


class UserTower(nn.Module):
    def __init__(self, num_users, emb_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU())

    def forward(self, user_id):
        return self.mlp(self.user_embedding(user_id))


class ExplainableItemTower(nn.Module):
    def __init__(self, num_items, num_categories, emb_dim=64, aws_dim=1024, dropout=0.3):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.item_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU())
        self.cat_mlp = nn.Sequential(nn.Linear(num_categories, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU())
        self.attention_layer = ModalityAttention(cat_dim=emb_dim, aws_dim=aws_dim)
        content_dim = emb_dim + aws_dim + aws_dim
        self.content_mlp = nn.Sequential(
            nn.Linear(content_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout * 0.67),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout * 0.67))
        self.final_mlp = nn.Sequential(
            nn.Linear(emb_dim + 256, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Dropout(dropout * 0.67))

    def forward(self, item_id, cat_id, text_emb, img_emb):
        emb_i = self.item_mlp(self.item_embedding(item_id))
        emb_c = self.cat_mlp(cat_id)
        c_w, t_w, i_w, attn = self.attention_layer(emb_c, text_emb, img_emb)
        content = self.content_mlp(torch.cat([c_w, t_w, i_w], dim=1))
        return self.final_mlp(torch.cat([emb_i, content], dim=1)), attn


class MultimodalExplainableGMF(nn.Module):
    def __init__(self, num_users, num_items, num_categories, emb_dim=64, aws_dim=1024, dropout=0.3):
        super().__init__()
        self.user_tower = UserTower(num_users, emb_dim)
        self.item_tower = ExplainableItemTower(num_items, num_categories, emb_dim, aws_dim, dropout)
        self.gmf_layer = nn.Linear(emb_dim, 1)

    def forward(self, user_id, item_id, cat_id, text_emb, img_emb):
        u = self.user_tower(user_id)
        i, attn = self.item_tower(item_id, cat_id, text_emb, img_emb)
        return torch.sigmoid(self.gmf_layer(u * i)), attn


class MultimodalExplainableGMF_TwoHeads(nn.Module):
    def __init__(self, num_users, num_items, num_categories, emb_dim=64, aws_dim=1024, dropout=0.3):
        super().__init__()
        self.user_tower = UserTower(num_users, emb_dim)
        self.item_tower = ExplainableItemTower(num_items, num_categories, emb_dim, aws_dim, dropout)
        self.head_interaction = nn.Linear(emb_dim, 1)
        self.head_rating = nn.Linear(emb_dim, 1)

    def forward(self, user_id, item_id, cat_id, text_emb, img_emb):
        u = self.user_tower(user_id)
        i, attn = self.item_tower(item_id, cat_id, text_emb, img_emb)
        v = u * i
        return torch.sigmoid(self.head_interaction(v)), torch.sigmoid(self.head_rating(v)), attn


def get_model_class(mode):
    if mode == "multitask_twoheads":
        return MultimodalExplainableGMF_TwoHeads
    return MultimodalExplainableGMF
'''


# ============================================================
# EMPAQUETADO (crear tar.gz con modelo + inference + metadata)
# ============================================================
def write_file(path, content):
    """Escribe contenido de texto a un archivo."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    logger.info(f"  Escrito: {path}")


def create_tar_gz(source_dir, output_path):
    """Crea un tar.gz del directorio fuente."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        for root, dirs, files in os.walk(source_dir):
            for f in files:
                filepath = os.path.join(root, f)
                arcname = os.path.relpath(filepath, source_dir)
                tar.add(filepath, arcname=arcname)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"  Empaquetado: {output_path} ({size_mb:.1f} MB)")


def package_full_model(model, metadata, work_dir, output_dir):
    """Empaqueta el modelo completo."""
    logger.info("Empaquetando Full Model...")
    pkg_dir = os.path.join(work_dir, "full_model")
    os.makedirs(pkg_dir, exist_ok=True)

    # Guardar pesos
    torch.save(model.state_dict(), os.path.join(pkg_dir, "model.pth"))

    # Guardar metadata
    with open(os.path.join(pkg_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Guardar scripts en code/ (estructura requerida por SageMaker inference)
    code_dir = os.path.join(pkg_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    write_file(os.path.join(code_dir, "inference.py"), INFERENCE_FULL_MODEL)
    write_file(os.path.join(code_dir, "model_architecture.py"), MODEL_ARCHITECTURE_PY)

    # Crear tar.gz
    tar_path = os.path.join(output_dir, "full_model.tar.gz")
    create_tar_gz(pkg_dir, tar_path)
    return tar_path


def package_user_tower(model, metadata, work_dir, output_dir):
    """Empaqueta solo la UserTower."""
    logger.info("Empaquetando User Tower...")
    pkg_dir = os.path.join(work_dir, "user_tower")
    os.makedirs(pkg_dir, exist_ok=True)

    # Extraer y guardar solo UserTower
    torch.save(model.user_tower.state_dict(), os.path.join(pkg_dir, "model.pth"))

    # Metadata reducida
    tower_meta = {
        "component": "user_tower",
        "num_users": metadata["num_users"],
        "emb_dim": metadata["emb_dim"],
        "output_dim": metadata["emb_dim"],
        "parent_model": metadata["mode"],
    }
    with open(os.path.join(pkg_dir, "model_metadata.json"), "w") as f:
        json.dump(tower_meta, f, indent=2)

    code_dir = os.path.join(pkg_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    write_file(os.path.join(code_dir, "inference.py"), INFERENCE_USER_TOWER)
    write_file(os.path.join(code_dir, "model_architecture.py"), MODEL_ARCHITECTURE_PY)

    tar_path = os.path.join(output_dir, "user_tower.tar.gz")
    create_tar_gz(pkg_dir, tar_path)
    return tar_path


def package_item_tower(model, metadata, work_dir, output_dir):
    """Empaqueta solo la ItemTower."""
    logger.info("Empaquetando Item Tower...")
    pkg_dir = os.path.join(work_dir, "item_tower")
    os.makedirs(pkg_dir, exist_ok=True)

    # Extraer y guardar solo ItemTower
    torch.save(model.item_tower.state_dict(), os.path.join(pkg_dir, "model.pth"))

    tower_meta = {
        "component": "item_tower",
        "num_items": metadata["num_items"],
        "num_categories": metadata["num_categories"],
        "emb_dim": metadata["emb_dim"],
        "aws_dim": metadata.get("aws_dim", 1024),
        "dropout": metadata.get("dropout", 0.3),
        "output_dim": metadata["emb_dim"],
        "parent_model": metadata["mode"],
    }
    with open(os.path.join(pkg_dir, "model_metadata.json"), "w") as f:
        json.dump(tower_meta, f, indent=2)

    code_dir = os.path.join(pkg_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    write_file(os.path.join(code_dir, "inference.py"), INFERENCE_ITEM_TOWER)
    write_file(os.path.join(code_dir, "model_architecture.py"), MODEL_ARCHITECTURE_PY)

    tar_path = os.path.join(output_dir, "item_tower.tar.gz")
    create_tar_gz(pkg_dir, tar_path)
    return tar_path


# ============================================================
# REGISTRO EN MODEL PACKAGE GROUPS (Versionado para CD)
# ============================================================

PYTORCH_INFERENCE_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker"
SAGEMAKER_ASSETS_BUCKET = "hymmrec-sagemaker-assets"

# Mapeo: componente → package group name
PACKAGE_GROUPS = {
    "full-model": "hymmrec-full-model-sm-pg",
    "user-tower": "hymmrec-user-tower-sm-pg",
    "item-tower": "hymmrec-item-tower-sm-pg",
}

# Mapeo: componente → S3 path del tar.gz (output del processing job)
PACKAGED_PATHS = {
    "full-model": f"s3://{SAGEMAKER_ASSETS_BUCKET}/hymmrec/packaged-models/full-model/full_model.tar.gz",
    "user-tower": f"s3://{SAGEMAKER_ASSETS_BUCKET}/hymmrec/packaged-models/user-tower/user_tower.tar.gz",
    "item-tower": f"s3://{SAGEMAKER_ASSETS_BUCKET}/hymmrec/packaged-models/item-tower/item_tower.tar.gz",
}


def register_packaged_models_in_registry(metadata):
    """
    Registra cada artefacto empaquetado en su Model Package Group correspondiente.
    Cada llamada a create_model_package genera una nueva versión automáticamente
    (/1, /2, /3...) dentro del grupo.

    Esto permite que Terraform en el CD referencie:
      model_package_name = "arn:aws:sagemaker:REGION:ACCOUNT:model-package/hymmrec-full-model-sm-pg/2"
    y obtenga el artefacto correcto con su inference.py incluido.
    """
    import boto3

    sm_client = boto3.client("sagemaker", region_name="us-east-1")

    descriptions = {
        "full-model": (
            f"HYMM-REC Full Model ({metadata.get('mode', 'unknown')}). "
            f"Predicción completa: interaction + rating + atención explicable."
        ),
        "user-tower": (
            f"HYMM-REC User Tower (emb_dim={metadata.get('emb_dim', 64)}). "
            f"Genera embedding de usuario 64D para búsqueda ANN en OpenSearch."
        ),
        "item-tower": (
            f"HYMM-REC Item Tower (emb_dim={metadata.get('emb_dim', 64)}). "
            f"Genera embedding de item 64D para indexación offline."
        ),
    }

    instance_types_realtime = {
        "full-model": ["ml.m5.large", "ml.c5.large", "ml.m5.xlarge"],
        "user-tower": ["ml.m5.large", "ml.c5.large"],
        "item-tower": ["ml.m5.large"],
    }

    instance_types_transform = {
        "full-model": ["ml.m5.large", "ml.m5.xlarge"],
        "user-tower": ["ml.m5.large"],
        "item-tower": ["ml.m5.large", "ml.m5.xlarge"],
    }

    for component, pg_name in PACKAGE_GROUPS.items():
        model_data_url = PACKAGED_PATHS[component]
        logger.info(f"  Registrando {component} en {pg_name}...")
        logger.info(f"    ModelDataUrl: {model_data_url}")

        try:
            response = sm_client.create_model_package(
                ModelPackageGroupName=pg_name,
                ModelPackageDescription=descriptions[component],
                InferenceSpecification={
                    "Containers": [
                        {
                            "Image": PYTORCH_INFERENCE_IMAGE,
                            "ModelDataUrl": model_data_url,
                            "Framework": "PYTORCH",
                        }
                    ],
                    "SupportedContentTypes": ["application/json"],
                    "SupportedResponseMIMETypes": ["application/json"],
                    "SupportedRealtimeInferenceInstanceTypes": instance_types_realtime[component],
                    "SupportedTransformInstanceTypes": instance_types_transform[component],
                },
                ModelApprovalStatus="PendingManualApproval",
                CustomerMetadataProperties={
                    "component": component,
                    "mode": metadata.get("mode", "unknown"),
                    "emb_dim": str(metadata.get("emb_dim", 64)),
                    "num_users": str(metadata.get("num_users", 0)),
                    "num_items": str(metadata.get("num_items", 0)),
                },
            )
            model_pkg_arn = response["ModelPackageArn"]
            logger.info(f"    Registrado: {model_pkg_arn}")
        except Exception as e:
            logger.error(f"    Error registrando {component}: {e}")
            logger.error(f"    Continuando con el siguiente componente...")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="HYMM-REC Model Packaging Job")
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    inicio = time.time()
    logger.info("=" * 60)
    logger.info("Model Packaging Job: Extraer torres y empaquetar artefactos")
    logger.info("=" * 60)

    # Paths
    model_input = "/opt/ml/processing/input/model"
    winner_input = "/opt/ml/processing/input/winner"
    output_full = "/opt/ml/processing/output/full-model"
    output_user = "/opt/ml/processing/output/user-tower"
    output_item = "/opt/ml/processing/output/item-tower"
    work_dir = "/tmp/packaging"

    # 1. Leer metadata del ganador
    logger.info("\n[PASO 1] Leyendo metadata del modelo ganador...")
    meta_path = os.path.join(model_input, "model_metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    mode = metadata.get("mode", "regression")
    logger.info(f"  Modo: {mode}")
    logger.info(f"  Dimensiones: users={metadata['num_users']}, items={metadata['num_items']}, cats={metadata['num_categories']}")

    # 2. Cargar modelo
    logger.info("\n[PASO 2] Cargando modelo...")
    model_path = os.path.join(model_input, "model.pth")

    if mode == "multitask_twoheads":
        model = MultimodalExplainableGMF_TwoHeads(
            num_users=metadata["num_users"],
            num_items=metadata["num_items"],
            num_categories=metadata["num_categories"],
            emb_dim=metadata.get("emb_dim", args.emb_dim),
            aws_dim=metadata.get("aws_dim", 1024),
            dropout=metadata.get("dropout", args.dropout),
        )
    else:
        model = MultimodalExplainableGMF(
            num_users=metadata["num_users"],
            num_items=metadata["num_items"],
            num_categories=metadata["num_categories"],
            emb_dim=metadata.get("emb_dim", args.emb_dim),
            aws_dim=metadata.get("aws_dim", 1024),
            dropout=metadata.get("dropout", args.dropout),
        )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    logger.info("  Modelo cargado correctamente.")

    # 3. Empaquetar artefactos
    logger.info("\n[PASO 3] Empaquetando artefactos...")
    package_full_model(model, metadata, work_dir, output_full)
    package_user_tower(model, metadata, work_dir, output_user)
    package_item_tower(model, metadata, work_dir, output_item)

    # 4. Registrar artefactos en Model Package Groups (versionado)
    logger.info("\n[PASO 4] Registrando artefactos en Model Package Groups...")
    register_packaged_models_in_registry(metadata)

    # 5. Resumen
    elapsed = time.time() - inicio
    logger.info(f"\nPackaging completado en {elapsed:.1f}s")
    logger.info(f"  - Full Model: {output_full}/full_model.tar.gz")
    logger.info(f"  - User Tower: {output_user}/user_tower.tar.gz")
    logger.info(f"  - Item Tower: {output_item}/item_tower.tar.gz")


if __name__ == "__main__":
    main()
