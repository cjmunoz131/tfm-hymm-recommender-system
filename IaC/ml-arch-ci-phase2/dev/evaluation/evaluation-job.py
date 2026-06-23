"""
HYMM-REC: SageMaker Processing Job — Evaluación Completa de Modelos
====================================================================
Evalúa ambos modelos (Regresión + Two-Heads) sobre test + cold-start sets
con métricas completas de ranking, clasificación y explicabilidad.

Métricas calculadas:
  - Regresión: MSE, RMSE, RMSE Stars, NDCG@K, HR@K, Precision@K, Recall@K
  - Two-Heads: BCE, Accuracy, Precision, Recall, F1, NDCG@K, HR@K,
               Confusion Matrix, Hybrid Score ranking
  - Ambos: Explicabilidad promedio (atención por modalidad)

Inputs:
  - /opt/ml/processing/input/models/regression/     → model.pth (regresión)
  - /opt/ml/processing/input/models/twoheads/       → model.pth (two-heads)
  - /opt/ml/processing/input/datasets/              → train/, val/, test/, cold-starts/
  - /opt/ml/processing/input/embeddings/            → embeddings_catalog.pkl
  - /opt/ml/processing/input/encoders/              → encoders.pkl

Outputs:
  - /opt/ml/processing/output/reports/              → evaluation_report.json
  - /opt/ml/processing/output/reports/              → model_comparison.json
  - /opt/ml/processing/output/winner/               → best_model_metadata.json

Processor: SKLearnProcessor o PyTorchProcessor (ml.g4dn.xlarge para GPU eval)
"""

import argparse
import json
import logging
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# IMPORTAR ARQUITECTURA (copiada al source_dir del Processing Job)
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# ARQUITECTURA (inline para independencia del Processing Job)
# ============================================================
import torch.nn.functional as F


class ModalityAttention(nn.Module):
    def __init__(self, cat_dim=64, aws_dim=1024):
        super().__init__()
        self.cat_scorer = nn.Linear(cat_dim, 1)
        self.text_scorer = nn.Linear(aws_dim, 1)
        self.img_scorer = nn.Linear(aws_dim, 1)

    def forward(self, cat_vec, text_vec, img_vec):
        scores = torch.cat([
            self.cat_scorer(cat_vec),
            self.text_scorer(text_vec),
            self.img_scorer(img_vec),
        ], dim=1)
        weights = F.softmax(scores, dim=1)
        w_cat, w_text, w_img = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        return cat_vec * w_cat, text_vec * w_text, img_vec * w_img, weights


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
        content_input_dim = emb_dim + aws_dim + aws_dim
        self.content_mlp = nn.Sequential(
            nn.Linear(content_input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout * 0.67),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout * 0.67),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(emb_dim + 256, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Dropout(dropout * 0.67)
        )

    def forward(self, item_id, cat_id, text_emb, img_emb):
        emb_i = self.item_mlp(self.item_embedding(item_id))
        emb_c = self.cat_mlp(cat_id)
        emb_c_w, text_w, img_w, attn = self.attention_layer(emb_c, text_emb, img_emb)
        content = self.content_mlp(torch.cat([emb_c_w, text_w, img_w], dim=1))
        return self.final_mlp(torch.cat([emb_i, content], dim=1)), attn


class MultimodalExplainableGMF(nn.Module):
    """Single-head regression model."""
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
    """Two-heads multi-task model."""
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
# CARGA DE DATOS
# ============================================================
def load_parquet(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith(".parquet")]
        path = os.path.join(path, files[0]) if files else path
    df = pd.read_parquet(path)
    logger.info(f"  Cargado: {path} → {len(df):,} filas")
    return df


def load_pkl(path: str):
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith(".pkl")]
        path = os.path.join(path, files[0]) if files else path
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_metadata(model_dir: str) -> dict:
    meta_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


# ============================================================
# MÉTRICAS DE RANKING: NDCG@K y HIT RATE@K (True Negatives)
# ============================================================
def evaluate_ranking_true_negatives(
    model, df_test, df_all, dict_embeddings, device,
    mode="regression", k=10, num_decoys=99,
    threshold_stars=3.5, bad_threshold_stars=2.5,
):
    """
    Evaluación rigurosa de ranking con verdaderos negativos.
    - Selecciona señuelos de películas explícitamente mal calificadas (<=bad_threshold)
    - Para MTL usa hybrid_score = prob_interaction * pred_rating
    - Para regresión usa directamente pred_rating
    """
    logger.info(f"Evaluación ranking (mode={mode}, k={k}, decoys={num_decoys})...")
    model.eval()

    item_pool = set(df_all["movieId"].unique())
    global_history = df_all.groupby("userId")["movieId"].apply(set).to_dict()
    df_items = df_all[["movieId", "movieId_idx", "genres_multihot"]].drop_duplicates(
        subset=["movieId"]
    ).set_index("movieId")

    # Positivos en test (rating >= threshold)
    threshold_scaled = (threshold_stars - 1.0) / 4.0
    bad_threshold_scaled = (bad_threshold_stars - 1.0) / 4.0
    df_test_pos = df_test[df_test["rating_scaled"] >= threshold_scaled]

    # Pool de películas malas
    df_bad = df_all[df_all["rating_scaled"] <= bad_threshold_scaled]
    bad_pool = set(df_bad["movieId"].unique())
    logger.info(f"  Pool señuelos malos: {len(bad_pool):,} películas")

    hits, ndcg_sum, total_evals = 0, 0.0, 0
    rng = np.random.default_rng(42)

    with torch.no_grad():
        for user_id in df_test_pos["userId"].unique():
            user_idx = df_all[df_all["userId"] == user_id]["userId_idx"].iloc[0]
            user_movies = df_test_pos[df_test_pos["userId"] == user_id]["movieId"].values
            historial = global_history.get(user_id, set())
            posibles_senuelos = list(bad_pool - historial)

            for target_movie in user_movies:
                if target_movie not in df_items.index:
                    continue
                if target_movie not in dict_embeddings:
                    continue

                # Seleccionar señuelos
                if len(posibles_senuelos) < num_decoys:
                    extra = list(item_pool - bad_pool - historial)
                    n_extra = min(num_decoys - len(posibles_senuelos), len(extra))
                    decoys = posibles_senuelos + list(rng.choice(extra, size=n_extra, replace=False))
                else:
                    decoys = list(rng.choice(posibles_senuelos, size=num_decoys, replace=False))

                # Filtrar decoys que no tengan embeddings
                decoys = [d for d in decoys if d in dict_embeddings and d in df_items.index]
                if len(decoys) < 10:
                    continue

                eval_items = [target_movie] + decoys
                batch_users = torch.tensor([user_idx] * len(eval_items), dtype=torch.long).to(device)
                batch_items = torch.tensor(
                    df_items.loc[eval_items]["movieId_idx"].values, dtype=torch.long
                ).to(device)
                batch_genres = torch.tensor(
                    np.vstack(df_items.loc[eval_items]["genres_multihot"].values), dtype=torch.float32
                ).to(device)
                batch_text = torch.tensor(
                    np.vstack([dict_embeddings[i]["text_emb"] for i in eval_items]), dtype=torch.float32
                ).to(device)
                batch_img = torch.tensor(
                    np.vstack([dict_embeddings[i]["img_emb"] for i in eval_items]), dtype=torch.float32
                ).to(device)

                if mode == "twoheads":
                    prob_int, pred_rat, _ = model(batch_users, batch_items, batch_genres, batch_text, batch_img)
                    scores = (prob_int.view(-1) * pred_rat.view(-1)).cpu().numpy()
                else:
                    pred_rat, _ = model(batch_users, batch_items, batch_genres, batch_text, batch_img)
                    scores = pred_rat.view(-1).cpu().numpy()

                rankings = (-scores).argsort().argsort()
                target_rank = rankings[0]

                if target_rank < k:
                    hits += 1
                    ndcg_sum += 1.0 / math.log2(target_rank + 2)
                total_evals += 1

    hr = hits / total_evals if total_evals > 0 else 0
    ndcg = ndcg_sum / total_evals if total_evals > 0 else 0
    logger.info(f"  HR@{k}: {hr:.4f} | NDCG@{k}: {ndcg:.4f} (sobre {total_evals:,} evaluaciones)")
    return {"hit_rate_at_k": hr, "ndcg_at_k": ndcg, "k": k, "total_evaluations": total_evals}


# ============================================================
# MÉTRICAS DE CLASIFICACIÓN Y REGRESIÓN (sobre DataLoader)
# ============================================================
def evaluate_pointwise_metrics(model, df_eval, dict_embeddings, device, mode="regression", batch_size=512):
    """
    Calcula métricas pointwise sobre un DataFrame (test o cold-start):
      - Regresión: MSE, RMSE, RMSE Stars
      - Two-Heads: BCE, Accuracy, Precision, Recall, F1, Confusion Matrix
      - Ambos: Explicabilidad promedio
    """
    model.eval()
    from torch.utils.data import DataLoader, Dataset

    class EvalDataset(Dataset):
        def __init__(self, df, emb_dict):
            self.users = torch.tensor(df["userId_idx"].values, dtype=torch.long)
            self.items = torch.tensor(df["movieId_idx"].values, dtype=torch.long)
            self.genres = torch.tensor(np.vstack(df["genres_multihot"].values), dtype=torch.float32)
            self.ratings = torch.tensor(df["rating_scaled"].values, dtype=torch.float32)
            self.item_ids = df["movieId"].values
            self.emb_dict = emb_dict

        def __len__(self):
            return len(self.users)

        def __getitem__(self, idx):
            mid = self.item_ids[idx]
            entry = self.emb_dict.get(mid)
            if entry:
                t = torch.tensor(entry["text_emb"], dtype=torch.float32)
                i = torch.tensor(entry["img_emb"], dtype=torch.float32)
            else:
                t = torch.zeros(1024, dtype=torch.float32)
                i = torch.zeros(1024, dtype=torch.float32)
            return self.users[idx], self.items[idx], self.genres[idx], t, i, self.ratings[idx]

    dataset = EvalDataset(df_eval, dict_embeddings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds_rating = []
    all_preds_interaction = []
    all_true_ratings = []
    total_attn = torch.zeros(3).to(device)
    n_samples = 0

    with torch.no_grad():
        for users, items, genres, text, img, ratings in loader:
            users, items, genres = users.to(device), items.to(device), genres.to(device)
            text, img, ratings = text.to(device), img.to(device), ratings.to(device)

            if mode == "twoheads":
                prob_int, pred_rat, attn = model(users, items, genres, text, img)
                all_preds_interaction.extend(prob_int.view(-1).cpu().numpy())
                all_preds_rating.extend(pred_rat.view(-1).cpu().numpy())
            else:
                pred_rat, attn = model(users, items, genres, text, img)
                all_preds_rating.extend(pred_rat.view(-1).cpu().numpy())

            all_true_ratings.extend(ratings.cpu().numpy())
            total_attn += attn.sum(dim=0)
            n_samples += users.size(0)

    # Explicabilidad
    avg_attn = (total_attn / n_samples) * 100.0
    explainability = {
        "category_pct": avg_attn[0].item(),
        "text_pct": avg_attn[1].item(),
        "image_pct": avg_attn[2].item(),
    }

    true_ratings = np.array(all_true_ratings)
    pred_ratings = np.array(all_preds_rating)

    # Métricas de regresión (siempre)
    mse = float(np.mean((pred_ratings - true_ratings) ** 2))
    rmse = mse ** 0.5
    rmse_stars = rmse * 4.0

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "rmse_stars": rmse_stars,
        "explainability": explainability,
        "n_samples": n_samples,
    }

    # Métricas de clasificación (solo two-heads)
    if mode == "twoheads" and all_preds_interaction:
        pred_int = np.array(all_preds_interaction)
        true_labels = (true_ratings > 0.0).astype(float)
        pred_labels = (pred_int >= 0.5).astype(float)

        tp = float(((pred_labels == 1) & (true_labels == 1)).sum())
        fp = float(((pred_labels == 1) & (true_labels == 0)).sum())
        fn = float(((pred_labels == 0) & (true_labels == 1)).sum())
        tn = float(((pred_labels == 0) & (true_labels == 0)).sum())

        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        bce = float(-np.mean(true_labels * np.log(pred_int + 1e-8) + (1 - true_labels) * np.log(1 - pred_int + 1e-8)))

        metrics.update({
            "bce": bce,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        })

    return metrics


# ============================================================
# PRECISION@K y RECALL@K
# ============================================================
def evaluate_precision_recall_at_k(model, df_eval, dict_embeddings, device, mode="regression", k=10, threshold_stars=3.5):
    """Precision@K y Recall@K agrupados por usuario."""
    model.eval()
    threshold_scaled = (threshold_stars - 1.0) / 4.0
    user_ratings = defaultdict(list)

    unique_users = df_eval["userId_idx"].unique()
    df_items_lookup = df_eval[["movieId", "movieId_idx", "genres_multihot"]].drop_duplicates(subset=["movieId"])

    with torch.no_grad():
        for _, row in df_eval.iterrows():
            mid = row["movieId"]
            if mid not in dict_embeddings:
                continue

            u = torch.tensor([row["userId_idx"]], dtype=torch.long).to(device)
            i = torch.tensor([row["movieId_idx"]], dtype=torch.long).to(device)
            g = torch.tensor(row["genres_multihot"]).unsqueeze(0).float().to(device)
            t = torch.tensor(dict_embeddings[mid]["text_emb"]).unsqueeze(0).float().to(device)
            img = torch.tensor(dict_embeddings[mid]["img_emb"]).unsqueeze(0).float().to(device)

            if mode == "twoheads":
                _, pred_rat, _ = model(u, i, g, t, img)
            else:
                pred_rat, _ = model(u, i, g, t, img)

            pred_stars = pred_rat.item() * 4.0 + 1.0
            true_stars = row["rating_scaled"] * 4.0 + 1.0
            user_ratings[row["userId_idx"]].append((pred_stars, true_stars))

    precisions, recalls = [], []
    for uid, ratings in user_ratings.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(1 for _, t in ratings if t >= threshold_stars)
        if n_rel == 0:
            continue
        n_rec_k = sum(1 for p, _ in ratings[:k] if p >= threshold_stars)
        n_rel_rec_k = sum(1 for p, t in ratings[:k] if p >= threshold_stars and t >= threshold_stars)
        precisions.append(n_rel_rec_k / n_rec_k if n_rec_k > 0 else 0)
        recalls.append(n_rel_rec_k / n_rel)

    avg_prec = float(np.mean(precisions)) if precisions else 0.0
    avg_recall = float(np.mean(recalls)) if recalls else 0.0

    logger.info(f"  Precision@{k}: {avg_prec:.4f} | Recall@{k}: {avg_recall:.4f}")
    return {"precision_at_k": avg_prec, "recall_at_k": avg_recall, "k": k}


# ============================================================
# EVALUACIÓN COMPLETA DE UN MODELO
# ============================================================
def evaluate_model_complete(model, mode, df_test, df_coldstart, df_all, dict_embeddings, device, k=10, metadata=None):
    """Ejecuta todas las evaluaciones para un modelo."""
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUACIÓN COMPLETA: {mode.upper()}")
    logger.info(f"{'='*60}")

    if metadata is None:
        metadata = {
            "num_users": int(df_all["userId_idx"].max()) + 1,
            "num_items": int(df_all["movieId_idx"].max()) + 1,
        }

    results = {"mode": mode}

    # 1. Métricas pointwise sobre Test
    logger.info("\n[1/5] Métricas pointwise (Test)...")
    results["test_pointwise"] = evaluate_pointwise_metrics(model, df_test, dict_embeddings, device, mode)

    # 2. Métricas pointwise sobre Cold-Start
    if df_coldstart is not None and len(df_coldstart) > 0:
        logger.info("\n[2/5] Métricas pointwise (Cold-Start)...")
        results["coldstart_pointwise"] = evaluate_pointwise_metrics(model, df_coldstart, dict_embeddings, device, mode)
    else:
        logger.info("\n[2/5] Cold-Start: sin datos disponibles")
        results["coldstart_pointwise"] = None

    # 3. Ranking con verdaderos negativos (HR@K, NDCG@K)
    logger.info("\n[3/5] Ranking con verdaderos negativos...")
    results["ranking_true_neg"] = evaluate_ranking_true_negatives(
        model, df_test, df_all, dict_embeddings, device, mode=mode, k=k
    )

    # 4. Precision@K y Recall@K
    logger.info("\n[4/5] Precision@K y Recall@K...")
    results["precision_recall"] = evaluate_precision_recall_at_k(
        model, df_test, dict_embeddings, device, mode=mode, k=k
    )

    # 5. Resumen
    logger.info("\n[5/5] Generando resumen...")
    results["summary"] = {
        "rmse_stars": results["test_pointwise"]["rmse_stars"],
        "hit_rate_at_k": results["ranking_true_neg"]["hit_rate_at_k"],
        "ndcg_at_k": results["ranking_true_neg"]["ndcg_at_k"],
        "precision_at_k": results["precision_recall"]["precision_at_k"],
        "recall_at_k": results["precision_recall"]["recall_at_k"],
        "explainability": results["test_pointwise"]["explainability"],
    }
    if "bce" in results["test_pointwise"]:
        results["summary"]["bce"] = results["test_pointwise"]["bce"]
        results["summary"]["f1"] = results["test_pointwise"]["f1"]
        results["summary"]["accuracy"] = results["test_pointwise"]["accuracy"]

    return results


# ============================================================
# COMPARACIÓN Y SELECCIÓN DEL MEJOR MODELO
# ============================================================
def compare_and_select_winner(results_regression, results_twoheads) -> dict:
    """
    Compara ambos modelos y selecciona el ganador basado en un score compuesto.

    Criterios de selección (ponderados):
      - NDCG@K (40%): Calidad del ranking — lo más importante para un recsys
      - HR@K (25%): Capacidad de encontrar ítems relevantes
      - RMSE Stars (20%): Precisión en predicción de rating (invertido: menor es mejor)
      - F1/Precision@K (15%): Balance precisión/recall

    El modelo con mejor score compuesto gana.
    """
    logger.info("\nComparando modelos...")

    def compute_score(results):
        s = results["summary"]
        # RMSE invertido (menor es mejor → 1 - normalizado)
        rmse_norm = max(0, 1.0 - s["rmse_stars"] / 4.0)  # 0 estrellas error = 1.0, 4 = 0.0
        ndcg = s["ndcg_at_k"]
        hr = s["hit_rate_at_k"]
        prec = s["precision_at_k"]

        score = 0.40 * ndcg + 0.25 * hr + 0.20 * rmse_norm + 0.15 * prec
        return score

    score_reg = compute_score(results_regression)
    score_th = compute_score(results_twoheads)

    logger.info(f"  Score Regresión: {score_reg:.4f}")
    logger.info(f"  Score Two-Heads: {score_th:.4f}")

    winner = "twoheads" if score_th >= score_reg else "regression"
    winner_results = results_twoheads if winner == "twoheads" else results_regression

    comparison = {
        "regression_score": score_reg,
        "twoheads_score": score_th,
        "winner": winner,
        "winner_summary": winner_results["summary"],
        "scoring_weights": {"ndcg": 0.40, "hr": 0.25, "rmse_inv": 0.20, "precision": 0.15},
    }

    logger.info(f"  GANADOR: {winner.upper()} (score: {max(score_reg, score_th):.4f})")
    return comparison


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="HYMM-REC Evaluation Processing Job")
    parser.add_argument("--k", type=int, default=10, help="K para métricas Top-K")
    parser.add_argument("--num-decoys", type=int, default=99, help="Señuelos para ranking")
    parser.add_argument("--emb-dim", type=int, default=64, help="Dimensión de embeddings (fallback)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout del modelo (fallback)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    inicio = time.time()

    # Paths
    models_base = "/opt/ml/processing/input/models"
    datasets_base = "/opt/ml/processing/input/datasets"
    embeddings_path = "/opt/ml/processing/input/embeddings"
    encoders_path = "/opt/ml/processing/input/encoders"
    output_path = "/opt/ml/processing/output/reports"

    # 1. Cargar datos
    logger.info("\n[PASO 1] Cargando datos...")
    df_train = load_parquet(os.path.join(datasets_base, "train"))
    df_val = load_parquet(os.path.join(datasets_base, "val"))
    df_test = load_parquet(os.path.join(datasets_base, "test"))

    coldstart_path = os.path.join(datasets_base, "cold-starts")
    df_coldstart = load_parquet(coldstart_path) if os.path.exists(coldstart_path) else None

    df_all = pd.concat([df_train, df_val, df_test])
    dict_embeddings = load_pkl(embeddings_path)

    # Dimensiones desde encoders.pkl (vocabulario completo, incluye cold-start)
    if os.path.exists(encoders_path):
        encoders = load_pkl(encoders_path)
        num_users = len(encoders["le_user"].classes_)
        num_items = len(encoders["le_item"].classes_)
        num_categories = len(encoders["mlb"].classes_)
        logger.info(f"  Dimensiones desde encoders.pkl (vocabulario completo)")
    else:
        num_users = int(df_all["userId_idx"].max()) + 1
        num_items = int(df_all["movieId_idx"].max()) + 1
        num_categories = len(df_all["genres_multihot"].iloc[0])
        logger.info(f"  Dimensiones desde max(dataset) — encoders.pkl no encontrado")

    logger.info(f"  Universo: {num_users:,} users | {num_items:,} items | {num_categories} cats")

    # 2. Cargar modelo de Regresión
    logger.info("\n[PASO 2] Cargando modelo de Regresión...")
    reg_model_path = os.path.join(models_base, "regression", "model.pth")
    reg_meta = load_model_metadata(os.path.join(models_base, "regression"))
    emb_dim = reg_meta.get("emb_dim", args.emb_dim)
    dropout = reg_meta.get("dropout", args.dropout)

    model_reg = MultimodalExplainableGMF(
        num_users, num_items, num_categories, emb_dim=emb_dim, dropout=dropout
    ).to(device)
    model_reg.load_state_dict(torch.load(reg_model_path, map_location=device))
    logger.info("  Modelo regresión cargado.")

    # 3. Cargar modelo Two-Heads
    logger.info("\n[PASO 3] Cargando modelo Two-Heads...")
    th_model_path = os.path.join(models_base, "twoheads", "model.pth")
    th_meta = load_model_metadata(os.path.join(models_base, "twoheads"))
    emb_dim_th = th_meta.get("emb_dim", args.emb_dim)
    dropout_th = th_meta.get("dropout", args.dropout)

    model_th = MultimodalExplainableGMF_TwoHeads(
        num_users, num_items, num_categories, emb_dim=emb_dim_th, dropout=dropout_th
    ).to(device)
    model_th.load_state_dict(torch.load(th_model_path, map_location=device))
    logger.info("  Modelo two-heads cargado.")

    # 4. Evaluar ambos modelos
    reg_metadata = {"num_users": num_users, "num_items": num_items, "num_categories": num_categories}
    th_metadata = {"num_users": num_users, "num_items": num_items, "num_categories": num_categories}

    results_reg = evaluate_model_complete(
        model_reg, "regression", df_test, df_coldstart, df_all, dict_embeddings, device, k=args.k, metadata=reg_metadata
    )
    results_th = evaluate_model_complete(
        model_th, "twoheads", df_test, df_coldstart, df_all, dict_embeddings, device, k=args.k, metadata=th_metadata
    )

    # 5. Comparar y seleccionar ganador
    logger.info("\n[PASO 5] Comparación y selección...")
    comparison = compare_and_select_winner(results_reg, results_th)

    # 6. Guardar reportes
    logger.info("\n[PASO 6] Guardando reportes...")
    os.makedirs(output_path, exist_ok=True)

    # Reporte completo de regresión
    with open(os.path.join(output_path, "evaluation_regression.json"), "w") as f:
        json.dump(results_reg, f, indent=2, default=str)

    # Reporte completo de two-heads
    with open(os.path.join(output_path, "evaluation_twoheads.json"), "w") as f:
        json.dump(results_th, f, indent=2, default=str)

    # Comparación y ganador
    with open(os.path.join(output_path, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # Metadata del ganador (para Model Registry)
    winner_meta = {
        "winner_model": comparison["winner"],
        "winner_score": max(comparison["regression_score"], comparison["twoheads_score"]),
        "winner_metrics": comparison["winner_summary"],
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evaluation_config": {"k": args.k, "num_decoys": args.num_decoys},
    }
    winner_path = "/opt/ml/processing/output/winner"
    os.makedirs(winner_path, exist_ok=True)
    with open(os.path.join(winner_path, "best_model_metadata.json"), "w") as f:
        json.dump(winner_meta, f, indent=2)

    elapsed = time.time() - inicio
    logger.info(f"\nEvaluación completada en {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"GANADOR: {comparison['winner'].upper()}")


if __name__ == "__main__":
    main()
