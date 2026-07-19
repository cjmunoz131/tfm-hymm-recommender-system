"""
HYMM-REC: SageMaker Training Job — Multi-Task Two-Heads (Retrieval + Calidad)
===============================================================================
Script de entrenamiento para el modelo de dos cabezas que optimiza dos
objetivos simultáneamente:

  Cabeza 1 (Retrieval/Ranking):
    - BCE Loss sobre TODOS los datos (positivos + negativos muestreados)
    - Aprende: "¿El usuario interactuará con esta película?"

  Cabeza 2 (Calidad/Rating):
    - MSE Loss ENMASCARADA (solo sobre interacciones positivas)
    - Aprende: "¿Cuánto le gustará al usuario esta película?"

  Loss Total = BCE + MSE(solo positivos)

Ventajas del enfoque Multi-Task:
  - El backbone compartido (towers) aprende representaciones más ricas
  - La señal BCE ayuda a discriminar relevancia (retrieval stage)
  - La señal MSE refina la calidad percibida (ranking stage)
  - Una sola pasada de inferencia produce ambas predicciones

Métricas reportadas a CloudWatch:
  - train_total_loss, train_bce, train_mse
  - val_total_loss, val_bce, val_mse
  - val_rmse_stars (RMSE solo sobre positivos, en escala de estrellas)
  - test_total_loss, test_bce, test_mse, test_rmse_stars
  - test_accuracy, test_precision, test_recall, test_f1

SageMaker Environment Variables:
  - SM_CHANNEL_TRAIN: Directorio con train/, val/, test/ parquets
  - SM_CHANNEL_EMBEDDINGS: Directorio con embeddings_catalog.pkl
  - SM_MODEL_DIR: Directorio donde guardar el modelo entrenado
  - SM_OUTPUT_DATA_DIR: Directorio para artefactos adicionales

Hyperparámetros:
  - epochs, batch_size, lr, weight_decay, emb_dim, dropout, patience, neg_ratio
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloaders import load_datasets_and_create_loaders
from nn_hymmrec import MultimodalExplainableGMF_TwoHeads

# ============================================================
# LOGGING
# ============================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# ============================================================
# TRAINING LOOP
# ============================================================
def train_one_epoch(model, train_loader, criterion_bce, criterion_mse, optimizer, device):
    """
    Entrena una época con pérdida multi-task.
    Loss = BCE(todos) + MSE(solo positivos)
    """
    model.train()
    running_total = 0.0
    running_bce = 0.0
    running_mse = 0.0
    running_samples = 0

    for step, batch in enumerate(train_loader):
        user = batch["user"].to(device)
        item = batch["item"].to(device)
        cat = batch["genres"].to(device)
        text_emb = batch["text_emb"].to(device)
        img_emb = batch["img_emb"].to(device)
        target_interaction = batch["interaction"].to(device)
        target_rating = batch["rating"].to(device)

        optimizer.zero_grad()

        # Forward: dos predicciones
        pred_interaction, pred_rating, _ = model(user, item, cat, text_emb, img_emb)
        pred_interaction = pred_interaction.squeeze()
        pred_rating = pred_rating.squeeze()

        # Cabeza 1: BCE sobre TODOS los datos (positivos + negativos)
        loss_bce = criterion_bce(pred_interaction, target_interaction)

        # Cabeza 2: MSE ENMASCARADO (solo sobre interacciones positivas)
        mask_positivos = (target_interaction == 1.0)
        if mask_positivos.sum() > 0:
            loss_mse = criterion_mse(pred_rating[mask_positivos], target_rating[mask_positivos])
        else:
            loss_mse = torch.tensor(0.0, device=device)

        # Loss total multi-task
        loss = loss_bce + loss_mse

        loss.backward()
        optimizer.step()

        batch_size = user.size(0)
        running_total += loss.item() * batch_size
        running_bce += loss_bce.item() * batch_size
        running_mse += loss_mse.item() * batch_size
        running_samples += batch_size

        if step % 200 == 0 and step > 0:
            logger.info(
                f"  [Train] Step {step} | "
                f"Samples: {running_samples:,}/{len(train_loader.dataset):,} | "
                f"Total: {loss.item():.4f} (BCE: {loss_bce.item():.4f} | MSE: {loss_mse.item():.4f})"
            )

    epoch_total = running_total / running_samples
    epoch_bce = running_bce / running_samples
    epoch_mse = running_mse / running_samples

    return epoch_total, epoch_bce, epoch_mse


def validate(model, val_loader, criterion_bce, criterion_mse, device):
    """Evalúa en validación con pérdida multi-task."""
    model.eval()
    running_total = 0.0
    running_bce = 0.0
    running_mse = 0.0
    running_samples = 0
    running_mse_samples = 0  # Solo positivos para RMSE stars

    with torch.no_grad():
        for batch in val_loader:
            user = batch["user"].to(device)
            item = batch["item"].to(device)
            cat = batch["genres"].to(device)
            text_emb = batch["text_emb"].to(device)
            img_emb = batch["img_emb"].to(device)
            target_interaction = batch["interaction"].to(device)
            target_rating = batch["rating"].to(device)

            pred_interaction, pred_rating, _ = model(user, item, cat, text_emb, img_emb)
            pred_interaction = pred_interaction.squeeze()
            pred_rating = pred_rating.squeeze()

            loss_bce = criterion_bce(pred_interaction, target_interaction)

            mask_positivos = (target_interaction == 1.0)
            if mask_positivos.sum() > 0:
                loss_mse = criterion_mse(pred_rating[mask_positivos], target_rating[mask_positivos])
                running_mse_samples += mask_positivos.sum().item()
            else:
                loss_mse = torch.tensor(0.0, device=device)

            loss = loss_bce + loss_mse

            batch_size = user.size(0)
            running_total += loss.item() * batch_size
            running_bce += loss_bce.item() * batch_size
            running_mse += loss_mse.item() * batch_size
            running_samples += batch_size

    epoch_total = running_total / running_samples
    epoch_bce = running_bce / running_samples
    epoch_mse = running_mse / running_samples

    # RMSE en estrellas (solo sobre positivos de validación)
    epoch_rmse_stars = (epoch_mse ** 0.5) * 4.0 if epoch_mse > 0 else 0.0

    return epoch_total, epoch_bce, epoch_mse, epoch_rmse_stars


def evaluate_test(model, test_loader, criterion_bce, criterion_mse, device, threshold=0.5):
    """
    Evaluación final en test set con métricas completas:
      - Multi-task losses (total, BCE, MSE)
      - Métricas de clasificación (accuracy, precision, recall, F1)
      - RMSE en estrellas (solo sobre positivos)
      - Explicabilidad (pesos de atención promedio)
    """
    model.eval()
    running_total = 0.0
    running_bce = 0.0
    running_mse = 0.0
    running_samples = 0
    total_attn_weights = torch.zeros(3).to(device)

    all_preds_interaction = []
    all_targets_interaction = []
    all_preds_rating = []
    all_targets_rating = []
    all_masks = []

    with torch.no_grad():
        for batch in test_loader:
            user = batch["user"].to(device)
            item = batch["item"].to(device)
            cat = batch["genres"].to(device)
            text_emb = batch["text_emb"].to(device)
            img_emb = batch["img_emb"].to(device)
            target_interaction = batch["interaction"].to(device)
            target_rating = batch["rating"].to(device)

            pred_interaction, pred_rating, attn_weights = model(user, item, cat, text_emb, img_emb)
            pred_interaction = pred_interaction.squeeze()
            pred_rating = pred_rating.squeeze()

            loss_bce = criterion_bce(pred_interaction, target_interaction)

            mask_positivos = (target_interaction == 1.0)
            if mask_positivos.sum() > 0:
                loss_mse = criterion_mse(pred_rating[mask_positivos], target_rating[mask_positivos])
            else:
                loss_mse = torch.tensor(0.0, device=device)

            loss = loss_bce + loss_mse

            batch_size = user.size(0)
            running_total += loss.item() * batch_size
            running_bce += loss_bce.item() * batch_size
            running_mse += loss_mse.item() * batch_size
            running_samples += batch_size
            total_attn_weights += attn_weights.sum(dim=0)

            all_preds_interaction.append(pred_interaction.cpu())
            all_targets_interaction.append(target_interaction.cpu())
            all_preds_rating.append(pred_rating.cpu())
            all_targets_rating.append(target_rating.cpu())
            all_masks.append(mask_positivos.cpu())

    # Métricas de loss
    test_total = running_total / running_samples
    test_bce = running_bce / running_samples
    test_mse = running_mse / running_samples
    test_rmse_stars = (test_mse ** 0.5) * 4.0 if test_mse > 0 else 0.0

    # Métricas de clasificación por UMBRAL DE CALIDAD (≥3.5★)
    # Alineado con evaluation-job.py: evalúa la cabeza de rating como clasificador
    # de calidad. NO usa la cabeza de interacción (que no tiene negativos en test).
    all_preds_rat = torch.cat(all_preds_rating)
    all_targets_rat = torch.cat(all_targets_rating)

    # Convertir a escala de estrellas (1-5)
    pred_stars = all_preds_rat * 4.0 + 1.0
    true_stars = all_targets_rat * 4.0 + 1.0

    # Binarizar por umbral de calidad
    quality_threshold = 3.5
    predicted_quality = (pred_stars >= quality_threshold).float()
    true_quality = (true_stars >= quality_threshold).float()

    tp = ((predicted_quality == 1) & (true_quality == 1)).sum().item()
    fp = ((predicted_quality == 1) & (true_quality == 0)).sum().item()
    fn = ((predicted_quality == 0) & (true_quality == 1)).sum().item()
    tn = ((predicted_quality == 0) & (true_quality == 0)).sum().item()

    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Explicabilidad
    avg_attn = (total_attn_weights / running_samples) * 100.0

    metrics = {
        "test_total_loss": test_total,
        "test_bce": test_bce,
        "test_mse": test_mse,
        "test_rmse_stars": test_rmse_stars,
        "quality_threshold_stars": quality_threshold,
        "test_accuracy": accuracy,
        "test_precision_quality": precision,
        "test_recall_quality": recall,
        "test_f1_quality": f1,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "attn_category_pct": avg_attn[0].item(),
        "attn_text_pct": avg_attn[1].item(),
        "attn_image_pct": avg_attn[2].item(),
    }

    return metrics


# ============================================================
# MAIN
# ============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    logger.info(f"Hyperparámetros: {vars(args)}")

    start_time = time.time()

    # 1. Cargar datos con muestreo negativo
    logger.info("=" * 60)
    logger.info("PASO 1: Cargando datasets + muestreo negativo in-memory...")
    logger.info("=" * 60)

    train_loader, val_loader, test_loader, metadata = load_datasets_and_create_loaders(
        data_dir=args.data_dir,
        embeddings_dir=args.embeddings_dir,
        encoders_dir=args.encoders_dir,
        mode="multitask",
        batch_size=args.batch_size,
        neg_ratio=args.neg_ratio,
        num_workers=args.num_workers,
    )

    num_users = metadata["num_users"]
    num_items = metadata["num_items"]
    num_categories = metadata["num_categories"]

    # 2. Construir modelo Two-Heads
    logger.info("=" * 60)
    logger.info("PASO 2: Construyendo modelo MultimodalExplainableGMF_TwoHeads...")
    logger.info("=" * 60)

    model = MultimodalExplainableGMF_TwoHeads(
        num_users=num_users,
        num_items=num_items,
        num_categories=num_categories,
        emb_dim=args.emb_dim,
        aws_dim=1024,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parámetros totales: {total_params:,} | Entrenables: {trainable_params:,}")

    # 3. Dos funciones de pérdida + Optimizer + Scheduler
    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ReduceLROnPlateau: reduce LR cuando val_total_loss deja de mejorar
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
        verbose=False,
    )
    logger.info(
        f"  Scheduler: ReduceLROnPlateau (factor={args.scheduler_factor}, "
        f"patience={args.scheduler_patience}, min_lr={args.min_lr})"
    )

    # 4. Training Loop con Early Stopping + LR Scheduling
    logger.info("=" * 60)
    logger.info("PASO 3: Entrenamiento Multi-Task con Early Stopping + LR Scheduler...")
    logger.info("=" * 60)

    best_val_total = float("inf")
    patience_counter = 0
    history = {
        "train_total": [], "train_bce": [], "train_mse": [],
        "val_total": [], "val_bce": [], "val_mse": [], "val_rmse_stars": [], "lr": [],
    }

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e} ---")

        # Train
        train_total, train_bce, train_mse = train_one_epoch(
            model, train_loader, criterion_bce, criterion_mse, optimizer, device
        )

        # Validate
        val_total, val_bce, val_mse, val_rmse_stars = validate(
            model, val_loader, criterion_bce, criterion_mse, device
        )

        # Registrar historia
        history["train_total"].append(train_total)
        history["train_bce"].append(train_bce)
        history["train_mse"].append(train_mse)
        history["val_total"].append(val_total)
        history["val_bce"].append(val_bce)
        history["val_mse"].append(val_mse)
        history["val_rmse_stars"].append(val_rmse_stars)
        history["lr"].append(current_lr)

        # Step del scheduler (monitorea val_total_loss)
        scheduler.step(val_total)

        # Métricas para CloudWatch (formato parseable)
        logger.info(f"train_total_loss={train_total:.6f};")
        logger.info(f"train_bce={train_bce:.6f};")
        logger.info(f"train_mse={train_mse:.6f};")
        logger.info(f"val_total_loss={val_total:.6f};")
        logger.info(f"val_bce={val_bce:.6f};")
        logger.info(f"val_mse={val_mse:.6f};")
        logger.info(f"val_rmse_stars={val_rmse_stars:.4f};")
        logger.info(f"lr={current_lr:.2e};")

        # Detectar si el scheduler redujo el LR
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            logger.info(f"  Scheduler: LR reducido {current_lr:.2e} → {new_lr:.2e}")

        logger.info(
            f"Resumen Epoch {epoch} | "
            f"Train: {train_total:.4f} (BCE:{train_bce:.4f} MSE:{train_mse:.4f}) | "
            f"Val: {val_total:.4f} (BCE:{val_bce:.4f} MSE:{val_mse:.4f}) | "
            f"Val RMSE Stars: {val_rmse_stars:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Early Stopping (sobre loss total de validación)
        if val_total < best_val_total:
            best_val_total = val_total
            patience_counter = 0
            os.makedirs(args.model_dir, exist_ok=True)
            model_path = os.path.join(args.model_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"  Mejor modelo guardado: Val Total={best_val_total:.6f}")
        else:
            patience_counter += 1
            logger.info(f"  Sin mejora ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            logger.info(f"Early Stopping: sin mejora en {args.patience} épocas.")
            break

    # 5. Evaluación final en Test
    logger.info("=" * 60)
    logger.info("PASO 4: Evaluación final en Test Set...")
    logger.info("=" * 60)

    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pth")))
    test_metrics = evaluate_test(model, test_loader, criterion_bce, criterion_mse, device)

    # Reportar métricas para CloudWatch
    logger.info(f"test_total_loss={test_metrics['test_total_loss']:.6f};")
    logger.info(f"test_bce={test_metrics['test_bce']:.6f};")
    logger.info(f"test_mse={test_metrics['test_mse']:.6f};")
    logger.info(f"test_rmse_stars={test_metrics['test_rmse_stars']:.4f};")
    logger.info(f"test_accuracy={test_metrics['test_accuracy']:.4f};")
    logger.info(f"test_precision_quality={test_metrics['test_precision_quality']:.4f};")
    logger.info(f"test_recall_quality={test_metrics['test_recall_quality']:.4f};")
    logger.info(f"test_f1_quality={test_metrics['test_f1_quality']:.4f};")

    logger.info(f"\nMÉTRICAS FINALES DE TEST (Multi-Task Two-Heads):")
    logger.info(f"   - Total Loss: {test_metrics['test_total_loss']:.6f}")
    logger.info(f"   - BCE Loss (Ranking): {test_metrics['test_bce']:.6f}")
    logger.info(f"   - MSE Loss (Calidad): {test_metrics['test_mse']:.6f}")
    logger.info(f"   - RMSE (Estrellas): {test_metrics['test_rmse_stars']:.4f}")
    logger.info(f"   - Quality Threshold: {test_metrics['quality_threshold_stars']}★")
    logger.info(f"   - Accuracy (Quality): {test_metrics['test_accuracy']:.4f}")
    logger.info(f"   - Precision (Quality): {test_metrics['test_precision_quality']:.4f}")
    logger.info(f"   - Recall (Quality): {test_metrics['test_recall_quality']:.4f}")
    logger.info(f"   - F1 (Quality): {test_metrics['test_f1_quality']:.4f}")
    logger.info(f"   - Confusion Matrix: {test_metrics['confusion_matrix']}")
    logger.info(f"\nEXPLICABILIDAD GLOBAL (Atención):")
    logger.info(f"   - Categoría: {test_metrics['attn_category_pct']:.2f}%")
    logger.info(f"   - Texto (Nova): {test_metrics['attn_text_pct']:.2f}%")
    logger.info(f"   - Imagen (Nova): {test_metrics['attn_image_pct']:.2f}%")

    # 6. Guardar artefactos
    elapsed = time.time() - start_time
    logger.info(f"\nTiempo total: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    output_dir = args.output_data_dir
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = {
        "mode": "multitask_twoheads",
        "hyperparameters": vars(args),
        "best_val_total_loss": best_val_total,
        "test_metrics": test_metrics,
        "training_history": history,
        "training_time_seconds": elapsed,
        "total_epochs_trained": len(history["train_total"]),
    }
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Métricas guardadas: {metrics_path}")

    # También guardar en model_dir para que persista en model.tar.gz
    # (SM_OUTPUT_DATA_DIR no se empaqueta cuando se ejecuta como TrainingStep en Pipeline)
    metrics_path_model = os.path.join(args.model_dir, "training_metrics.json")
    with open(metrics_path_model, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Métricas guardadas (model_dir): {metrics_path_model}")

    model_metadata = {
        "num_users": num_users,
        "num_items": num_items,
        "num_categories": num_categories,
        "emb_dim": args.emb_dim,
        "aws_dim": 1024,
        "dropout": args.dropout,
        "mode": "multitask_twoheads",
        "neg_ratio": args.neg_ratio,
    }
    metadata_path = os.path.join(args.model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=2)
    logger.info(f"Metadata guardada: {metadata_path}")

    logger.info("\nEntrenamiento Multi-Task completado exitosamente.")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HYMM-REC Training Job (Multi-Task Two-Heads)")

    # Hyperparámetros
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--neg_ratio", type=int, default=4, help="Ratio de negativos por positivo para muestreo")
    parser.add_argument("--num_workers", type=int, default=2)

    # LR Scheduler (ReduceLROnPlateau)
    parser.add_argument("--scheduler_patience", type=int, default=2, help="Épocas sin mejora antes de reducir LR")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="Factor de reducción del LR")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Piso mínimo del learning rate")

    # SageMaker environment
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--embeddings-dir", type=str, default=os.environ.get("SM_CHANNEL_EMBEDDINGS", "/opt/ml/input/data/embeddings"))
    parser.add_argument("--encoders-dir", type=str, default=os.environ.get("SM_CHANNEL_ENCODERS", "/opt/ml/input/data/encoders"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    args = parser.parse_args()
    args.model_dir = getattr(args, "model_dir", None) or args.__dict__.get("model-dir", "/opt/ml/model")
    args.data_dir = getattr(args, "data_dir", None) or args.__dict__.get("data-dir", "/opt/ml/input/data/train")
    args.embeddings_dir = getattr(args, "embeddings_dir", None) or args.__dict__.get("embeddings-dir", "/opt/ml/input/data/embeddings")
    args.encoders_dir = getattr(args, "encoders_dir", None) or args.__dict__.get("encoders-dir", "/opt/ml/input/data/encoders")
    args.output_data_dir = getattr(args, "output_data_dir", None) or args.__dict__.get("output-data-dir", "/opt/ml/output/data")

    main(args)
