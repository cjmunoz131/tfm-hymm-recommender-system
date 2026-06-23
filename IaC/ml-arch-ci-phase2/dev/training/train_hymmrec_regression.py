"""
HYMM-REC: SageMaker Training Job — Regresión (Rating Prediction)
=================================================================
Script de entrenamiento compatible con SageMaker Training Jobs.
Predice rating escalado [0, 1] usando MSELoss.

Métricas reportadas a CloudWatch:
  - train_mse: MSE promedio de entrenamiento por época
  - val_mse: MSE promedio de validación por época
  - val_rmse: RMSE en validación (escala 0-1)
  - val_rmse_stars: RMSE en escala real de estrellas (1-5)
  - test_mse: MSE final en test set
  - test_rmse: RMSE final en test set (escala 0-1)
  - test_rmse_stars: RMSE final en escala de estrellas

SageMaker Environment Variables:
  - SM_CHANNEL_TRAIN: Directorio con train/, val/, test/ parquets
  - SM_CHANNEL_EMBEDDINGS: Directorio con embeddings_catalog.pkl
  - SM_MODEL_DIR: Directorio donde guardar el modelo entrenado
  - SM_OUTPUT_DATA_DIR: Directorio para artefactos adicionales (métricas JSON)

Hyperparámetros (vía argparse / SageMaker Estimator):
  - epochs, batch_size, lr, weight_decay, emb_dim, dropout, patience
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

# Agregar el directorio actual al path para importar módulos compartidos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloaders import load_datasets_and_create_loaders
from nn_hymmrec import MultimodalExplainableGMF

# ============================================================
# LOGGING
# ============================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# ============================================================
# TRAINING LOOP
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Entrena una época completa. Retorna MSE promedio."""
    model.train()
    running_loss = 0.0
    running_samples = 0

    for step, batch in enumerate(train_loader):
        user = batch["user"].to(device)
        item = batch["item"].to(device)
        cat = batch["genres"].to(device)
        text_emb = batch["text_emb"].to(device)
        img_emb = batch["img_emb"].to(device)
        rating_real = batch["rating"].to(device)

        optimizer.zero_grad()
        rating_pred, _ = model(user, item, cat, text_emb, img_emb)
        rating_pred = rating_pred.squeeze()

        loss = criterion(rating_pred, rating_real)
        loss.backward()
        optimizer.step()

        batch_size = user.size(0)
        running_loss += loss.item() * batch_size
        running_samples += batch_size

        if step % 200 == 0 and step > 0:
            logger.info(
                f"  [Train] Step {step} | "
                f"Samples: {running_samples:,}/{len(train_loader.dataset):,} | "
                f"Batch MSE: {loss.item():.6f}"
            )

    epoch_mse = running_loss / running_samples
    return epoch_mse


def validate(model, val_loader, criterion, device):
    """Evalúa en validación. Retorna MSE promedio."""
    model.eval()
    running_loss = 0.0
    running_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            user = batch["user"].to(device)
            item = batch["item"].to(device)
            cat = batch["genres"].to(device)
            text_emb = batch["text_emb"].to(device)
            img_emb = batch["img_emb"].to(device)
            rating_real = batch["rating"].to(device)

            rating_pred, _ = model(user, item, cat, text_emb, img_emb)
            rating_pred = rating_pred.squeeze()

            loss = criterion(rating_pred, rating_real)
            batch_size = user.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size

    epoch_mse = running_loss / running_samples
    return epoch_mse


def evaluate_test(model, test_loader, criterion, device):
    """Evaluación final en test set con métricas completas y explicabilidad."""
    model.eval()
    running_loss = 0.0
    running_samples = 0
    total_attn_weights = torch.zeros(3).to(device)

    with torch.no_grad():
        for batch in test_loader:
            user = batch["user"].to(device)
            item = batch["item"].to(device)
            cat = batch["genres"].to(device)
            text_emb = batch["text_emb"].to(device)
            img_emb = batch["img_emb"].to(device)
            rating_real = batch["rating"].to(device)

            rating_pred, attn_weights = model(user, item, cat, text_emb, img_emb)
            rating_pred = rating_pred.squeeze()

            loss = criterion(rating_pred, rating_real)
            batch_size = user.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size
            total_attn_weights += attn_weights.sum(dim=0)

    test_mse = running_loss / running_samples
    test_rmse = test_mse**0.5
    test_rmse_stars = test_rmse * 4.0  # Escala [0,1] → rango de 4 estrellas (1-5)

    # Explicabilidad: pesos promedio de atención
    avg_attn = (total_attn_weights / running_samples) * 100.0

    metrics = {
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_rmse_stars": test_rmse_stars,
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

    # 1. Cargar datos
    logger.info("=" * 60)
    logger.info("PASO 1: Cargando datasets y embeddings...")
    logger.info("=" * 60)

    train_loader, val_loader, test_loader, metadata = load_datasets_and_create_loaders(
        data_dir=args.data_dir,
        embeddings_dir=args.embeddings_dir,
        encoders_dir=args.encoders_dir,
        mode="regression",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_users = metadata["num_users"]
    num_items = metadata["num_items"]
    num_categories = metadata["num_categories"]

    # 2. Construir modelo
    logger.info("=" * 60)
    logger.info("PASO 2: Construyendo modelo MultimodalExplainableGMF...")
    logger.info("=" * 60)

    model = MultimodalExplainableGMF(
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

    # 3. Loss, Optimizer y Scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ReduceLROnPlateau: reduce LR cuando val_mse deja de mejorar
    # Complementa al early stopping — le da "segunda oportunidad" con LR más bajo
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
    logger.info("PASO 3: Entrenamiento con Early Stopping + LR Scheduler...")
    logger.info("=" * 60)

    best_val_mse = float("inf")
    patience_counter = 0
    history = {"train_mse": [], "val_mse": [], "val_rmse": [], "val_rmse_stars": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e} ---")

        # Train
        train_mse = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_mse = validate(model, val_loader, criterion, device)
        val_rmse = val_mse**0.5
        val_rmse_stars = val_rmse * 4.0

        # Step del scheduler (monitorea val_mse)
        scheduler.step(val_mse)

        # Registrar historia
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_rmse"].append(val_rmse)
        history["val_rmse_stars"].append(val_rmse_stars)
        history["lr"].append(current_lr)

        # Métricas para CloudWatch (SageMaker las parsea del stdout)
        logger.info(f"train_mse={train_mse:.6f};")
        logger.info(f"val_mse={val_mse:.6f};")
        logger.info(f"val_rmse={val_rmse:.6f};")
        logger.info(f"val_rmse_stars={val_rmse_stars:.4f};")
        logger.info(f"lr={current_lr:.2e};")

        # Detectar si el scheduler redujo el LR
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            logger.info(f"  Scheduler: LR reducido {current_lr:.2e} → {new_lr:.2e}")

        logger.info(
            f"Resumen Epoch {epoch} | "
            f"Train MSE: {train_mse:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val RMSE Stars: {val_rmse_stars:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Early Stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            # Guardar mejor modelo
            os.makedirs(args.model_dir, exist_ok=True)
            model_path = os.path.join(args.model_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"  Mejor modelo guardado: Val MSE={best_val_mse:.6f}")
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

    # Cargar mejor modelo
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pth")))
    test_metrics = evaluate_test(model, test_loader, criterion, device)

    # Reportar métricas de test para CloudWatch
    logger.info(f"test_mse={test_metrics['test_mse']:.6f};")
    logger.info(f"test_rmse={test_metrics['test_rmse']:.6f};")
    logger.info(f"test_rmse_stars={test_metrics['test_rmse_stars']:.4f};")

    logger.info(f"\nMÉTRICAS FINALES DE TEST:")
    logger.info(f"   - MSE (Escala 0-1): {test_metrics['test_mse']:.6f}")
    logger.info(f"   - RMSE (Escala 0-1): {test_metrics['test_rmse']:.6f}")
    logger.info(f"   - RMSE (Estrellas): {test_metrics['test_rmse_stars']:.4f}")
    logger.info(f"\nEXPLICABILIDAD GLOBAL (Atención):")
    logger.info(f"   - Categoría: {test_metrics['attn_category_pct']:.2f}%")
    logger.info(f"   - Texto (Nova): {test_metrics['attn_text_pct']:.2f}%")
    logger.info(f"   - Imagen (Nova): {test_metrics['attn_image_pct']:.2f}%")

    # 6. Guardar artefactos adicionales
    elapsed = time.time() - start_time
    logger.info(f"\nTiempo total de entrenamiento: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    output_dir = args.output_data_dir
    os.makedirs(output_dir, exist_ok=True)

    # Guardar métricas como JSON
    all_metrics = {
        "mode": "regression",
        "hyperparameters": vars(args),
        "best_val_mse": best_val_mse,
        "test_metrics": test_metrics,
        "training_history": history,
        "training_time_seconds": elapsed,
        "total_epochs_trained": len(history["train_mse"]),
    }
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Métricas guardadas: {metrics_path}")

    # Guardar metadata del modelo para inferencia
    model_metadata = {
        "num_users": num_users,
        "num_items": num_items,
        "num_categories": num_categories,
        "emb_dim": args.emb_dim,
        "aws_dim": 1024,
        "dropout": args.dropout,
        "mode": "regression",
    }
    metadata_path = os.path.join(args.model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=2)
    logger.info(f"Metadata del modelo guardada: {metadata_path}")

    logger.info("\nEntrenamiento completado exitosamente.")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HYMM-REC Training Job (Regression)")

    # Hyperparámetros
    parser.add_argument("--epochs", type=int, default=20, help="Número máximo de épocas")
    parser.add_argument("--batch_size", type=int, default=256, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate inicial")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2)")
    parser.add_argument("--emb_dim", type=int, default=64, help="Dimensión de embeddings internos")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=5, help="Épocas de paciencia para early stopping")
    parser.add_argument("--num_workers", type=int, default=2, help="Workers para DataLoader")

    # LR Scheduler (ReduceLROnPlateau)
    parser.add_argument("--scheduler_patience", type=int, default=2, help="Épocas sin mejora antes de reducir LR")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="Factor de reducción del LR (LR *= factor)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Piso mínimo del learning rate")

    # SageMaker environment
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--embeddings-dir", type=str, default=os.environ.get("SM_CHANNEL_EMBEDDINGS", "/opt/ml/input/data/embeddings"))
    parser.add_argument("--encoders-dir", type=str, default=os.environ.get("SM_CHANNEL_ENCODERS", "/opt/ml/input/data/encoders"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    args = parser.parse_args()

    # Normalizar nombres de args con guiones a underscores
    args.model_dir = getattr(args, "model_dir", None) or args.__dict__.get("model-dir", "/opt/ml/model")
    args.data_dir = getattr(args, "data_dir", None) or args.__dict__.get("data-dir", "/opt/ml/input/data/train")
    args.embeddings_dir = getattr(args, "embeddings_dir", None) or args.__dict__.get("embeddings-dir", "/opt/ml/input/data/embeddings")
    args.encoders_dir = getattr(args, "encoders_dir", None) or args.__dict__.get("encoders-dir", "/opt/ml/input/data/encoders")
    args.output_data_dir = getattr(args, "output_data_dir", None) or args.__dict__.get("output-data-dir", "/opt/ml/output/data")

    main(args)
