"""
HYMM-REC: SageMaker Hyperparameter Tuning Job — Regresión
==========================================================
Script optimizado para HPO (Hyperparameter Optimization) en SageMaker.
Se diferencia del training script completo en:
  - Reporta la métrica objetivo de forma que SageMaker HPO pueda parsearla
  - Usa menos épocas (early stopping agresivo) para explorar más combinaciones
  - No guarda artefactos extras (solo modelo y métrica)

Métrica Objetivo (para SageMaker HyperparameterTuner):
  - Nombre: "val_rmse_stars"
  - Regex: "val_rmse_stars=(.*?);"
  - Tipo: Minimize

Hyperparámetros tunables:
  - lr: [0.0001, 0.01] (Continuous)
  - batch_size: [128, 256, 512] (Categorical)
  - emb_dim: [32, 64, 128] (Categorical)
  - dropout: [0.1, 0.5] (Continuous)
  - weight_decay: [1e-6, 1e-3] (Continuous)
"""

import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloaders import load_datasets_and_create_loaders
from nn_hymmrec import MultimodalExplainableGMF

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Una época de entrenamiento."""
    model.train()
    running_loss = 0.0
    running_samples = 0

    for batch in train_loader:
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

        running_loss += loss.item() * user.size(0)
        running_samples += user.size(0)

    return running_loss / running_samples


def validate_epoch(model, val_loader, criterion, device):
    """Validación de una época."""
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
            running_loss += loss.item() * user.size(0)
            running_samples += user.size(0)

    return running_loss / running_samples


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"HPO Regression | Device: {device}")
    logger.info(f"Hyperparámetros: lr={args.lr}, batch={args.batch_size}, "
                f"emb_dim={args.emb_dim}, dropout={args.dropout}, wd={args.weight_decay}")

    start_time = time.time()

    # 1. Cargar datos
    train_loader, val_loader, test_loader, metadata = load_datasets_and_create_loaders(
        data_dir=args.data_dir,
        embeddings_dir=args.embeddings_dir,
        encoders_dir=args.encoders_dir,
        mode="regression",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2. Construir modelo
    model = MultimodalExplainableGMF(
        num_users=metadata["num_users"],
        num_items=metadata["num_items"],
        num_categories=metadata["num_categories"],
        emb_dim=args.emb_dim,
        aws_dim=1024,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler para HPO (patience más agresivo)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6, verbose=False
    )

    # 3. Training con Early Stopping (agresivo para HPO)
    best_val_mse = float("inf")
    patience_counter = 0
    best_rmse_stars = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_mse = train_epoch(model, train_loader, criterion, optimizer, device)
        val_mse = validate_epoch(model, val_loader, criterion, device)

        val_rmse = val_mse**0.5
        val_rmse_stars = val_rmse * 4.0

        # Step scheduler
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]["lr"]

        # Formato de reporte para SageMaker HPO (regex parsing)
        logger.info(f"Epoch {epoch} | train_mse={train_mse:.6f}; val_mse={val_mse:.6f}; val_rmse_stars={val_rmse_stars:.4f}; lr={current_lr:.2e};")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_rmse_stars = val_rmse_stars
            patience_counter = 0
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # 4. Evaluación final en test
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pth")))
    test_mse = validate_epoch(model, test_loader, criterion, device)
    test_rmse_stars = (test_mse**0.5) * 4.0

    logger.info(f"test_mse={test_mse:.6f}; test_rmse_stars={test_rmse_stars:.4f};")
    logger.info(f"Training time: {time.time() - start_time:.1f}s")
    logger.info(f"Best val_rmse_stars={best_rmse_stars:.4f};")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HYMM-REC HPO Job (Regression)")

    # Hyperparámetros tunables
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)

    # SageMaker environment
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--embeddings-dir", type=str, default=os.environ.get("SM_CHANNEL_EMBEDDINGS", "/opt/ml/input/data/embeddings"))
    parser.add_argument("--encoders-dir", type=str, default=os.environ.get("SM_CHANNEL_ENCODERS", "/opt/ml/input/data/encoders"))

    args = parser.parse_args()
    args.model_dir = getattr(args, "model_dir", None) or args.__dict__.get("model-dir", "/opt/ml/model")
    args.data_dir = getattr(args, "data_dir", None) or args.__dict__.get("data-dir", "/opt/ml/input/data/train")
    args.embeddings_dir = getattr(args, "embeddings_dir", None) or args.__dict__.get("embeddings-dir", "/opt/ml/input/data/embeddings")
    args.encoders_dir = getattr(args, "encoders_dir", None) or args.__dict__.get("encoders-dir", "/opt/ml/input/data/encoders")

    main(args)
