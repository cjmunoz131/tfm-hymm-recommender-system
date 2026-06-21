"""
HYMM-REC: SageMaker Hyperparameter Tuning Job — Multi-Task Two-Heads
=====================================================================
Script optimizado para HPO del modelo de dos cabezas (Retrieval + Calidad).

Métrica Objetivo (para SageMaker HyperparameterTuner):
  - Nombre: "val_total_loss"
  - Regex: "val_total_loss=(.*?);"
  - Tipo: Minimize

La métrica total combina ambos objetivos: BCE(ranking) + MSE(calidad).
Esto permite que el tuner optimice el equilibrio entre ambas tareas.

Hyperparámetros tunables:
  - lr: [0.0001, 0.01] (Continuous)
  - batch_size: [128, 256, 512] (Categorical)
  - emb_dim: [32, 64, 128] (Categorical)
  - dropout: [0.1, 0.5] (Continuous)
  - weight_decay: [1e-6, 1e-3] (Continuous)
  - num_negatives: [10, 20, 50] (Categorical)
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
from nn_hymmrec import MultimodalExplainableGMF_TwoHeads

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train_epoch(model, train_loader, criterion_bce, criterion_mse, optimizer, device):
    """Una época de entrenamiento multi-task."""
    model.train()
    running_total = 0.0
    running_bce = 0.0
    running_mse = 0.0
    running_samples = 0

    for batch in train_loader:
        user = batch["user"].to(device)
        item = batch["item"].to(device)
        cat = batch["genres"].to(device)
        text_emb = batch["text_emb"].to(device)
        img_emb = batch["img_emb"].to(device)
        target_interaction = batch["interaction"].to(device)
        target_rating = batch["rating"].to(device)

        optimizer.zero_grad()

        pred_interaction, pred_rating, _ = model(user, item, cat, text_emb, img_emb)
        pred_interaction = pred_interaction.squeeze()
        pred_rating = pred_rating.squeeze()

        # BCE sobre todos
        loss_bce = criterion_bce(pred_interaction, target_interaction)

        # MSE enmascarado (solo positivos)
        mask_positivos = (target_interaction == 1.0)
        if mask_positivos.sum() > 0:
            loss_mse = criterion_mse(pred_rating[mask_positivos], target_rating[mask_positivos])
        else:
            loss_mse = torch.tensor(0.0, device=device)

        loss = loss_bce + loss_mse

        loss.backward()
        optimizer.step()

        running_total += loss.item() * user.size(0)
        running_bce += loss_bce.item() * user.size(0)
        running_mse += loss_mse.item() * user.size(0)
        running_samples += user.size(0)

    return (
        running_total / running_samples,
        running_bce / running_samples,
        running_mse / running_samples,
    )


def validate_epoch(model, val_loader, criterion_bce, criterion_mse, device):
    """Validación multi-task."""
    model.eval()
    running_total = 0.0
    running_bce = 0.0
    running_mse = 0.0
    running_samples = 0

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
            else:
                loss_mse = torch.tensor(0.0, device=device)

            loss = loss_bce + loss_mse

            running_total += loss.item() * user.size(0)
            running_bce += loss_bce.item() * user.size(0)
            running_mse += loss_mse.item() * user.size(0)
            running_samples += user.size(0)

    return (
        running_total / running_samples,
        running_bce / running_samples,
        running_mse / running_samples,
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"HPO Two-Heads | Device: {device}")
    logger.info(f"Hyperparámetros: lr={args.lr}, batch={args.batch_size}, "
                f"emb_dim={args.emb_dim}, dropout={args.dropout}, "
                f"wd={args.weight_decay}, negatives={args.num_negatives}")

    start_time = time.time()

    # 1. Cargar datos con muestreo negativo
    train_loader, val_loader, test_loader, metadata = load_datasets_and_create_loaders(
        data_dir=args.data_dir,
        embeddings_dir=args.embeddings_dir,
        mode="multitask",
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        num_workers=args.num_workers,
    )

    # 2. Construir modelo
    model = MultimodalExplainableGMF_TwoHeads(
        num_users=metadata["num_users"],
        num_items=metadata["num_items"],
        num_categories=metadata["num_categories"],
        emb_dim=args.emb_dim,
        aws_dim=1024,
        dropout=args.dropout,
    ).to(device)

    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler para HPO (patience más agresivo)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6, verbose=False
    )

    # 3. Training con Early Stopping agresivo (para HPO)
    best_val_total = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_total, train_bce, train_mse = train_epoch(
            model, train_loader, criterion_bce, criterion_mse, optimizer, device
        )
        val_total, val_bce, val_mse = validate_epoch(
            model, val_loader, criterion_bce, criterion_mse, device
        )

        val_rmse_stars = (val_mse ** 0.5) * 4.0 if val_mse > 0 else 0.0

        # Step scheduler
        scheduler.step(val_total)
        current_lr = optimizer.param_groups[0]["lr"]

        # Formato de reporte para SageMaker HPO (regex parsing)
        logger.info(
            f"Epoch {epoch} | "
            f"train_total_loss={train_total:.6f}; "
            f"val_total_loss={val_total:.6f}; "
            f"val_bce={val_bce:.6f}; "
            f"val_mse={val_mse:.6f}; "
            f"val_rmse_stars={val_rmse_stars:.4f}; "
            f"lr={current_lr:.2e};"
        )

        if val_total < best_val_total:
            best_val_total = val_total
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
    test_total, test_bce, test_mse = validate_epoch(
        model, test_loader, criterion_bce, criterion_mse, device
    )
    test_rmse_stars = (test_mse ** 0.5) * 4.0 if test_mse > 0 else 0.0

    logger.info(f"test_total_loss={test_total:.6f}; test_bce={test_bce:.6f}; test_mse={test_mse:.6f}; test_rmse_stars={test_rmse_stars:.4f};")
    logger.info(f"Training time: {time.time() - start_time:.1f}s")
    logger.info(f"Best val_total_loss={best_val_total:.6f};")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HYMM-REC HPO Job (Multi-Task Two-Heads)")

    # Hyperparámetros tunables
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num_negatives", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)

    # SageMaker environment
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--embeddings-dir", type=str, default=os.environ.get("SM_CHANNEL_EMBEDDINGS", "/opt/ml/input/data/embeddings"))

    args = parser.parse_args()
    args.model_dir = getattr(args, "model_dir", None) or args.__dict__.get("model-dir", "/opt/ml/model")
    args.data_dir = getattr(args, "data_dir", None) or args.__dict__.get("data-dir", "/opt/ml/input/data/train")
    args.embeddings_dir = getattr(args, "embeddings_dir", None) or args.__dict__.get("embeddings-dir", "/opt/ml/input/data/embeddings")

    main(args)
