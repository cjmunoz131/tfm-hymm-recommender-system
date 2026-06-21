from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import logging
import sys
import time

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def train(model, train_loader, val_loader, criterion, optimizer, epochs, model_dir, device):
    best_loss = float('inf')
    dataset={'train':train_loader, 'valid':val_loader}
    loss_counter=0

    for epoch in range(1, epochs + 1):
        for phase in ['train', 'valid']:
            logger.info(f"\n--- Epoch {epoch} | Phase: {phase.upper()} ---")
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_samples = 0

            for step, batch in enumerate(dataset[phase]):
                user = batch['user'].to(device)
                item = batch['item'].to(device)
                cat = batch['genres'].to(device)
                text_emb = batch['text_emb'].to(device)
                img_emb = batch['img_emb'].to(device)
                rating_real = batch['rating'].to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # ✨ NUEVO: Desempaquetamos la tupla.
                    # Ignoramos los pesos de atención con "_" porque no los necesitamos para optimizar
                    rating_pred, _ = model(user, item, cat, text_emb, img_emb)
                    rating_pred = rating_pred.squeeze()

                    loss = criterion(rating_pred, rating_real)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                batch_size = user.size(0)
                running_loss += loss.item() * batch_size
                running_samples += batch_size

                if step % 100 == 0 and step > 0:
                    logger.info("{} epoch: {}  [{}/{} ({:.0f}%)] Batch MSE: {:.4f}".format(
                            phase,
                            epoch,
                            running_samples,
                            len(dataset[phase].dataset),
                            100.0 * (running_samples / len(dataset[phase].dataset)),
                            loss.item()
                        )
                    )

            epoch_loss = running_loss / running_samples
            epoch_rmse_estrellas = (epoch_loss ** 0.5) * 4.0

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_rmse_estrellas = epoch_rmse_estrellas
                    # Guardamos el mejor modelo encontrado hasta ahora
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                    logger.info(f"Mejor modelo guardado con Loss: {best_loss:.4f}")
                    logger.info(f"🌟 ¡NUEVO MEJOR MODELO! | MSE: {best_loss:.4f} | ⭐ Error: {best_rmse_estrellas:.4f} estrellas")
                    loss_counter = 0 # Reiniciamos si mejora
                else:
                    loss_counter += 1

            logger.info(f"Resumen {phase.upper()} | MSE (0-1): {epoch_loss:.4f} | ⭐ RMSE (Estrellas): {epoch_rmse_estrellas:.4f}")

        if loss_counter >= 3:
            logger.info(f"🛑 Early Stopping: La validación no ha mejorado en 3 épocas. Fin del entrenamiento.")
            break

    return best_loss


def test(model, test_loader, criterion, device):
    model.eval()

    running_loss = 0
    running_samples = 0

    # ✨ NUEVO: Tensor para acumular la suma de los pesos de atención en todo el test set
    total_attn_weights = torch.zeros(3).to(device)

    with torch.no_grad():
        for batch in test_loader:
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            cat = batch['genres'].to(device)
            text_emb = batch['text_emb'].to(device)
            img_emb = batch['img_emb'].to(device)
            rating_real = batch['rating'].to(device)

            # ✨ NUEVO: Forward Pass desempaquetando predicción Y pesos de atención
            rating_pred, attn_weights = model(user, item, cat, text_emb, img_emb)
            rating_pred = rating_pred.squeeze()

            # Cálculo del Error (Loss)
            loss = criterion(rating_pred, rating_real)
            batch_size = user.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size

            # ✨ NUEVO: Acumulamos los pesos de atención sumándolos a lo largo de este batch
            # attn_weights tiene forma [batch_size, 3]
            total_attn_weights += attn_weights.sum(dim=0)

    # 1. Error Promedio en escala 0-1 (MSE)
    total_loss_escalado = running_loss / running_samples

    # 2. Raíz del error en escala 0-1 (RMSE)
    rmse_escalado = total_loss_escalado ** 0.5

    # 3. Error en el mundo real (Multiplicamos por el rango original 1 a 5)
    rmse_real_estrellas = rmse_escalado * 4.0

    # ✨ NUEVO: Calculamos los porcentajes promedios de atención
    avg_attn_weights = (total_attn_weights / running_samples) * 100.0

    logger.info(f"\n📊 MÉTRICAS FINALES DE TEST:")
    logger.info(f"   - MSE (Escala 0-1): {total_loss_escalado:.4f}")
    logger.info(f"   - RMSE (Escala 0-1): {rmse_escalado:.4f}")
    logger.info(f"   ⭐ RMSE (Estrellas Reales): {rmse_real_estrellas:.4f}")

    # ✨ NUEVO: Impresión de la Explicabilidad
    logger.info(f"\n🧠 EXPLICABILIDAD GLOBAL (PESOS DE ATENCIÓN):")
    logger.info(f"   - Importancia de Categoría: {avg_attn_weights[0].item():.2f}%")
    logger.info(f"   - Importancia de Texto (Nova): {avg_attn_weights[1].item():.2f}%")
    logger.info(f"   - Importancia de Imagen (Nova): {avg_attn_weights[2].item():.2f}%")

    return rmse_real_estrellas