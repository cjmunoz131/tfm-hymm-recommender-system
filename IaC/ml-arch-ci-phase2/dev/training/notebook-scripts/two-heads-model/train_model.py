import torch
import os
import logging
import sys

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))

def train(model, train_loader, val_loader, criterion_bce, criterion_mse, optimizer, epochs, model_dir, device):
    best_loss = float('inf')
    dataset={'train':train_loader, 'valid':val_loader}
    loss_counter=0

    for epoch in range(1, epochs + 1):
        for phase in ['train', 'valid']:
            logger.info(f"\n--- Epoch {epoch} | Phase: {phase.upper()} ---")
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_bce = 0.0
            running_mse = 0.0
            running_samples = 0

            for step, batch in enumerate(dataset[phase]):
                user = batch['user'].to(device)
                item = batch['item'].to(device)
                cat = batch['genres'].to(device)
                text_emb = batch['text_emb'].to(device)
                img_emb = batch['img_emb'].to(device)

                target_interaction = batch['interaction'].to(device)
                target_rating = batch['rating'].to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    pred_interaction, pred_rating, _ = model(user, item, cat, text_emb, img_emb)
                    pred_interaction = pred_interaction.squeeze()
                    pred_rating = pred_rating.squeeze()

                    # 🌟 1. Pérdida BCE (Sobre TODOS los datos, para rankear)
                    loss_bce = criterion_bce(pred_interaction, target_interaction)

                    # 🌟 2. Pérdida MSE Enmascarada (SOLO sobre las películas que sí vio)
                    mask_positivos = (target_interaction == 1.0)
                    if mask_positivos.sum() > 0:
                        loss_mse = criterion_mse(pred_rating[mask_positivos], target_rating[mask_positivos])
                    else:
                        loss_mse = torch.tensor(0.0).to(device)

                    # 🌟 3. Combinación Multi-Task (Suma directa)
                    loss = loss_bce + loss_mse

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                batch_size = user.size(0)
                running_loss += loss.item() * batch_size
                running_bce += loss_bce.item() * batch_size
                running_mse += loss_mse.item() * batch_size
                running_samples += batch_size

                if step % 100 == 0 and step > 0:
                    logger.info("{} epoch: {} [{}/{}] | Total Loss: {:.4f} (BCE: {:.4f} | MSE: {:.4f})".format(
                            phase, epoch, running_samples, len(dataset[phase].dataset),
                            loss.item(), loss_bce.item(), loss_mse.item()
                        ))

            epoch_loss = running_loss / running_samples
            epoch_bce = running_bce / running_samples
            epoch_mse = running_mse / running_samples

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                    logger.info(f"🌟 ¡NUEVO MEJOR MODELO! | Valid Total Loss: {best_loss:.4f} (BCE: {epoch_bce:.4f}, MSE: {epoch_mse:.4f})")
                    loss_counter = 0
                else:
                    loss_counter += 1

            logger.info(f"Resumen {phase.upper()} | Total: {epoch_loss:.4f} | BCE: {epoch_bce:.4f} | MSE: {epoch_mse:.4f}")

        if loss_counter >= 3:
            logger.info(f"🛑 Early Stopping: Fin del entrenamiento.")
            break

    return best_loss

def test(model, test_loader, criterion_bce, criterion_mse, device):
    model.eval()
    running_loss, running_bce, running_mse = 0.0, 0.0, 0.0
    running_samples = 0
    total_attn_weights = torch.zeros(3).to(device)

    with torch.no_grad():
        for batch in test_loader:
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            cat = batch['genres'].to(device)
            text_emb = batch['text_emb'].to(device)
            img_emb = batch['img_emb'].to(device)
            target_interaction = batch['interaction'].to(device)
            target_rating = batch['rating'].to(device)

            pred_interaction, pred_rating, attn_weights = model(user, item, cat, text_emb, img_emb)
            pred_interaction = pred_interaction.squeeze()
            pred_rating = pred_rating.squeeze()

            loss_bce = criterion_bce(pred_interaction, target_interaction)

            mask_positivos = (target_interaction == 1.0)
            if mask_positivos.sum() > 0:
                loss_mse = criterion_mse(pred_rating[mask_positivos], target_rating[mask_positivos])
            else:
                loss_mse = torch.tensor(0.0).to(device)

            loss = loss_bce + loss_mse

            batch_size = user.size(0)
            running_loss += loss.item() * batch_size
            running_bce += loss_bce.item() * batch_size
            running_mse += loss_mse.item() * batch_size
            running_samples += batch_size
            total_attn_weights += attn_weights.sum(dim=0)

    total_loss = running_loss / running_samples
    logger.info(f"\n📊 MÉTRICAS FINALES DE TEST (Multi-Task):")
    logger.info(f"   - Total Loss: {total_loss:.4f}")
    logger.info(f"   - BCE Loss (Ranking): {running_bce / running_samples:.4f}")
    logger.info(f"   - MSE Loss (Calidad): {running_mse / running_samples:.4f}")

    return total_loss