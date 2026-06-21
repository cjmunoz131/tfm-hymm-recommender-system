import torch
import os
import logging
import sys

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))

def train(model, train_loader, val_loader, criterion, optimizer, epochs, model_dir, device):
    best_loss = float('inf')
    dataset={'train':train_loader, 'valid':val_loader}
    loss_counter=0

    for epoch in range(1, epochs + 1):
        for phase in ['train', 'valid']:
            logger.info(f"\n--- Epoch {epoch} | Phase: {phase.upper()} ---")
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_samples = 0

            for step, batch in enumerate(dataset[phase]):
                user = batch['user'].to(device)
                item = batch['item'].to(device)
                cat = batch['genres'].to(device)
                text_emb = batch['text_emb'].to(device)
                img_emb = batch['img_emb'].to(device)
                
                # 🌟 Variable objetivo es la interacción binaria
                target_interaction = batch['interaction'].to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    pred_interaction, _ = model(user, item, cat, text_emb, img_emb)
                    pred_interaction = pred_interaction.squeeze()

                    # 🌟 Pérdida BCE (Entropía Cruzada Binaria)
                    loss = criterion(pred_interaction, target_interaction)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                batch_size = user.size(0)
                running_loss += loss.item() * batch_size
                running_samples += batch_size

                if step % 100 == 0 and step > 0:
                    logger.info("{} epoch: {}  [{}/{} ({:.0f}%)] Batch BCELoss: {:.4f}".format(
                            phase, epoch, running_samples, len(dataset[phase].dataset),
                            100.0 * (running_samples / len(dataset[phase].dataset)), loss.item()
                        ))

            epoch_loss = running_loss / running_samples

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                    logger.info(f"🌟 ¡NUEVO MEJOR MODELO! | Valid BCELoss: {best_loss:.4f}")
                    loss_counter = 0 
                else:
                    loss_counter += 1

            logger.info(f"Resumen {phase.upper()} | BCELoss: {epoch_loss:.4f}")

        if loss_counter >= 3:
            logger.info(f"🛑 Early Stopping: Fin del entrenamiento.")
            break

    return best_loss

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0
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

            pred_interaction, attn_weights = model(user, item, cat, text_emb, img_emb)
            pred_interaction = pred_interaction.squeeze()

            loss = criterion(pred_interaction, target_interaction)
            
            batch_size = user.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size
            total_attn_weights += attn_weights.sum(dim=0)

    total_loss = running_loss / running_samples
    avg_attn_weights = (total_attn_weights / running_samples) * 100.0

    logger.info(f"\n📊 MÉTRICAS FINALES DE TEST (Clasificación):")
    logger.info(f"   - BCE Loss (LogLoss): {total_loss:.4f}")
    
    logger.info(f"\n🧠 EXPLICABILIDAD GLOBAL (PESOS DE ATENCIÓN):")
    logger.info(f"   - Importancia de Categoría: {avg_attn_weights[0].item():.2f}%")
    logger.info(f"   - Importancia de Texto (Nova): {avg_attn_weights[1].item():.2f}%")
    logger.info(f"   - Importancia de Imagen (Nova): {avg_attn_weights[2].item():.2f}%")

    return total_loss