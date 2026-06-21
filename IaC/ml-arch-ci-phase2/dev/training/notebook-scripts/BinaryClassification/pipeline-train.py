import torch
import torch.nn as nn
import torch.optim as optim
import os

def main(df_train, df_valid, df_test, dict_aws):
    print("🚀 Iniciando Orquestador del Recomendador Multimodal (Clasificación Binaria)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Dispositivo activo: {device}")

    df_total = pd.concat([df_train, df_valid, df_test])
    num_users = df_total['userId_idx'].max() + 1
    num_items = df_total['movieId_idx'].max() + 1
    num_genres = len(df_total['genres_multihot'].iloc[0])

    print(f"📊 Universo de datos: {num_users} Usuarios | {num_items} Películas | {num_genres} Categorías")

    # 🌟 Recuerda que tu función create_data_loaders aplica el muestreo negativo al train internamente
    train_loader, valid_loader, test_loader = create_data_loaders(
        df_train, df_valid, df_test,
        dict_embeddings=dict_aws,
        batch_size=256
    )

    print("🏗️ Construyendo Arquitectura de Dos Torres...")
    model = MultimodalExplainableGMF(
        num_users=num_users,
        num_items=num_items,
        num_categories=num_genres,
        emb_dim=64,
        aws_dim=1024
    ).to(device)

    # 🌟 NUEVO: Función de Pérdida para Clasificación Binaria
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    EPOCHS = 20
    MODEL_DIR = "/content/checkpoints"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("\n" + "="*50)
    print("🔥 INICIANDO BUCLE DE ENTRENAMIENTO")
    print("="*50)

    best_val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        model_dir=MODEL_DIR,
        device=device
    )

    print("\n" + "="*50)
    print("🧪 EVALUACIÓN FINAL EN DATOS NUNCA VISTOS (TEST SET)")
    print("="*50)

    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    
    test_loss = test(model, test_loader, criterion, device)
    print(f"\n🎉 ¡Entrenamiento finalizado! BCE Loss final: {test_loss:.4f}")