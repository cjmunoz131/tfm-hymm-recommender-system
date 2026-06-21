import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
import os

# Asegúrate de importar aquí tus funciones previas:
# from tu_archivo import create_data_loaders, train, test, MultimodalGMF

def main(df_train, df_valid, df_test, dict_aws):
    print("🚀 Iniciando Orquestador del Recomendador Multimodal...")

    # 0. Configuración de Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Dispositivo activo: {device}")

    # 1. CARGA DE DATOS REALES
    #print("📂 Cargando datasets y embeddings...")
    # df_train = pd.read_csv('train.csv')
    # df_valid = pd.read_csv('valid.csv')
    # df_test = pd.read_csv('test.csv')
    # dict_aws = pickle.load(open('embeddings_nova_9000_final.pkl', 'rb'))

    # Simulación para que el código no falle si lo pegas directamente:
    # Recuerda borrar esto y usar tus datos reales
    #df_train, df_valid, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    #dict_aws = {}

    # 2. CALCULAR DIMENSIONES GLOBALES (CRÍTICO)
    # Concatenamos temporalmente para asegurar que vemos el ID máximo de todo el universo
    df_total = pd.concat([df_train, df_valid, df_test])
    num_users = df_total['userId_idx'].max() + 1
    num_items = df_total['movieId_idx'].max() + 1
    num_genres = len(df_total['genres_multihot'].iloc[0])

    print(f"📊 Universo de datos: {num_users} Usuarios | {num_items} Películas | {num_genres} Categorías")

    # 3. Crear DataLoaders (Usando la función optimizada que corregimos)
    train_loader, valid_loader, test_loader = create_data_loaders(
        df_train, df_valid, df_test,
        dict_embeddings=dict_aws,
        batch_size=256
    )

    # 4. Instanciar la Arquitectura de Red
    print("🏗️ Construyendo Arquitectura de Dos Torres...")
    model = MultimodalExplainableGMF(
        num_users=num_users,
        num_items=num_items,
        num_categories=num_genres,
        emb_dim=64,
        aws_dim=1024 # Dimensión oficial de Amazon Nova
    ).to(device)

    # 5. Definir Función de Error y Optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 6. EL ENTRENAMIENTO (Llama a tu función con Early Stopping)
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

    # 7. LA EVALUACIÓN FINAL (El momento de la verdad para tu TFM)
    print("\n" + "="*50)
    print("🧪 EVALUACIÓN FINAL EN DATOS NUNCA VISTOS (TEST SET)")
    print("="*50)

    # Cargamos los pesos del MEJOR modelo encontrado en validación,
    # no el de la última época (que podría tener overfitting)
    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    print(f"Cargado el mejor modelo histórico desde: {best_model_path}")

    # Ejecutamos la función de test
    test_rmse = test(model, test_loader, criterion, device)

    print("\n🎉 ¡Flujo de Machine Learning completado con éxito!")
    print(f"Métrica Final para el TFM -> RMSE en Test: {test_rmse:.4f}")
    
    import pickle
ruta_guardado = '/content/drive/MyDrive/VIU-TFM/feature_store/embeddings_nova_multimodal_final.pkl'
with open(ruta_guardado, 'rb') as f:
    catalogo_final = pickle.load(f)

main(df_train, df_valid, df_test, catalogo_final)