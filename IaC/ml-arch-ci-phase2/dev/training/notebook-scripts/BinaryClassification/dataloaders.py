import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MultimodalRecDataset(Dataset):
    def __init__(self, df, dict_embeddings):
        self.num_interacciones = len(df)

        self.users = torch.tensor(df['userId_idx'].values, dtype=torch.long)
        self.items = torch.tensor(df['movieId_idx'].values, dtype=torch.long)

        matriz_generos = np.vstack(df['genres_multihot'].values)
        self.genres = torch.tensor(matriz_generos, dtype=torch.float32)

        # 🌟 NUEVO: Variable Binaria (1.0 = Positivo, 0.0 = Negativo)
        # Asumimos que los negativos generados tienen rating_scaled = 0.0
        interacciones = (df['rating_scaled'].values > 0.0).astype(np.float32)
        self.interactions = torch.tensor(interacciones, dtype=torch.float32)

        self.item_ids_original = df['movieId'].values
        self.dict_embeddings = dict_embeddings

    def __len__(self):
        return self.num_interacciones

    def __getitem__(self, idx):
        original_item_id = self.item_ids_original[idx]

        text_emb = torch.tensor(self.dict_embeddings[original_item_id]['text_emb'], dtype=torch.float32)
        img_emb = torch.tensor(self.dict_embeddings[original_item_id]['img_emb'], dtype=torch.float32)

        return {
            'user': self.users[idx],
            'item': self.items[idx],
            'genres': self.genres[idx],
            'text_emb': text_emb,
            'img_emb': img_emb,
            'interaction': self.interactions[idx] # 🌟 Ahora pasamos la etiqueta binaria
        }
        
import pandas as pd
import numpy as np

def apply_negative_sampling_train_fixed(df_train: pd.DataFrame, global_history: dict, item_pool: set, num_negatives_fixed: int, user_map: dict, item_idx_map: dict, item_genres_map: dict) -> pd.DataFrame:
    """
    Genera negativos EXCLUSIVAMENTE para el dataset de entrenamiento (Versión Fija).
    Restaura las características (índices y géneros) para que PyTorch no falle.
    """
    print(f"⚙️ Generando negativos FIJOS para Train ({num_negatives_fixed} por usuario)...")
    
    df_positives = df_train.copy()
    unique_users = df_positives['userId'].unique()
    interact_status = pd.DataFrame({'userId': unique_users})
    
    # Historial global
    interact_status['interacted_items'] = interact_status['userId'].map(global_history)
    
    def sample_negatives_fixed(interacted_set):
        if not isinstance(interacted_set, set):
            interacted_set = set()
            
        available_negatives = list(item_pool - interacted_set)
        n_samples = min(num_negatives_fixed, len(available_negatives))
        
        if n_samples == 0: return []
        return np.random.choice(available_negatives, size=n_samples, replace=False).tolist()

    # Expandimos los negativos
    interact_status['negative_samples'] = interact_status['interacted_items'].apply(sample_negatives_fixed)
    df_negatives = interact_status.drop(columns=['interacted_items']).explode('negative_samples').dropna()
    df_negatives = df_negatives.rename(columns={'negative_samples': 'movieId'})
    
    # 🌟 RESTAURAMOS LAS COLUMNAS PERDIDAS PARA PYTORCH
    df_negatives['userId_idx'] = df_negatives['userId'].map(user_map)
    df_negatives['movieId_idx'] = df_negatives['movieId'].map(item_idx_map)
    df_negatives['genres_multihot'] = df_negatives['movieId'].map(item_genres_map)
    
    # 🌟 Variable objetivo: usamos rating_scaled (0.0 para negativos)
    df_negatives['rating_scaled'] = 0.0 
    
    if 'timestamp' in df_positives.columns:
        last_timestamps = df_positives.groupby('userId')['timestamp'].max().to_dict()
        df_negatives['timestamp'] = df_negatives['userId'].map(last_timestamps)
    
    # 🌟 Conservamos TODAS las columnas que PyTorch necesita
    columnas_necesarias = ['userId', 'movieId', 'userId_idx', 'movieId_idx', 'genres_multihot', 'rating_scaled']
    if 'timestamp' in df_positives.columns:
        columnas_necesarias.append('timestamp')
        
    df_final = pd.concat([df_positives[columnas_necesarias], df_negatives[columnas_necesarias]], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  ↳ Train final: {len(df_final):,} filas (Positivos: {len(df_positives):,} | Negativos: {len(df_negatives):,})")
    return df_final

def create_data_loaders(df_train, df_valid, df_test, dict_embeddings, batch_size=256):
    
    print("🚀 Iniciando preprocesamiento de DataLoaders...")
    
    df_all = pd.concat([df_train, df_valid, df_test])
    item_pool = set(df_all['movieId'].unique())
    global_history = df_all.groupby('userId')['movieId'].apply(set).to_dict()
    
    # 🌟 LA SOLUCIÓN: Usar subset=['columna_id'] para no hashear los arrays de NumPy
    df_users_unique = df_all[['userId', 'userId_idx']].drop_duplicates(subset=['userId']).set_index('userId')
    df_items_unique = df_all[['movieId', 'movieId_idx', 'genres_multihot']].drop_duplicates(subset=['movieId']).set_index('movieId')
    
    user_map = df_users_unique['userId_idx'].to_dict()
    item_idx_map = df_items_unique['movieId_idx'].to_dict()
    item_genres_map = df_items_unique['genres_multihot'].to_dict()
    
    # Aplicamos muestreo pasando los mapeos
    df_train_ns = apply_negative_sampling_train_fixed(
        df_train, 
        global_history, 
        item_pool, 
        num_negatives_fixed=20,
        user_map=user_map,
        item_idx_map=item_idx_map,
        item_genres_map=item_genres_map
    )

    # Instanciamos DataLoaders
    train_dataset = MultimodalRecDataset(df_train_ns, dict_embeddings)
    valid_dataset = MultimodalRecDataset(df_valid, dict_embeddings)
    test_dataset = MultimodalRecDataset(df_test, dict_embeddings)

    n_workers = 2
    usar_pin_memory = torch.cuda.is_available() 

    print(f"🚀 Creando DataLoaders (Batch Size: {batch_size}, Workers: {n_workers}, Pin Memory: {usar_pin_memory})...")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=usar_pin_memory, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=usar_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=usar_pin_memory)
    
    return train_loader, valid_loader, test_loader