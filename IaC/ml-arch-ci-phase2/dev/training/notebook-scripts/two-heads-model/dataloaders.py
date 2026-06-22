import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MultimodalRecDataset(Dataset):
    def __init__(self, df, dict_embeddings):
        self.num_interacciones = len(df)

        self.users = torch.tensor(df['userId_idx'].values, dtype=torch.long)
        self.items = torch.tensor(df['movieId_idx'].values, dtype=torch.long)

        matriz_generos = np.vstack(df['genres_multihot'].values)
        self.genres = torch.tensor(matriz_generos, dtype=torch.float32)

        # 🌟 Variable Binaria (Para Ranking - BCE)
        interacciones = (df['rating_scaled'].values > 0.0).astype(np.float32)
        self.interactions = torch.tensor(interacciones, dtype=torch.float32)

        # 🌟 Variable Continua (Para Calidad - MSE)
        self.ratings = torch.tensor(df['rating_scaled'].values, dtype=torch.float32)

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
            'interaction': self.interactions[idx], # Para Cabeza 1
            'rating': self.ratings[idx]            # Para Cabeza 2
        }
        
import pandas as pd
import numpy as np


def apply_negative_sampling_per_positive(
    df_train: pd.DataFrame,
    global_history: dict,
    item_pool: set,
    neg_ratio: int,
    user_map: dict,
    item_idx_map: dict,
    item_genres_map: dict,
) -> pd.DataFrame:
    """
    Genera negativos por cada interacción positiva (ratio 1:neg_ratio).
    
    A diferencia de la versión fija por usuario, aquí cada fila positiva
    genera exactamente `neg_ratio` muestras negativas. Esto garantiza un
    balance consistente independiente del número de interacciones del usuario.
    
    Ejemplo con neg_ratio=4:
      - Usuario con 10 positivos → 40 negativos (ratio 1:4)
      - Usuario con 50 positivos → 200 negativos (ratio 1:4)
    """
    print(f"⚙️ Generando negativos por POSITIVO (ratio 1:{neg_ratio})...")

    df_positives = df_train.copy()
    item_pool_list = list(item_pool)

    negative_rows = []

    for _, row in df_positives.iterrows():
        user_id = row['userId']
        interacted = global_history.get(user_id, set())
        available = list(item_pool - interacted)

        n_samples = min(neg_ratio, len(available))
        if n_samples == 0:
            continue

        sampled_items = np.random.choice(available, size=n_samples, replace=False)

        for neg_item in sampled_items:
            negative_rows.append({
                'userId': user_id,
                'movieId': neg_item,
            })

    df_negatives = pd.DataFrame(negative_rows)

    # Restaurar columnas necesarias para PyTorch
    df_negatives['userId_idx'] = df_negatives['userId'].map(user_map)
    df_negatives['movieId_idx'] = df_negatives['movieId'].map(item_idx_map)
    df_negatives['genres_multihot'] = df_negatives['movieId'].map(item_genres_map)
    df_negatives['rating_scaled'] = 0.0

    if 'timestamp' in df_positives.columns:
        last_timestamps = df_positives.groupby('userId')['timestamp'].max().to_dict()
        df_negatives['timestamp'] = df_negatives['userId'].map(last_timestamps)

    # Conservar columnas que PyTorch necesita
    columnas_necesarias = ['userId', 'movieId', 'userId_idx', 'movieId_idx', 'genres_multihot', 'rating_scaled']
    if 'timestamp' in df_positives.columns:
        columnas_necesarias.append('timestamp')

    df_final = pd.concat(
        [df_positives[columnas_necesarias], df_negatives[columnas_necesarias]],
        ignore_index=True,
    )
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    ratio_real = len(df_negatives) / max(len(df_positives), 1)
    print(f"  ↳ Train final: {len(df_final):,} filas "
          f"(Positivos: {len(df_positives):,} | Negativos: {len(df_negatives):,} | "
          f"Ratio efectivo: 1:{ratio_real:.1f})")
    return df_final

def create_data_loaders(df_train, df_valid, df_test, dict_embeddings, batch_size=256, neg_ratio=4, num_workers=2):
    """
    Crea DataLoaders para el modelo Two-Heads con muestreo negativo por ratio.
    
    Args:
        neg_ratio: Número de negativos por cada interacción positiva (default: 4, ratio 1:4).
    """
    print("🚀 Iniciando preprocesamiento de DataLoaders...")
    print(f"   Negative sampling ratio: 1:{neg_ratio} (por positivo)")

    df_all = pd.concat([df_train, df_valid, df_test])
    item_pool = set(df_all['movieId'].unique())
    global_history = df_all.groupby('userId')['movieId'].apply(set).to_dict()

    # Usar subset=['columna_id'] para no hashear los arrays de NumPy
    df_users_unique = df_all[['userId', 'userId_idx']].drop_duplicates(subset=['userId']).set_index('userId')
    df_items_unique = df_all[['movieId', 'movieId_idx', 'genres_multihot']].drop_duplicates(subset=['movieId']).set_index('movieId')

    user_map = df_users_unique['userId_idx'].to_dict()
    item_idx_map = df_items_unique['movieId_idx'].to_dict()
    item_genres_map = df_items_unique['genres_multihot'].to_dict()

    # Muestreo negativo por ratio (1:neg_ratio por positivo)
    df_train_ns = apply_negative_sampling_per_positive(
        df_train,
        global_history,
        item_pool,
        neg_ratio=neg_ratio,
        user_map=user_map,
        item_idx_map=item_idx_map,
        item_genres_map=item_genres_map,
    )

    # Instanciamos DataLoaders
    train_dataset = MultimodalRecDataset(df_train_ns, dict_embeddings)
    valid_dataset = MultimodalRecDataset(df_valid, dict_embeddings)
    test_dataset = MultimodalRecDataset(df_test, dict_embeddings)

    usar_pin_memory = torch.cuda.is_available()

    print(f"🚀 Creando DataLoaders (Batch Size: {batch_size}, Workers: {num_workers}, Pin Memory: {usar_pin_memory})...")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=usar_pin_memory, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=usar_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=usar_pin_memory)

    return train_loader, valid_loader, test_loader