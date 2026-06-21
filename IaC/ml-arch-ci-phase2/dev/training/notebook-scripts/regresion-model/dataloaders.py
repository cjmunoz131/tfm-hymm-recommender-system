import torch
from torch.utils.data import Dataset, DataLoader

class MultimodalRecDataset(Dataset):
    """
    Dataset de PyTorch que alimenta la red neuronal con interacciones reales.
    """
    def __init__(self, df, dict_embeddings):
        self.num_interacciones = len(df)

        # 1. ÍNDICES PARA PYTORCH (Label Encoded)
        self.users = torch.tensor(df['userId_idx'].values, dtype=torch.long)
        self.items = torch.tensor(df['movieId_idx'].values, dtype=torch.long)

        # 2. VECTORES MULTI-HOT (Géneros)
        # Convertimos la columna de listas/arrays en una matriz 2D limpia
        matriz_generos = np.vstack(df['genres_multihot'].values)
        self.genres = torch.tensor(matriz_generos, dtype=torch.float32)

        # 3. VARIABLE OBJETIVO (Rating Escalado)
        self.ratings = torch.tensor(df['rating_scaled'].values, dtype=torch.float32)

        # 4. PUENTE PARA AWS (IDs Originales)
        self.item_ids_original = df['movieId'].values
        self.dict_embeddings = dict_embeddings

    def __len__(self):
        return self.num_interacciones

    def __getitem__(self, idx):
        # Rescatamos el ID original para buscar en el diccionario de Amazon Nova
        original_item_id = self.item_ids_original[idx]

        # Obtenemos los tensores pesados (1024D)
        text_emb = torch.tensor(self.dict_embeddings[original_item_id]['text_emb'], dtype=torch.float32)
        img_emb = torch.tensor(self.dict_embeddings[original_item_id]['img_emb'], dtype=torch.float32)

        # Retornamos el diccionario completo para el batch
        return {
            'user': self.users[idx],
            'item': self.items[idx],
            'genres': self.genres[idx],
            'text_emb': text_emb,
            'img_emb': img_emb,
            'rating': self.ratings[idx]
        }

def create_data_loaders(df_train, df_valid, df_test, dict_embeddings, batch_size = 256):
    train_dataset = MultimodalRecDataset(df_train, dict_embeddings)
    valid_dataset = MultimodalRecDataset(df_valid, dict_embeddings)
    test_dataset = MultimodalRecDataset(df_test, dict_embeddings)

    n_workers = 2
    usar_pin_memory = torch.cuda.is_available() # Solo es útil si hay GPU

    print(f"🚀 Creando DataLoaders (Batch Size: {batch_size}, Workers: {n_workers}, Pin Memory: {usar_pin_memory})...")

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=n_workers, pin_memory=usar_pin_memory,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=n_workers, pin_memory=usar_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=n_workers, pin_memory=usar_pin_memory)
    return train_loader, valid_loader, test_loader