"""
HYMM-REC: DataLoaders para Training Jobs en SageMaker (Shared Module)
=====================================================================
Proporciona Datasets y DataLoaders para los dos modos de entrenamiento:

  1. Regresión (mode='regression'):
     - Predice rating escalado [0, 1]
     - Sin muestreo negativo (solo interacciones reales)
     - Dataset retorna: user, item, genres, text_emb, img_emb, rating

  2. Multi-Task / Two-Heads (mode='multitask'):
     - Predice AMBOS: interacción binaria + rating escalado
     - Con muestreo negativo in-memory (solo para train)
     - Dataset retorna: user, item, genres, text_emb, img_emb, interaction, rating
     - Los negativos reciben: interaction=0.0, rating=0.0
     - El MSE se calcula SOLO sobre positivos (enmascarado en el training loop)

El muestreo negativo se realiza EN MEMORIA durante la construcción del
DataLoader, sin necesidad de un Processing Job adicional. Esto es más
eficiente para datasets de 100K-32M interacciones dado que:
  - No multiplica x20 el tamaño del dataset en S3/disco
  - Aprovecha la RAM de la instancia de training (ml.g4dn.xlarge = 16GB)

Uso (Regresión):
    from dataloaders import load_datasets_and_create_loaders
    loaders = load_datasets_and_create_loaders(data_dir, embeddings_dir, mode='regression', batch_size=256)

Uso (Multi-Task Two-Heads):
    from dataloaders import load_datasets_and_create_loaders
    loaders = load_datasets_and_create_loaders(data_dir, embeddings_dir, mode='multitask', batch_size=256, neg_ratio=4)
"""

import logging
import os
import pickle
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))


# ============================================================
# DATASETS
# ============================================================
class RegressionRecDataset(Dataset):
    """
    Dataset para el modo de regresión.
    Variable objetivo: rating_scaled ∈ [0, 1]
    """

    def __init__(self, df: pd.DataFrame, dict_embeddings: Dict):
        self.num_interacciones = len(df)
        self.users = torch.tensor(df["userId_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["movieId_idx"].values, dtype=torch.long)

        matriz_generos = np.vstack(df["genres_multihot"].values)
        self.genres = torch.tensor(matriz_generos, dtype=torch.float32)

        self.ratings = torch.tensor(df["rating_scaled"].values, dtype=torch.float32)
        self.item_ids_original = df["movieId"].values
        self.dict_embeddings = dict_embeddings

    def __len__(self):
        return self.num_interacciones

    def __getitem__(self, idx):
        original_item_id = self.item_ids_original[idx]
        emb_entry = self.dict_embeddings.get(original_item_id, None)

        if emb_entry is not None:
            text_emb = torch.tensor(emb_entry["text_emb"], dtype=torch.float32)
            img_emb = torch.tensor(emb_entry["img_emb"], dtype=torch.float32)
        else:
            text_emb = torch.zeros(1024, dtype=torch.float32)
            img_emb = torch.zeros(1024, dtype=torch.float32)

        return {
            "user": self.users[idx],
            "item": self.items[idx],
            "genres": self.genres[idx],
            "text_emb": text_emb,
            "img_emb": img_emb,
            "rating": self.ratings[idx],
        }


class MultiTaskRecDataset(Dataset):
    """
    Dataset para el modo Multi-Task (Two-Heads).
    Variables objetivo:
      - interaction ∈ {0.0, 1.0} → Para cabeza BCE (ranking/retrieval)
      - rating ∈ [0, 1] → Para cabeza MSE (calidad, solo sobre positivos)

    Los negativos tienen: interaction=0.0, rating=0.0
    El enmascaramiento del MSE (solo positivos) se hace en el training loop.
    """

    def __init__(self, df: pd.DataFrame, dict_embeddings: Dict):
        self.num_interacciones = len(df)
        self.users = torch.tensor(df["userId_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["movieId_idx"].values, dtype=torch.long)

        matriz_generos = np.vstack(df["genres_multihot"].values)
        self.genres = torch.tensor(matriz_generos, dtype=torch.float32)

        # Variable binaria: rating_scaled > 0 → positivo (1.0), else → negativo (0.0)
        interacciones = (df["rating_scaled"].values > 0.0).astype(np.float32)
        self.interactions = torch.tensor(interacciones, dtype=torch.float32)

        # Variable continua: rating_scaled (0.0 para negativos)
        self.ratings = torch.tensor(df["rating_scaled"].values, dtype=torch.float32)

        self.item_ids_original = df["movieId"].values
        self.dict_embeddings = dict_embeddings

    def __len__(self):
        return self.num_interacciones

    def __getitem__(self, idx):
        original_item_id = self.item_ids_original[idx]
        emb_entry = self.dict_embeddings.get(original_item_id, None)

        if emb_entry is not None:
            text_emb = torch.tensor(emb_entry["text_emb"], dtype=torch.float32)
            img_emb = torch.tensor(emb_entry["img_emb"], dtype=torch.float32)
        else:
            text_emb = torch.zeros(1024, dtype=torch.float32)
            img_emb = torch.zeros(1024, dtype=torch.float32)

        return {
            "user": self.users[idx],
            "item": self.items[idx],
            "genres": self.genres[idx],
            "text_emb": text_emb,
            "img_emb": img_emb,
            "interaction": self.interactions[idx],  # Para Cabeza 1 (BCE)
            "rating": self.ratings[idx],            # Para Cabeza 2 (MSE)
        }


# ============================================================
# NEGATIVE SAMPLING (IN-MEMORY)
# ============================================================
def apply_negative_sampling(
    df_train: pd.DataFrame,
    df_all: pd.DataFrame,
    neg_ratio: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera muestras negativas EXCLUSIVAMENTE para el dataset de entrenamiento.
    Se ejecuta en memoria durante la inicialización del DataLoader.

    Estrategia (ratio por positivo):
      - Para cada interacción positiva, muestrea `neg_ratio` ítems que el
        usuario NUNCA ha visto (en todo el historial global, no solo train)
      - Los negativos reciben rating_scaled = 0.0
      - Se restauran userId_idx, movieId_idx y genres_multihot desde mapeos

    Esto garantiza un balance consistente independiente del número de
    interacciones del usuario (ratio 1:neg_ratio para todos).

    Args:
        df_train: DataFrame de entrenamiento (solo positivos)
        df_all: DataFrame completo (train+val+test) para historial global
        neg_ratio: Negativos por cada interacción positiva (default: 4 → ratio 1:4)
        seed: Semilla para reproducibilidad

    Returns:
        DataFrame con positivos + negativos mezclados
    """
    rng = np.random.default_rng(seed)

    # Pool global de ítems y historial
    item_pool = set(df_all["movieId"].unique())
    global_history = df_all.groupby("userId")["movieId"].apply(set).to_dict()

    # Mapeos para restaurar columnas
    df_users_unique = (
        df_all[["userId", "userId_idx"]]
        .drop_duplicates(subset=["userId"])
        .set_index("userId")
    )
    df_items_unique = (
        df_all[["movieId", "movieId_idx", "genres_multihot"]]
        .drop_duplicates(subset=["movieId"])
        .set_index("movieId")
    )

    user_map = df_users_unique["userId_idx"].to_dict()
    item_idx_map = df_items_unique["movieId_idx"].to_dict()
    item_genres_map = df_items_unique["genres_multihot"].to_dict()

    logger.info(
        f"Generando negativos por POSITIVO (ratio 1:{neg_ratio}) "
        f"({len(df_train):,} positivos, pool: {len(item_pool):,} ítems)..."
    )

    # Generar negativos por cada interacción positiva
    negative_records = []
    for _, row in df_train.iterrows():
        user_id = row["userId"]
        interacted = global_history.get(user_id, set())
        available = list(item_pool - interacted)
        n_samples = min(neg_ratio, len(available))

        if n_samples == 0:
            continue

        sampled_items = rng.choice(available, size=n_samples, replace=False)

        for item_id in sampled_items:
            negative_records.append(
                {
                    "userId": user_id,
                    "movieId": item_id,
                    "userId_idx": user_map.get(user_id),
                    "movieId_idx": item_idx_map.get(item_id),
                    "genres_multihot": item_genres_map.get(item_id),
                    "rating_scaled": 0.0,
                }
            )

    df_negatives = pd.DataFrame(negative_records)

    # Filtrar negativos sin mapeo válido (ítems nuevos que no tienen encoding)
    df_negatives = df_negatives.dropna(subset=["userId_idx", "movieId_idx", "genres_multihot"])
    df_negatives["userId_idx"] = df_negatives["userId_idx"].astype(int)
    df_negatives["movieId_idx"] = df_negatives["movieId_idx"].astype(int)

    # Columnas necesarias para PyTorch
    columnas = ["userId", "movieId", "userId_idx", "movieId_idx", "genres_multihot", "rating_scaled"]
    df_positives = df_train[columnas].copy()

    df_final = pd.concat([df_positives, df_negatives[columnas]], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=seed).reset_index(drop=True)

    ratio_real = len(df_negatives) / max(len(df_positives), 1)
    logger.info(
        f"  Train final: {len(df_final):,} filas "
        f"(Positivos: {len(df_positives):,} | Negativos: {len(df_negatives):,} | "
        f"Ratio efectivo: 1:{ratio_real:.1f})"
    )
    return df_final


# ============================================================
# CARGA DE DATOS
# ============================================================
def load_parquet_dataset(path: str) -> pd.DataFrame:
    """
    Carga un dataset desde un directorio de Parquet.
    Soporta tanto un directorio con un .parquet dentro como un archivo directo.
    """
    if os.path.isdir(path):
        # Buscar archivos parquet dentro del directorio
        parquet_files = [f for f in os.listdir(path) if f.endswith(".parquet")]
        if parquet_files:
            filepath = os.path.join(path, parquet_files[0])
        else:
            # Intentar leer el directorio completo (multi-part parquet)
            filepath = path
    else:
        filepath = path

    df = pd.read_parquet(filepath)
    logger.info(f"  Cargado: {filepath} → {len(df):,} filas")
    return df


def load_embeddings(path: str) -> Dict:
    """Carga el diccionario de embeddings multimodales (pickle)."""
    if os.path.isdir(path):
        pkl_files = [f for f in os.listdir(path) if f.endswith(".pkl")]
        if pkl_files:
            filepath = os.path.join(path, pkl_files[0])
        else:
            raise FileNotFoundError(f"No se encontró .pkl en {path}")
    else:
        filepath = path

    with open(filepath, "rb") as f:
        embeddings = pickle.load(f)

    logger.info(f"  Embeddings cargados: {filepath} → {len(embeddings):,} ítems")
    return embeddings


def load_encoders(path: str) -> Dict:
    """Carga el diccionario de encoders (LabelEncoder, MultiLabelBinarizer) desde pickle."""
    if os.path.isdir(path):
        pkl_files = [f for f in os.listdir(path) if f.endswith(".pkl")]
        if pkl_files:
            filepath = os.path.join(path, pkl_files[0])
        else:
            raise FileNotFoundError(f"No se encontró .pkl en {path}")
    else:
        filepath = path

    with open(filepath, "rb") as f:
        encoders = pickle.load(f)

    logger.info(
        f"  Encoders cargados: {filepath} → "
        f"{len(encoders['le_user'].classes_):,} users, "
        f"{len(encoders['le_item'].classes_):,} items, "
        f"{len(encoders['mlb'].classes_)} categorías"
    )
    return encoders


# ============================================================
# FACTORY PRINCIPAL
# ============================================================
def load_datasets_and_create_loaders(
    data_dir: str,
    embeddings_dir: str,
    encoders_dir: str = None,
    mode: str = "regression",
    batch_size: int = 256,
    neg_ratio: int = 4,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Factory principal: carga datos, aplica transformaciones y crea DataLoaders.

    Args:
        data_dir: Directorio base con subcarpetas train/, val/, test/
        embeddings_dir: Directorio con embeddings_catalog.pkl
        encoders_dir: Directorio con encoders.pkl (vocabulario completo del LabelEncoder).
                      Si se provee, usa el vocabulario COMPLETO para dimensionar embeddings
                      (incluye cold-start items/users). Si no, usa max(dataset idx) + 1.
        mode: 'regression' o 'multitask'
        batch_size: Tamaño de batch
        neg_ratio: Negativos por cada positivo (solo en mode='multitask', ratio 1:neg_ratio)
        num_workers: Workers para DataLoader
        seed: Semilla para muestreo negativo

    Returns:
        (train_loader, val_loader, test_loader, metadata)
        metadata incluye: num_users, num_items, num_categories
    """
    logger.info(f"Cargando datasets desde: {data_dir}")
    logger.info(f"Modo: {mode} | Batch: {batch_size} | Workers: {num_workers}")
    if mode == "multitask":
        logger.info(f"Negative sampling ratio: 1:{neg_ratio} (por positivo)")

    # 1. Cargar splits
    df_train = load_parquet_dataset(os.path.join(data_dir, "train"))
    df_val = load_parquet_dataset(os.path.join(data_dir, "val"))
    df_test = load_parquet_dataset(os.path.join(data_dir, "test"))

    # 2. Cargar embeddings multimodales
    dict_embeddings = load_embeddings(embeddings_dir)

    # 3. Calcular dimensiones globales del vocabulario
    df_all = pd.concat([df_train, df_val, df_test])

    if encoders_dir and os.path.exists(encoders_dir):
        # Usar vocabulario COMPLETO del LabelEncoder (incluye cold-start)
        encoders = load_encoders(encoders_dir)
        num_users = len(encoders["le_user"].classes_)
        num_items = len(encoders["le_item"].classes_)
        num_categories = len(encoders["mlb"].classes_)
        logger.info(f"  Dimensiones desde encoders.pkl (vocabulario completo, incluye cold-start)")
    else:
        # Fallback: usar max del dataset (no cubre cold-start)
        num_users = int(df_all["userId_idx"].max()) + 1
        num_items = int(df_all["movieId_idx"].max()) + 1
        num_categories = len(df_all["genres_multihot"].iloc[0])
        logger.info(f"  Dimensiones desde max(dataset idx) — sin cobertura cold-start")

    logger.info(
        f"Universo: {num_users:,} usuarios | {num_items:,} ítems | {num_categories} categorías"
    )

    metadata = {
        "num_users": num_users,
        "num_items": num_items,
        "num_categories": num_categories,
    }

    # 4. Preparar DataLoaders según modo
    use_pin_memory = torch.cuda.is_available()

    if mode == "multitask":
        # Multi-Task (Two-Heads): Aplicar muestreo negativo al train (ratio por positivo)
        df_train_ns = apply_negative_sampling(
            df_train, df_all, neg_ratio=neg_ratio, seed=seed
        )
        train_dataset = MultiTaskRecDataset(df_train_ns, dict_embeddings)
        val_dataset = MultiTaskRecDataset(df_val, dict_embeddings)
        test_dataset = MultiTaskRecDataset(df_test, dict_embeddings)
    else:
        # Regresión (single-head): sin muestreo negativo
        train_dataset = RegressionRecDataset(df_train, dict_embeddings)
        val_dataset = RegressionRecDataset(df_val, dict_embeddings)
        test_dataset = RegressionRecDataset(df_test, dict_embeddings)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    logger.info(
        f"DataLoaders creados: Train={len(train_dataset):,} | "
        f"Val={len(val_dataset):,} | Test={len(test_dataset):,}"
    )

    return train_loader, val_loader, test_loader, metadata
