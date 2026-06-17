"""
SageMaker Processing Job 3: Dataset Preparation (Filter + Temporal Splits)
==========================================================================
Lee el Parquet consolidado de features (output del Job 1), aplica filtrado
de ruido simple, genera splits temporales-estratificados, y copia artefactos
al directorio de output para que el Training Job tenga todo consolidado.

Pipeline:
  1. Carga features desde Parquet consolidado (pandas)
  2. Filtrado de ruido: elimina usuarios/items con < N interacciones
  3. Split temporal-estratificado por usuario (80/10/10)
  4. Persiste cold-starts (interacciones descartadas)
  5. Copia encoders + embeddings al output

Inputs:
  - /opt/ml/processing/input/features/    → feature_interactions.parquet (Job 1)
  - /opt/ml/processing/input/encoders/    → encoders.pkl
  - /opt/ml/processing/input/embeddings/  → embeddings_catalog.pkl

Outputs (→ Platinum bucket):
  - /opt/ml/processing/output/train/        → Parquet
  - /opt/ml/processing/output/val/          → Parquet
  - /opt/ml/processing/output/test/         → Parquet
  - /opt/ml/processing/output/cold-starts/  → Parquet
  - /opt/ml/processing/output/encoders/     → encoders.pkl (copia)
  - /opt/ml/processing/output/embeddings/   → embeddings_catalog.pkl (copia)

Processor: SKLearnProcessor (ml.m5.large para 100K, ml.m5.xlarge para 32M)
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURACIÓN
# ============================================================
@dataclass
class SplitConfig:
    """Configuración del pipeline de splits."""

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    min_user_interactions: int = 5
    min_item_interactions: int = 5


# ============================================================
# 1. CARGA DE DATOS
# ============================================================
def load_features(path: str) -> pd.DataFrame:
    """Lee el Parquet consolidado de features."""
    logger.info(f"Cargando features desde: {path}")
    df = pd.read_parquet(path)
    n_users = df["userId"].nunique()
    n_items = df["movieId"].nunique()
    logger.info(f"  → {len(df):,} interacciones | {n_users:,} usuarios | {n_items:,} items")
    return df


# ============================================================
# 2. FILTRADO DE RUIDO (SIMPLE, NO ITERATIVO)
# ============================================================
def filter_noise(
    df: pd.DataFrame, min_user: int, min_item: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtrado simple de ruido:
    - Elimina usuarios con < min_user interacciones
    - Elimina items con < min_item interacciones

    No es iterativo: un solo pase es suficiente cuando los umbrales son bajos (≤5)
    y el dataset ya pasó por un filtrado previo en el Job 1.

    Returns:
        (df_clean, df_discarded)
    """
    logger.info(f"Filtrando ruido (usuarios≥{min_user}, items≥{min_item})...")
    initial_count = len(df)

    # Filtrar usuarios con pocas interacciones
    user_counts = df["userId"].value_counts()
    valid_users = user_counts[user_counts >= min_user].index
    df_clean = df[df["userId"].isin(valid_users)]
    removed_users = initial_count - len(df_clean)

    # Filtrar items con pocas interacciones
    item_counts = df_clean["movieId"].value_counts()
    valid_items = item_counts[item_counts >= min_item].index
    df_clean = df_clean[df_clean["movieId"].isin(valid_items)]
    removed_items = (initial_count - removed_users) - len(df_clean)

    # Interacciones descartadas (cold-start set)
    df_discarded = df[~df.index.isin(df_clean.index)]

    logger.info(f"  → Eliminados por usuarios: {removed_users:,}")
    logger.info(f"  → Eliminados por items: {removed_items:,}")
    logger.info(f"  → Clean: {len(df_clean):,} | Descartadas: {len(df_discarded):,}")
    logger.info(
        f"  → Usuarios: {df_clean['userId'].nunique():,} | Items: {df_clean['movieId'].nunique():,}"
    )

    return df_clean, df_discarded


# ============================================================
# 3. SPLIT TEMPORAL-ESTRATIFICADO
# ============================================================
def temporal_stratified_split(
    df: pd.DataFrame, config: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal proporcional por usuario (alineado con el notebook).
    Usa rank(pct=True) para asignar percentiles temporales por usuario.

    - Ordena interacciones de cada usuario cronológicamente
    - El 80% más antiguo → train
    - El siguiente 10% → val
    - El 10% más reciente → test
    """
    logger.info("Realizando split temporal-estratificado...")

    # Ordenar por usuario + timestamp ascendente (lo más viejo primero)
    df_sorted = df.sort_values(["userId", "timestamp"], ascending=[True, True]).copy()

    # Calcular percentil temporal de cada fila por usuario (0.0 a 1.0)
    df_sorted["pct_tiempo"] = df_sorted.groupby("userId")["timestamp"].rank(
        pct=True, method="first"
    )

    # Cortar según los porcentajes
    limite_val = config.train_ratio + config.val_ratio

    df_train = df_sorted[df_sorted["pct_tiempo"] <= config.train_ratio].copy()
    df_val = df_sorted[
        (df_sorted["pct_tiempo"] > config.train_ratio)
        & (df_sorted["pct_tiempo"] <= limite_val)
    ].copy()
    df_test = df_sorted[df_sorted["pct_tiempo"] > limite_val].copy()

    # Limpiar columna auxiliar
    df_train = df_train.drop(columns=["pct_tiempo"]).reset_index(drop=True)
    df_val = df_val.drop(columns=["pct_tiempo"]).reset_index(drop=True)
    df_test = df_test.drop(columns=["pct_tiempo"]).reset_index(drop=True)

    # Sanity check
    total = len(df_train) + len(df_val) + len(df_test)
    assert total == len(df_sorted), f"Pérdida de datos: {len(df_sorted)} → {total}"

    logger.info(f"  → Train: {len(df_train):,} ({len(df_train)/total*100:.1f}%)")
    logger.info(f"  → Val:   {len(df_val):,} ({len(df_val)/total*100:.1f}%)")
    logger.info(f"  → Test:  {len(df_test):,} ({len(df_test)/total*100:.1f}%)")

    return df_train, df_val, df_test


# ============================================================
# 4. ESCRITURA DE OUTPUTS
# ============================================================
def save_parquet(df: pd.DataFrame, output_dir: str, name: str) -> None:
    """Guarda un DataFrame como Parquet en un directorio."""
    dirpath = os.path.join(output_dir, name)
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, f"{name}.parquet")
    df.to_parquet(filepath, index=False)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info(f"  → {name}: {len(df):,} filas ({size_mb:.1f} MB)")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation: Filter + Temporal Splits")
    parser.add_argument(
        "--min-user-interactions", type=int, default=5,
        help="Mínimo de interacciones por usuario para incluirlo (default: 5)",
    )
    parser.add_argument(
        "--min-item-interactions", type=int, default=5,
        help="Mínimo de interacciones por item para incluirlo (default: 5)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.80,
        help="Proporción de datos para training (default: 0.80)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.10,
        help="Proporción de datos para validación (default: 0.10)",
    )
    args = parser.parse_args()

    config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=round(1.0 - args.train_ratio - args.val_ratio, 2),
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )

    inicio = time.time()
    logger.info("=" * 60)
    logger.info("Processing Job 3: Dataset Preparation (pandas)")
    logger.info(f"  Filter: users≥{config.min_user_interactions} | items≥{config.min_item_interactions}")
    logger.info(f"  Splits: {config.train_ratio}/{config.val_ratio}/{config.test_ratio}")
    logger.info("=" * 60)

    # Paths
    input_features = "/opt/ml/processing/input/features"
    output_base = "/opt/ml/processing/output"

    # 1. Cargar datos
    logger.info("\n[PASO 1/4] Cargando features...")
    df = load_features(input_features)

    # 2. Filtrado de ruido
    logger.info("\n[PASO 2/4] Filtrando ruido...")
    df_clean, df_coldstart = filter_noise(
        df,
        min_user=config.min_user_interactions,
        min_item=config.min_item_interactions,
    )

    # 3. Split temporal-estratificado
    logger.info("\n[PASO 3/4] Split temporal-estratificado...")
    df_train, df_val, df_test = temporal_stratified_split(df_clean, config)

    # 4. Escribir outputs
    logger.info("\n[PASO 4/4] Escribiendo outputs...")
    save_parquet(df_train, output_base, "train")
    save_parquet(df_val, output_base, "val")
    save_parquet(df_test, output_base, "test")
    save_parquet(df_coldstart, output_base, "cold-starts")

    # Resumen
    duracion = round(time.time() - inicio, 2)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Job completado en {duracion}s")
    logger.info(f"  Train:       {len(df_train):,}")
    logger.info(f"  Val:         {len(df_val):,}")
    logger.info(f"  Test:        {len(df_test):,}")
    logger.info(f"  Cold-starts: {len(df_coldstart):,}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
