"""
SageMaker Processing Job 2: Dataset Preparation (Filtering + Splits + Cold-Start)
==================================================================================
Lee el Feature Store offline, aplica filtrado de ruido (k-core), genera splits
temporales-estratificados, y construye un set de cold-start para evaluación.

Pipeline:
  1. K-Core Filtering: elimina usuarios con <20 interacciones e items con <10
     (iterativo hasta convergencia para eliminar ruido del modelo)
  2. Split Temporal-Estratificado: preserva cronología por usuario
     - train (80%) | val (10%) | test (10%)
  3. Cold-Start Set: películas con pocas interacciones

Outputs (→ Platinum bucket):
  - /platinum/train/           → Parquet (~80% interacciones)
  - /platinum/val/             → Parquet (~10%)
  - /platinum/test/            → Parquet (~10%)
  - /platinum/cold-starts/     → Parquet (películas con pocas interacciones)
  - /platinum/encoders/        → encoders.pkl (copia)
  - /platinum/embeddings/      → embeddings_catalog.pkl (copia)

Inputs:
  - /opt/ml/processing/input/features/    → Feature interactions (Parquet del Feature Store offline)
  - /opt/ml/processing/input/movies/      → obt_movies completo (Parquet de Silver, para cold-start)
  - /opt/ml/processing/input/encoders/    → encoders.pkl
  - /opt/ml/processing/input/embeddings/  → embeddings_catalog.pkl
"""

import argparse
import logging
import os
import shutil
import time

from pyspark.sql import SparkSession, functions as F, Window

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTES POR DEFECTO
# ============================================================
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

MIN_USER_INTERACTIONS = 20   # Usuarios con menos de esto → ruido
MIN_ITEM_INTERACTIONS = 10   # Items con menos de esto → ruido
KCORE_MAX_ITERATIONS = 5     # Máximo de iteraciones del k-core filtering


# ============================================================
# 1. INICIALIZACIÓN
# ============================================================
def get_spark_session() -> SparkSession:
    """Inicializa SparkSession."""
    spark = SparkSession.builder \
        .appName("HymmRec-DatasetPreparation") \
        .config("spark.sql.parquet.enableVectorizedReader", "true") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    logger.info(f"SparkSession inicializada: {spark.version}")
    return spark


# ============================================================
# 2. LECTURA DE DATOS
# ============================================================
def load_features(spark: SparkSession, path: str):
    """Lee las features procesadas (Feature Store offline export).
    
    Feature Store offline almacena en estructura particionada:
      .../data/year=YYYY/month=MM/day=DD/hour=HH/*.parquet
    Usamos recursiveFileLookup para leer todos los parquets sin conflictos de particiones.
    """
    logger.info(f"Cargando features desde: {path}")
    df = spark.read.option("recursiveFileLookup", "true").parquet(path)
    # Consolidar micro-archivos del Feature Store en pocas particiones en memoria
    # Esto evita el overhead de scheduling de cientos de micro-tasks
    n_partitions = max(4, df.rdd.getNumPartitions() // 50)
    df = df.coalesce(n_partitions)
    logger.info(f"  → Consolidado de {df.rdd.getNumPartitions()} particiones a {n_partitions}")

    # Filtrar registros marcados como eliminados por Feature Store (si existe la columna)
    if "is_deleted" in df.columns:
        df = df.filter(df["is_deleted"] == False).drop("is_deleted")

    # Eliminar columnas internas del Feature Store que no necesitamos
    cols_to_drop = [c for c in df.columns if c in (
        "write_time", "api_invocation_time", "is_deleted",
        "hymmrec_eventtime_sm_et_fn_trunc"
    )]
    df = df.drop(*cols_to_drop)
    count = df.count()
    n_users = df.select("userId").distinct().count()
    n_items = df.select("movieId").distinct().count()
    logger.info(f"  → {count:,} interacciones | {n_users:,} usuarios | {n_items:,} items")
    return df


# ============================================================
# 3. K-CORE FILTERING (ELIMINACIÓN DE RUIDO)
# ============================================================
def kcore_filter(df, min_user: int, min_item: int, max_iterations: int):
    """
    Filtrado iterativo k-core:
    - Elimina usuarios con < min_user interacciones
    - Elimina items con < min_item interacciones
    - Repite hasta convergencia (porque al quitar usuarios se pierden items y viceversa)

    Returns:
        (df_clean, df_discarded): Dataset limpio + interacciones descartadas (cold-start)
    """
    logger.info(f"Aplicando k-core filtering (min_user={min_user}, min_item={min_item}, max_iter={max_iterations})...")

    prev_count = df.count()
    logger.info(f"  → Inicio: {prev_count:,} interacciones")

    df_clean = df

    for iteration in range(1, max_iterations + 1):
        # Filtrar usuarios con pocas interacciones
        user_counts = df_clean.groupBy("userId").count().filter(F.col("count") >= min_user).select("userId")
        df_clean = df_clean.join(user_counts, on="userId", how="inner")

        # Filtrar items con pocas interacciones
        item_counts = df_clean.groupBy("movieId").count().filter(F.col("count") >= min_item).select("movieId")
        df_clean = df_clean.join(item_counts, on="movieId", how="inner")

        current_count = df_clean.count()
        removed = prev_count - current_count
        logger.info(f"  → Iteración {iteration}: {current_count:,} (-{removed:,} eliminadas)")

        # Convergencia: si no se eliminó nada, terminamos
        if current_count == prev_count:
            logger.info(f"  → Convergencia alcanzada en iteración {iteration}")
            break

        prev_count = current_count

    # Obtener las interacciones descartadas (lo que NO sobrevivió al filtrado)
    df_discarded = df.join(df_clean.select("userId", "movieId"), on=["userId", "movieId"], how="left_anti")

    n_users_clean = df_clean.select("userId").distinct().count()
    n_items_clean = df_clean.select("movieId").distinct().count()
    discarded_count = df_discarded.count()

    logger.info(f"  → Clean: {prev_count:,} interacciones | {n_users_clean:,} usuarios | {n_items_clean:,} items")
    logger.info(f"  → Descartadas (cold-start): {discarded_count:,} interacciones")

    return df_clean, df_discarded


# ============================================================
# 4. SPLIT TEMPORAL-ESTRATIFICADO
# ============================================================
def temporal_stratified_split(df):
    """
    Split temporal por usuario:
    - Ordena interacciones de cada usuario cronológicamente
    - El 80% más antiguo → train
    - El siguiente 10% → val
    - El 10% más reciente → test

    Simula predicción del futuro: "dado lo que vio antes, predice lo que verá después"
    """
    logger.info("Realizando split temporal-estratificado...")

    # Ranking temporal dentro de cada usuario
    user_window = Window.partitionBy("userId").orderBy("timestamp")
    count_window = Window.partitionBy("userId")

    df_ranked = df \
        .withColumn("row_num", F.row_number().over(user_window)) \
        .withColumn("total_per_user", F.count("*").over(count_window)) \
        .withColumn("position_ratio", F.col("row_num") / F.col("total_per_user"))

    # Asignar splits
    df_split = df_ranked.withColumn(
        "split",
        F.when(F.col("position_ratio") <= TRAIN_RATIO, "train")
         .when(F.col("position_ratio") <= TRAIN_RATIO + VAL_RATIO, "val")
         .otherwise("test")
    )

    # Separar y limpiar columnas auxiliares
    cols_to_drop = ["row_num", "total_per_user", "position_ratio", "split"]
    df_train = df_split.filter(F.col("split") == "train").drop(*cols_to_drop)
    df_val = df_split.filter(F.col("split") == "val").drop(*cols_to_drop)
    df_test = df_split.filter(F.col("split") == "test").drop(*cols_to_drop)

    train_count = df_train.count()
    val_count = df_val.count()
    test_count = df_test.count()
    total = train_count + val_count + test_count

    logger.info(f"  → Train: {train_count:,} ({train_count/total*100:.1f}%)")
    logger.info(f"  → Val:   {val_count:,} ({val_count/total*100:.1f}%)")
    logger.info(f"  → Test:  {test_count:,} ({test_count/total*100:.1f}%)")

    return df_train, df_val, df_test


# ============================================================
# 5. ESCRITURA DE OUTPUTS
# ============================================================
def write_spark_parquet(df, output_path: str, name: str):
    """Escribe Spark DataFrame como Parquet particionado eficientemente."""
    filepath = os.path.join(output_path, name)
    count = df.count()
    # ~1-2M rows por archivo para lectura eficiente en Training Job
    n_partitions = max(1, min(count // 1_500_000, 32))
    df.coalesce(max(1, n_partitions)).write.mode("overwrite").parquet(filepath)
    logger.info(f"  → {name}: {count:,} filas → {filepath}")


def copy_artifact(src_dir: str, dst_dir: str, filename: str):
    """Copia un artefacto al output."""
    os.makedirs(dst_dir, exist_ok=True)
    src = os.path.join(src_dir, filename)
    dst = os.path.join(dst_dir, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        logger.info(f"  → {filename} ({size_mb:.1f} MB) → {dst_dir}")
    else:
        logger.warning(f"  → Artefacto no encontrado: {src}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-user-interactions", type=int, default=MIN_USER_INTERACTIONS,
                        help="Mínimo de interacciones por usuario (k-core)")
    parser.add_argument("--min-item-interactions", type=int, default=MIN_ITEM_INTERACTIONS,
                        help="Mínimo de interacciones por item (k-core)")
    parser.add_argument("--kcore-iterations", type=int, default=KCORE_MAX_ITERATIONS,
                        help="Máximo de iteraciones del k-core filtering")
    args = parser.parse_args()

    inicio = time.time()
    logger.info("=" * 60)
    logger.info("Processing Job 2: Dataset Preparation")
    logger.info(f"  K-core: users≥{args.min_user_interactions} | items≥{args.min_item_interactions}")
    logger.info(f"  Splits: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    logger.info("=" * 60)

    # Paths
    input_features = "file:///opt/ml/processing/input/features"
    input_encoders = "/opt/ml/processing/input/encoders"
    input_embeddings = "/opt/ml/processing/input/embeddings"
    output_platinum = "/opt/ml/processing/output/platinum"

    # 1. Inicializar Spark
    logger.info("\n[PASO 1/5] Inicializando Spark...")
    spark = get_spark_session()

    # 2. Cargar datos
    logger.info("\n[PASO 2/5] Cargando datos...")
    df_features = load_features(spark, input_features)

    # 3. K-Core Filtering (eliminación de ruido → cold-start como residuo)
    logger.info("\n[PASO 3/5] Filtrando ruido (k-core)...")
    df_clean, df_coldstart = kcore_filter(
        df_features,
        min_user=args.min_user_interactions,
        min_item=args.min_item_interactions,
        max_iterations=args.kcore_iterations,
    )

    # 4. Split temporal-estratificado sobre el dataset limpio
    logger.info("\n[PASO 4/5] Split temporal-estratificado...")
    df_train, df_val, df_test = temporal_stratified_split(df_clean)

    # 5. Escribir outputs en Platinum
    logger.info("\n[PASO 5/5] Escribiendo datasets en Platinum...")

    # Datasets de entrenamiento/evaluación
    write_spark_parquet(df_train, output_platinum, "train")
    write_spark_parquet(df_val, output_platinum, "val")
    write_spark_parquet(df_test, output_platinum, "test")

    # Cold-starts (interacciones descartadas por k-core, para pruebas en inferencia)
    write_spark_parquet(df_coldstart, output_platinum, "cold-starts")

    # Copiar artefactos al Platinum (Training Job necesita todo junto)
    #copy_artifact(input_encoders, os.path.join(output_platinum, "encoders"), "encoders.pkl")
    #copy_artifact(input_embeddings, os.path.join(output_platinum, "embeddings"), "embeddings_catalog.pkl")

    # Cleanup
    spark.stop()

    # Resumen
    duracion = round(time.time() - inicio, 2)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing Job 2 completado en {duracion}s")
    logger.info(f"  Train:       {df_train.count():,}")
    logger.info(f"  Val:         {df_val.count():,}")
    logger.info(f"  Test:        {df_test.count():,}")
    logger.info(f"  Cold-starts: {df_coldstart.count():,} (descartadas por k-core)")
    logger.info(f"  Output: {output_platinum}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
