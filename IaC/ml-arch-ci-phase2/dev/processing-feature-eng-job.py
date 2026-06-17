"""
SageMaker Processing Job 1: Feature Engineering + Feature Store Ingestion
=========================================================================
Pipeline de preparación de features para el sistema recomendador híbrido.

Responsabilidades:
  1. Lee tablas Iceberg de Silver (cleansed_ratings + cleansed_movies) con PySpark
  2. Feature Engineering con Spark: merge géneros, rating scaling
  3. Feature Engineering con sklearn en driver: LabelEncoder, MultiLabelBinarizer
  4. Ingesta features procesadas en SageMaker Feature Store (offline, Iceberg-backed)
  5. Guarda artefactos: encoders.pkl

NOTA: La generación de embeddings multimodales con Bedrock Nova se realiza
en un Processing Job separado (processing-embeddings-job.py) para optimizar
costos y tiempo de ejecución.

Inputs (desde S3 Silver):
  - /opt/ml/processing/input/ratings/     → Parquet (Iceberg export de cleansed_ratings)
  - /opt/ml/processing/input/movies/      → Parquet (Iceberg export de cleansed_movies)

Outputs (hacia S3 Gold):
  - /opt/ml/processing/output/encoders/           → encoders.pkl
  - /opt/ml/processing/output/feature_interactions/ → feature_interactions.parquet (fallback)

Feature Store Output:
  - Feature Group: hymmrec-interactions-sm-fg (offline store, Iceberg en S3)

Execution:
  Usar PySparkProcessor con ml.m5.xlarge para las transformaciones Spark.
  Tiempo estimado: ~5-10 min para 100K interacciones.
"""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install","scikit-learn==1.3.2", "numpy<2", "--quiet"])


import argparse
import logging
import os
import pickle
import time
from typing import Dict

import boto3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTES
# ============================================================
FEATURE_GROUP_NAME = "hymmrec-interactions-sm-fg"


# ============================================================
# 1. INICIALIZACIÓN SPARK
# ============================================================
def get_spark_session() -> SparkSession:
    """Inicializa SparkSession para el Processing Job."""
    spark = SparkSession.builder \
        .appName("HymmRec-FeatureEngineering") \
        .config("spark.sql.parquet.enableVectorizedReader", "true") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .getOrCreate()

    # Habilitar lectura recursiva para encontrar parquets en subdirectorios
    # (Iceberg guarda los .parquet dentro de subcarpetas data/ o particiones)
    spark.conf.set("spark.sql.sources.partitionDiscovery.enabled", "false")
    spark.sparkContext._jsc.hadoopConfiguration().set("mapreduce.input.fileinputformat.input.dir.recursive", "true")

    logger.info(f"SparkSession inicializada: {spark.version}")
    return spark


# ============================================================
# 2. LECTURA DE DATOS (SPARK)
# ============================================================
def load_ratings_spark(spark: SparkSession, path: str):
    """Lee ratings desde Parquet (Iceberg materializado).
    
    Iceberg almacena los parquets en subcarpeta 'data/' o con particiones.
    Intentamos leer el path directo; si falla, buscamos en subdirectorios.
    """
    logger.info(f"Cargando ratings desde: {path}")

    # Intentar leer recursivamente con wildcard para cubrir estructura Iceberg
    try:
        df = spark.read.parquet(path)
        if df.schema.fields:  # Schema inferido OK → hay archivos
            count = df.count()
            n_users = df.select("userId").distinct().count()
            n_items = df.select("movieId").distinct().count()
            logger.info(f"  → {count:,} interacciones | {n_users:,} usuarios | {n_items:,} películas")
            return df
    except Exception as e:
        logger.warning(f"  → Lectura directa falló: {e}")

    # Fallback: buscar parquets en subdirectorios (estructura Iceberg: /data/*.parquet)
    recursive_path = path.rstrip("/") + "/data"
    logger.info(f"  → Intentando path Iceberg: {recursive_path}")
    df = spark.read.parquet(recursive_path)
    count = df.count()
    n_users = df.select("userId").distinct().count()
    n_items = df.select("movieId").distinct().count()
    logger.info(f"  → {count:,} interacciones | {n_users:,} usuarios | {n_items:,} películas")
    return df


def load_movies_spark(spark: SparkSession, path: str):
    """Lee movies desde Parquet (Iceberg materializado).
    
    Misma lógica que ratings: intenta directo, fallback a subcarpeta data/.
    """
    logger.info(f"Cargando movies desde: {path}")

    try:
        df = spark.read.parquet(path)
        if df.schema.fields:
            count = df.count()
            logger.info(f"  → {count:,} películas")
            return df
    except Exception as e:
        logger.warning(f"  → Lectura directa falló: {e}")

    recursive_path = path.rstrip("/") + "/data"
    logger.info(f"  → Intentando path Iceberg: {recursive_path}")
    df = spark.read.parquet(recursive_path)
    count = df.count()
    logger.info(f"  → {count:,} películas")
    return df


# ============================================================
# 3. FEATURE ENGINEERING (SPARK + SKLEARN EN DRIVER)
# ============================================================
def flatten_array_to_pipe_separated(col_value) -> str:
    """
    UDF helper: convierte array<string> (Iceberg) a string pipe-separated.

    En Iceberg las columnas como 'generos' son array<string>:
      ['Action', 'Drama', 'Thriller'] → "Action|Drama|Thriller"

    En el notebook de Colab venían como strings ya:
      "Action|Drama|Thriller" (así que no necesitaba conversión)

    Aquí manejamos ambos formatos por seguridad.
    """
    if col_value is None:
        return "Desconocido"
    if isinstance(col_value, list):
        elementos = [str(x).strip() for x in col_value if x and str(x).strip()]
        return "|".join(elementos) if elementos else "Desconocido"
    # Si por alguna razón viene como string, lo retornamos tal cual
    return str(col_value) if str(col_value).strip() else "Desconocido"


flatten_array_udf = F.udf(flatten_array_to_pipe_separated, StringType())


def build_feature_interactions(spark: SparkSession, df_ratings, df_movies):
    """
    Construye las features de interacciones: ratings + géneros (merge).

    Lógica equivalente al notebook:
      1. Merge ratings con géneros (pipe-separated) vía left join
      2. Rating scaling dinámico: (rating - min) / (max - min) → [0, 1]
      3. Label Encoding: userId → userId_idx, movieId → movieId_idx
      4. Multi-Hot Encoding: géneros → vector binario de N dimensiones

    NOTA: NO se aplica k-core filtering aquí. Eso lo hace el Processing Job 2.

    Returns:
        (df_features_pd: pd.DataFrame, encoders: dict)
    """
    logger.info("Construyendo feature_interactions...")

    # --- SPARK: Merge ratings + géneros ---
    # Extraemos solo movieId + generos (equivalente a df_generos = df_movies[['movieId', 'generos']])
    df_generos = df_movies.select("movieId", "generos")

    # Aplanar array<string> a string pipe-separated
    # En Iceberg: ['Action', 'Drama'] → "Action|Drama"
    df_generos = df_generos.withColumn("generos_str", flatten_array_udf(F.col("generos")))
    df_generos = df_generos.select("movieId", F.col("generos_str").alias("generos"))

    # Left join (equivalente al pd.merge del notebook)
    df_merged = df_ratings.join(df_generos, on="movieId", how="left")
    df_merged = df_merged.fillna({"generos": "Desconocido"})

    # Cast tipos para consistencia
    df_merged = df_merged.select(
        F.col("userId").cast(IntegerType()),
        F.col("movieId").cast(IntegerType()),
        F.col("rating").cast(FloatType()),
        F.col("timestamp").cast("long"),
        F.col("generos").cast(StringType()),
    )

    count = df_merged.count()
    logger.info(f"  → Merge completado: {count:,} filas")

    # --- DRIVER (Pandas): Rating Scaling + Label Encoding + Multi-Hot ---
    logger.info("  → Collecting to driver para encoding (sklearn)...")
    df_pd = df_merged.toPandas()

    # Rating Scaling dinámico [min, max] → [0, 1]
    # Equivalente al notebook: (rating - min_rating) / (max_rating - min_rating)
    min_rating = df_pd["rating"].min()
    max_rating = df_pd["rating"].max()
    df_pd["rating_scaled"] = (df_pd["rating"] - min_rating) / (max_rating - min_rating)
    logger.info(f"  → Rating scaling: [{min_rating}, {max_rating}] → [0, 1]")

    # Label Encoding (equivalente al notebook: le_user, le_item)
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    df_pd["userId_idx"] = le_user.fit_transform(df_pd["userId"].values)
    df_pd["movieId_idx"] = le_item.fit_transform(df_pd["movieId"].values)
    logger.info(f"  → Label Encoding: {len(le_user.classes_):,} usuarios, {len(le_item.classes_):,} ítems")

    # Multi-Hot Encoding (equivalente al notebook)
    # El notebook hace: si es string → split('|'), si es lista → usar directamente
    # Aquí siempre viene como string pipe-separated (ya lo convertimos en Spark)
    df_pd["genres_list"] = df_pd["generos"].apply(
        lambda x: [g.strip() for g in x.split("|") if g.strip()] if isinstance(x, str) else ["Desconocido"]
    )

    mlb = MultiLabelBinarizer()
    genres_matrix = mlb.fit_transform(df_pd["genres_list"])
    df_pd["genres_multihot"] = [row.tolist() for row in genres_matrix]
    logger.info(f"  → Multi-Hot Encoding: {len(mlb.classes_)} géneros únicos: {list(mlb.classes_)}")

    encoders = {"le_user": le_user, "le_item": le_item, "mlb": mlb}
    return df_pd, encoders


# ============================================================
# 4. INGESTA EN FEATURE STORE
# ============================================================
def ingest_to_feature_store(df: pd.DataFrame, feature_group_name: str, region: str):
    """
    Ingesta el DataFrame de features en un SageMaker Feature Group (offline store).
    El Feature Group debe estar pre-creado con el schema correcto (Terraform).

    Si el Feature Group no existe o falla, escribe como Parquet fallback.
    """
    import sagemaker
    from sagemaker.feature_store.feature_group import FeatureGroup

    try:
        sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
        feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)

        # Preparar DataFrame para ingestion
        # Feature Store requiere: event_time como string ISO, record_id como string
        df_ingest = df[["userId", "movieId", "rating", "timestamp", "generos",
                        "rating_scaled", "userId_idx", "movieId_idx"]].copy()

        # Event time (requerido por Feature Store)
        df_ingest["hymmrec_eventtime_sm_et_fn"] = pd.to_datetime(
            df_ingest["timestamp"], unit="s"
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Record identifier (compuesto: userId_movieId)
        df_ingest["hymmrec_recordid_sm_ri_fn"] = (
            df_ingest["userId"].astype(str) + "_" + df_ingest["movieId"].astype(str)
        )

        # Tipos compatibles con Feature Store
        df_ingest["userId"] = df_ingest["userId"].astype("int64")
        df_ingest["movieId"] = df_ingest["movieId"].astype("int64")
        df_ingest["userId_idx"] = df_ingest["userId_idx"].astype("int64")
        df_ingest["movieId_idx"] = df_ingest["movieId_idx"].astype("int64")
        df_ingest["rating"] = df_ingest["rating"].astype("float64")
        df_ingest["rating_scaled"] = df_ingest["rating_scaled"].astype("float64")
        df_ingest["timestamp"] = df_ingest["timestamp"].astype("int64")

        logger.info(f"Ingesting {len(df_ingest):,} records into Feature Group: {feature_group_name}")
        feature_group.ingest(data_frame=df_ingest, max_workers=4, wait=True)
        logger.info("  → Ingestion completada exitosamente.")

    except Exception as e:
        logger.warning(f"Feature Store ingestion failed ({e}). Fallback a Parquet en output.")
        _save_fallback_parquet(df)


def _save_fallback_parquet(df: pd.DataFrame):
    """
    Guarda features como Parquet consolidado.
    Se ejecuta SIEMPRE (no es un fallback) para que Job 3 (splits)
    lea un archivo optimizado en vez de los miles de micro-archivos
    que genera Feature Store offline.
    """
    fallback_path = "/opt/ml/processing/output/feature_interactions"
    os.makedirs(fallback_path, exist_ok=True)
    filepath = os.path.join(fallback_path, "feature_interactions.parquet")
    df_save = df[["userId", "movieId", "rating", "timestamp", "generos",
                  "rating_scaled", "userId_idx", "movieId_idx", "genres_multihot"]].copy()
    df_save.to_parquet(filepath, index=False)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info(f"  → Parquet consolidado guardado: {filepath} ({size_mb:.1f} MB)")


# ============================================================
# 5. ESCRITURA DE ARTEFACTOS
# ============================================================
def save_encoders(encoders: dict, output_path: str):
    """Guarda los encoders como pickle (le_user, le_item, mlb)."""
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, "encoders.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(encoders, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Encoders guardados: {filepath}")
    logger.info(f"  le_user: {len(encoders['le_user'].classes_):,} clases")
    logger.info(f"  le_item: {len(encoders['le_item'].classes_):,} clases")
    logger.info(f"  mlb: {len(encoders['mlb'].classes_)} géneros → {list(encoders['mlb'].classes_)}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--feature-group-name", type=str, default=FEATURE_GROUP_NAME)
    args = parser.parse_args()

    inicio = time.time()
    logger.info("=" * 60)
    logger.info("Processing Job 1: Feature Engineering")
    logger.info(f"  Region: {args.region}")
    logger.info(f"  Feature Group: {args.feature_group_name}")
    logger.info("=" * 60)

    # Paths (file:// prefix para que Spark lea del filesystem local, no HDFS)
    input_ratings = "file:///opt/ml/processing/input/ratings"
    input_movies = "file:///opt/ml/processing/input/movies"
    output_encoders = "/opt/ml/processing/output/encoders"

    # 1. Inicializar Spark
    logger.info("\n[PASO 1/4] Inicializando Spark...")
    spark = get_spark_session()

    # 2. Cargar datos de Silver
    logger.info("\n[PASO 2/4] Cargando datos desde Silver Layer...")
    df_ratings_spark = load_ratings_spark(spark, input_ratings)
    df_movies_spark = load_movies_spark(spark, input_movies)

    # 3. Feature Engineering (Spark + sklearn)
    logger.info("\n[PASO 3/4] Feature Engineering...")
    df_features_pd, encoders = build_feature_interactions(spark, df_ratings_spark, df_movies_spark)

    # 4. Ingestar en Feature Store
    logger.info("\n[PASO 4/4] Ingesting en Feature Store (offline)...")
    ingest_to_feature_store(df_features_pd, args.feature_group_name, args.region)

    # Siempre guardar Parquet consolidado (para que Job 3 lo use en vez de los micro-archivos del Feature Store)
    _save_fallback_parquet(df_features_pd)

    # Guardar encoders (siempre, independiente del Feature Store)
    save_encoders(encoders, output_encoders)

    # Cleanup Spark
    spark.stop()

    # Resumen
    duracion = round(time.time() - inicio, 2)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing Job 1 completado en {duracion}s")
    logger.info(f"  Features: {len(df_features_pd):,} interacciones")
    logger.info(f"  Usuarios: {len(encoders['le_user'].classes_):,}")
    logger.info(f"  Ítems: {len(encoders['le_item'].classes_):,}")
    logger.info(f"  Géneros: {len(encoders['mlb'].classes_)}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
