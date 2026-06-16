"""
SageMaker Processing Job 1: Feature Engineering + Embeddings Generation
=========================================================================
Pipeline de preparación de features para el sistema recomendador híbrido.

Responsabilidades:
  1. Lee tablas Iceberg de Silver (cleansed_ratings + cleansed_movies) con PySpark
  2. Feature Engineering con Spark: merge géneros, label encoding, multi-hot, rating scaling
  3. Ingesta features procesadas en SageMaker Feature Store (offline, Iceberg-backed)
  4. Genera catálogo de embeddings multimodales (texto + imagen) con Amazon Bedrock Nova
  5. Guarda artefactos: embeddings_catalog.pkl + encoders.pkl

Inputs (desde S3 Silver):
  - /opt/ml/processing/input/ratings/     → Parquet (Iceberg export de cleansed_ratings)
  - /opt/ml/processing/input/movies/      → Parquet (Iceberg export de cleansed_movies)
  - /opt/ml/processing/input/posters/     → Imágenes JPG ({movieId}.jpg)

Outputs (hacia S3 Gold):
  - /opt/ml/processing/output/embeddings/ → embeddings_catalog.pkl
  - /opt/ml/processing/output/encoders/   → encoders.pkl

Feature Store Output:
  - Feature Group: hymmrec-feature-interactions (offline store, Iceberg en S3)

Execution:
  Usar PySparkProcessor con ml.m5.xlarge (2-3 workers) para las transformaciones Spark.
  La generación de embeddings se ejecuta en el driver (I/O bound con Bedrock API).
"""
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "pillow", "boto3"])
import argparse
import ast
import base64
import json
import logging
import os
import pickle
import sys
import time
from io import BytesIO
from typing import Any, Dict

import boto3
import numpy as np
import pandas as pd
from PIL import Image
from botocore.exceptions import ClientError
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType, ArrayType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTES
# ============================================================
BEDROCK_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
BEDROCK_EMBEDDING_DIM = 1024
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
        .getOrCreate()
    logger.info(f"SparkSession inicializada: {spark.version}")
    return spark


# ============================================================
# 2. LECTURA DE DATOS (SPARK)
# ============================================================
def load_ratings_spark(spark: SparkSession, path: str):
    """Lee ratings desde Parquet (Iceberg materializado)."""
    logger.info(f"Cargando ratings desde: {path}")
    df = spark.read.parquet(path)
    count = df.count()
    n_users = df.select("userId").distinct().count()
    n_items = df.select("movieId").distinct().count()
    logger.info(f"  → {count:,} interacciones | {n_users:,} usuarios | {n_items:,} películas")
    return df


def load_movies_spark(spark: SparkSession, path: str):
    """Lee movies desde Parquet (Iceberg materializado)."""
    logger.info(f"Cargando movies desde: {path}")
    df = spark.read.parquet(path)
    count = df.count()
    logger.info(f"  → {count:,} películas")
    return df


# ============================================================
# 3. FEATURE ENGINEERING (SPARK + SKLEARN EN DRIVER)
# ============================================================
def flatten_array_column(col_value) -> str:
    """UDF helper: convierte array<string> a string pipe-separated."""
    if col_value is None:
        return "Desconocido"
    if isinstance(col_value, list):
        elementos = [str(x).strip() for x in col_value if x and str(x).strip()]
        return "|".join(elementos) if elementos else "Desconocido"
    return str(col_value)


flatten_array_udf = F.udf(flatten_array_column, StringType())


def build_feature_interactions(spark: SparkSession, df_ratings, df_movies):
    """
    Construye las features de interacciones con Spark + scikit-learn.

    Spark maneja: merge, casting, UDF para géneros
    Driver (pandas): Label Encoding, Multi-Hot Encoding (scikit-learn)

    Returns:
        (df_features_pd: pd.DataFrame, encoders: dict)
    """
    logger.info("Construyendo feature_interactions...")

    # --- SPARK: Merge ratings + géneros ---
    df_generos = df_movies.select("movieId", "generos")

    # Aplanar array<string> a string pipe-separated
    df_generos = df_generos.withColumn("generos_str", flatten_array_udf(F.col("generos")))
    df_generos = df_generos.select("movieId", F.col("generos_str").alias("generos"))

    # Left join
    df_merged = df_ratings.join(df_generos, on="movieId", how="left")
    df_merged = df_merged.fillna({"generos": "Desconocido"})

    # Rating escalado [1,5] → [0,1]
    df_merged = df_merged.withColumn("rating_scaled", (F.col("rating") - 1.0) / 4.0)

    # Cast tipos
    df_merged = df_merged.select(
        F.col("userId").cast(IntegerType()),
        F.col("movieId").cast(IntegerType()),
        F.col("rating").cast(FloatType()),
        F.col("timestamp").cast("long"),
        F.col("generos").cast(StringType()),
        F.col("rating_scaled").cast(FloatType()),
    )

    count = df_merged.count()
    logger.info(f"  → Merge + scaling completado: {count:,} filas")

    # --- DRIVER (Pandas): Label Encoding + Multi-Hot ---
    logger.info("  → Collecting to driver para encoding (Label + Multi-Hot)...")
    df_pd = df_merged.toPandas()

    # Label Encoding
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    df_pd["userId_idx"] = le_user.fit_transform(df_pd["userId"].values)
    df_pd["movieId_idx"] = le_item.fit_transform(df_pd["movieId"].values)
    logger.info(f"  → Label Encoding: {len(le_user.classes_):,} usuarios, {len(le_item.classes_):,} ítems")

    # Multi-Hot Encoding
    df_pd["genres_list"] = df_pd["generos"].apply(lambda x: [g.strip() for g in x.split("|") if g.strip()])
    mlb = MultiLabelBinarizer()
    genres_matrix = mlb.fit_transform(df_pd["genres_list"])
    df_pd["genres_multihot"] = [row.tolist() for row in genres_matrix]
    logger.info(f"  → Multi-Hot: {len(mlb.classes_)} géneros únicos")

    encoders = {"le_user": le_user, "le_item": le_item, "mlb": mlb}
    return df_pd, encoders


# ============================================================
# 4. INGESTA EN FEATURE STORE
# ============================================================
def ingest_to_feature_store(df: pd.DataFrame, feature_group_name: str, region: str):
    """
    Ingesta el DataFrame de features en un SageMaker Feature Group (offline store).
    El Feature Group debe estar pre-creado con el schema correcto.

    Si el Feature Group no existe, escribe como Parquet fallback.
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
        df_ingest["hymmrec_eventtime_sm_et_fn"] = pd.to_datetime(df_ingest["timestamp"], unit="s").dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Record identifier (compuesto)
        df_ingest["hymmrec_recordid_sm_ri_fn"] = df_ingest["userId"].astype(str) + "_" + df_ingest["movieId"].astype(str)

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
        logger.info(f"  → Ingestion completada exitosamente.")

    except Exception as e:
        logger.warning(f"Feature Store ingestion failed ({e}). Fallback a Parquet en output.")
        # Fallback: guardar como Parquet para que el Job 2 pueda leerlo
        fallback_path = "/opt/ml/processing/output/feature_interactions"
        os.makedirs(fallback_path, exist_ok=True)
        filepath = os.path.join(fallback_path, "feature_interactions.parquet")
        df_save = df[["userId", "movieId", "rating", "timestamp", "generos",
                      "rating_scaled", "userId_idx", "movieId_idx", "genres_multihot"]].copy()
        df_save.to_parquet(filepath, index=False)
        logger.info(f"  → Fallback Parquet guardado: {filepath}")


# ============================================================
# 5. GENERACIÓN DE EMBEDDINGS MULTIMODALES (BEDROCK NOVA)
# ============================================================
def limpiar_lista_a_string(texto) -> str:
    """Convierte array/lista a string separado por comas."""
    if texto is None:
        return ""
    if isinstance(texto, (list, tuple, np.ndarray)):
        return ", ".join([str(item) for item in texto if pd.notna(item)])
    if pd.isna(texto):
        return ""
    if isinstance(texto, str):
        if texto.lower() == "nan":
            return ""
        try:
            lista_real = ast.literal_eval(texto)
            if isinstance(lista_real, list):
                return ", ".join([str(item) for item in lista_real])
        except (ValueError, SyntaxError):
            pass
        return texto.replace("[", "").replace("]", "")
    return str(texto)


def crear_sinopsis_semantica(row) -> str:
    """Crea la super-sinopsis para el embedding de texto."""
    datos = {
        "title": str(row.get("titulo", "Desconocido")),
        "tagline": str(row.get("frase_promocional", "")),
        "director": str(row.get("director", "Desconocido")),
        "cast": limpiar_lista_a_string(row.get("actores", "")),
        "keywords": limpiar_lista_a_string(row.get("palabras_clave", "")),
        "sinopsis": str(row.get("sinopsis", "Sin descripción.")),
    }
    partes = []
    for clave, valor in datos.items():
        if valor.strip() and valor.lower() != "nan":
            partes.append(f"{clave}: {valor}")
    return ". ".join(partes)


def generate_embeddings_catalog(
    df_movies_pd: pd.DataFrame,
    posters_path: str,
    region: str,
    batch_log_interval: int = 500,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Genera embeddings multimodales (texto + imagen) para cada película.
    Ejecuta en el driver (I/O bound con Bedrock API).
    """
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)

    # Preparar sinopsis semántica
    columnas_array = ["actores", "palabras_clave", "generos"]
    df_prep = df_movies_pd.copy()
    for col in columnas_array:
        if col in df_prep.columns:
            df_prep[col] = df_prep[col].apply(limpiar_lista_a_string)
    df_prep["sinopsis_semantica"] = df_prep.apply(crear_sinopsis_semantica, axis=1)

    catalog = {}
    total = len(df_prep)
    errores = 0

    logger.info(f"Generando embeddings para {total:,} películas con Bedrock Nova...")

    for idx, (_, row) in enumerate(df_prep.iterrows()):
        movie_id = int(row["movieId"])
        sinopsis = row["sinopsis_semantica"]
        img_path = os.path.join(posters_path, f"{movie_id}.jpg")

        try:
            # Embedding de texto
            payload_txt = {
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_INDEX",
                    "embeddingDimension": BEDROCK_EMBEDDING_DIM,
                    "text": {"truncationMode": "END", "value": sinopsis},
                },
            }
            resp_txt = bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID, contentType="application/json",
                accept="application/json", body=json.dumps(payload_txt),
            )
            v_txt = np.array(
                json.loads(resp_txt["body"].read())["embeddings"][0]["embedding"],
                dtype=np.float32,
            )

            # Embedding de imagen
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=90)
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                payload_img = {
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": {
                        "embeddingPurpose": "GENERIC_INDEX",
                        "embeddingDimension": BEDROCK_EMBEDDING_DIM,
                        "image": {"format": "jpeg", "source": {"bytes": img_b64}},
                    },
                }
                resp_img = bedrock_client.invoke_model(
                    modelId=BEDROCK_MODEL_ID, contentType="application/json",
                    accept="application/json", body=json.dumps(payload_img),
                )
                v_img = np.array(
                    json.loads(resp_img["body"].read())["embeddings"][0]["embedding"],
                    dtype=np.float32,
                )
            else:
                v_img = np.zeros(BEDROCK_EMBEDDING_DIM, dtype=np.float32)

            catalog[movie_id] = {"text_emb": v_txt, "img_emb": v_img}

        except ClientError as e:
            errores += 1
            if errores <= 10:
                logger.error(f"Bedrock error movieId={movie_id}: {e.response['Error']['Code']}")
        except Exception as e:
            errores += 1
            if errores <= 10:
                logger.error(f"Error movieId={movie_id}: {e}")

        if (idx + 1) % batch_log_interval == 0:
            logger.info(f"  Progreso: {idx + 1:,}/{total:,} | OK: {len(catalog):,} | Errores: {errores}")

    logger.info(f"Embeddings completados: {len(catalog):,}/{total:,} | Errores: {errores}")
    return catalog


# ============================================================
# 6. ESCRITURA DE ARTEFACTOS
# ============================================================
def save_embeddings(catalog: dict, output_path: str):
    """Guarda el catálogo de embeddings como pickle."""
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, "embeddings_catalog.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(catalog, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info(f"Embeddings guardados: {filepath} ({len(catalog):,} películas, {size_mb:.1f} MB)")


def save_encoders(encoders: dict, output_path: str):
    """Guarda los encoders como pickle."""
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, "encoders.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(encoders, f)
    logger.info(f"Encoders guardados: {filepath}")
    logger.info(f"  le_user: {len(encoders['le_user'].classes_):,} clases")
    logger.info(f"  le_item: {len(encoders['le_item'].classes_):,} clases")
    logger.info(f"  mlb: {len(encoders['mlb'].classes_)} géneros")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--feature-group-name", type=str, default=FEATURE_GROUP_NAME)
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding generation (for testing)")
    args = parser.parse_args()

    inicio = time.time()
    logger.info("=" * 60)
    logger.info("Processing Job 1: Feature Engineering + Embeddings")
    logger.info(f"Region: {args.region}")
    logger.info(f"Feature Group: {args.feature_group_name}")
    logger.info("=" * 60)

    # Paths
    input_ratings = "/opt/ml/processing/input/ratings"
    input_movies = "/opt/ml/processing/input/movies"
    input_posters = "/opt/ml/processing/input/posters"
    output_embeddings = "/opt/ml/processing/output/embeddings"
    output_encoders = "/opt/ml/processing/output/encoders"

    # 1. Inicializar Spark
    logger.info("\n[PASO 1/5] Inicializando Spark...")
    spark = get_spark_session()

    # 2. Cargar datos de Silver
    logger.info("\n[PASO 2/5] Cargando datos desde Silver Layer...")
    df_ratings_spark = load_ratings_spark(spark, input_ratings)
    df_movies_spark = load_movies_spark(spark, input_movies)

    # 3. Feature Engineering (Spark + sklearn)
    logger.info("\n[PASO 3/5] Feature Engineering...")
    df_features_pd, encoders = build_feature_interactions(spark, df_ratings_spark, df_movies_spark)

    # 4. Ingestar en Feature Store
    logger.info("\n[PASO 4/5] Ingesting en Feature Store (offline)...")
    ingest_to_feature_store(df_features_pd, args.feature_group_name, args.region)

    # 5. Generar embeddings multimodales
    if not args.skip_embeddings:
        logger.info("\n[PASO 5/5] Generando embeddings multimodales (Bedrock Nova)...")
        # Collect movies a pandas para el driver
        df_movies_pd = df_movies_spark.toPandas()
        embeddings_catalog = generate_embeddings_catalog(df_movies_pd, input_posters, args.region)
        save_embeddings(embeddings_catalog, output_embeddings)
    else:
        logger.info("\n[PASO 5/5] Embeddings SKIPPED (--skip-embeddings flag)")

    # Guardar encoders (siempre)
    save_encoders(encoders, output_encoders)

    # Cleanup Spark
    spark.stop()

    # Resumen
    duracion = round(time.time() - inicio, 2)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing Job 1 completado en {duracion}s")
    logger.info(f"  Features: {len(df_features_pd):,} interacciones")
    if not args.skip_embeddings:
        logger.info(f"  Embeddings: {len(embeddings_catalog):,} películas")
    logger.info(f"  Encoders: users={len(encoders['le_user'].classes_):,} | items={len(encoders['le_item'].classes_):,} | genres={len(encoders['mlb'].classes_)}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
