"""
AWS Glue PySpark Job: Create OBT Silver Tables (cleansed_ratings & cleansed_movies)
====================================================================================
Construye las tablas limpias de la capa Silver para el sistema recomendador:

  1. cleansed_ratings (obt_ratings):
     Intersección de ratings (MovieLens) + metadata TMDB (texto) + posters TMDB (visual).
     Solo incluye interacciones que tengan las 3 fuentes de datos disponibles.
     Columnas: userId, movieId, rating, timestamp

  2. cleansed_movies (obt_movies):
     Catálogo de películas filtrado por los movieIds presentes en cleansed_ratings.
     Incluye metadata textual completa + columna sinopsis enriquecida para embeddings.
     Columnas: movieId, titulo, sinopsis, generos, director, actores, palabras_clave, ...

Database destino: obt_movie_affinity (pre-creada en Glue Catalog via Terraform)
Formato de salida: Parquet

Estrategia de lectura:
  - Intenta leer desde tablas del Glue Catalog (ml_bronze, tmdb_bronze)
  - Si la tabla o DB no existe, hace fallback a lectura directa desde S3 path

Argumentos Glue (--key value):
  - JOB_NAME
  - obt_config_parameter:   nombre del parámetro SSM con paths S3
  - source_ml_database:     base de datos bronze de MovieLens (ej: ml_bronze)
  - source_ml_ratings_table: tabla de ratings en ml_bronze
  - source_ml_links_table:  tabla de links en ml_bronze
  - source_tmdb_database:   base de datos bronze de TMDB (ej: tmdb_bronze)
  - source_tmdb_movies_table: tabla de movies en tmdb_bronze
  - target_database:        base de datos destino Silver (ej: obt_movie_affinity)
  - target_ratings_table:   nombre tabla ratings destino (ej: cleansed_ratings)
  - target_movies_table:    nombre tabla movies destino (ej: cleansed_movies)
  - pipeline_id
  - correlation_id
  - aws_region
"""

import sys
import json
import logging
import time
from datetime import datetime, timezone

import boto3
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType

# ============================================================
# ARGUMENTOS
# ============================================================
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'obt_config_parameter',
    'source_ml_database',
    'source_ml_ratings_table',
    'source_ml_links_table',
    'source_tmdb_database',
    'source_tmdb_movies_table',
    'target_database',
    'target_ratings_table',
    'target_movies_table',
    'pipeline_id',
    'correlation_id',
    'aws_region',
])

# ============================================================
# INICIALIZACIÓN SPARK + GLUE
# ============================================================
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ============================================================
# LOGGING
# ============================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s'))
logger.addHandler(handler)

# ============================================================
# CONSTANTES
# ============================================================
REGION = args['aws_region']
PIPELINE_ID = args['pipeline_id']
CORRELATION_ID = args['correlation_id']
OBT_CONFIG_PARAMETER = args['obt_config_parameter']

# Databases y tablas (vienen por argumentos de Glue)
SOURCE_ML_DATABASE = args['source_ml_database']
SOURCE_ML_RATINGS_TABLE = args['source_ml_ratings_table']
SOURCE_ML_LINKS_TABLE = args['source_ml_links_table']
SOURCE_TMDB_DATABASE = args['source_tmdb_database']
SOURCE_TMDB_MOVIES_TABLE = args['source_tmdb_movies_table']
TARGET_DATABASE = args['target_database']
TARGET_RATINGS_TABLE = args['target_ratings_table']
TARGET_MOVIES_TABLE = args['target_movies_table']

# Clientes AWS
ssm_client = boto3.client('ssm', region_name=REGION)
s3_client = boto3.client('s3', region_name=REGION)
glue_client = boto3.client('glue', region_name=REGION)


# ============================================================
# CONFIGURACIÓN DESDE PARAMETER STORE
# ============================================================
def load_config_from_ssm() -> dict:
    """
    Lee toda la configuración del job desde SSM Parameter Store.
    Centraliza paths, databases y tablas en un solo parámetro JSON.
    """
    logger.info(f"Cargando configuración desde SSM: {OBT_CONFIG_PARAMETER}")
    try:
        response = ssm_client.get_parameter(Name=OBT_CONFIG_PARAMETER, WithDecryption=True)
        config = json.loads(response['Parameter']['Value'])
        logger.info("Configuración cargada exitosamente.")
        return config
    except Exception as e:
        logger.error(f"Error leyendo configuración desde SSM: {e}")
        raise


# ============================================================
# FUNCIONES UTILITARIAS
# ============================================================
def _table_exists(database_name: str, table_name: str) -> bool:
    """Verifica si una tabla existe en el Glue Data Catalog."""
    try:
        glue_client.get_table(DatabaseName=database_name, Name=table_name)
        return True
    except glue_client.exceptions.EntityNotFoundException:
        return False
    except Exception as e:
        logger.warning(f"Error verificando tabla {database_name}.{table_name}: {e}")
        return False


def _parse_s3_path(s3_path: str) -> tuple:
    """Parsea s3://bucket/prefix en (bucket, prefix)."""
    path = s3_path.replace("s3://", "")
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


# ============================================================
# FUNCIONES DE LECTURA (Catalog-first con fallback a S3)
# ============================================================
def read_table_or_path(database: str, table: str, s3_fallback_path: str, file_format: str = "csv"):
    """
    Estrategia de lectura dual:
      1. Si la tabla existe en Glue Catalog → lee desde catálogo (óptimo para Athena/particiones)
      2. Si no existe → lee directamente desde el path S3 (fallback robusto)
    """
    if _table_exists(database, table):
        logger.info(f"  Leyendo desde Catalog: {database}.{table}")
        df = glueContext.create_dynamic_frame.from_catalog(
            database=database, table_name=table
        ).toDF()
    else:
        logger.info(f"  Tabla {database}.{table} no encontrada. Fallback a S3: {s3_fallback_path}")
        if file_format == "json":
            df = spark.read.json(s3_fallback_path)
        else:
            df = spark.read.csv(s3_fallback_path, header=True, inferSchema=True)

    count = df.count()
    logger.info(f"  → Registros cargados: {count:,}")
    return df


def load_ratings(config: dict):
    """Carga ratings de MovieLens."""
    s3_path = config['source_s3_paths']['bronze_ratings_path']
    logger.info("Cargando ratings...")
    return read_table_or_path(
        database=SOURCE_ML_DATABASE,
        table=SOURCE_ML_RATINGS_TABLE,
        s3_fallback_path=s3_path,
        file_format="csv"
    )


def load_links(config: dict):
    """Carga links de MovieLens (movieId <-> tmdbId)."""
    s3_path = config['source_s3_paths']['bronze_links_path']
    logger.info("Cargando links...")
    df = read_table_or_path(
        database=SOURCE_ML_DATABASE,
        table=SOURCE_ML_LINKS_TABLE,
        s3_fallback_path=s3_path,
        file_format="csv"
    )
    # Limpiar: eliminar nulos y duplicados
    df = df.dropna(subset=['tmdbId']).dropDuplicates(['movieId', 'tmdbId'])
    df = df.withColumn('tmdbId', F.col('tmdbId').cast(IntegerType()))
    logger.info(f"  → Links válidos (deduplicados): {df.count():,}")
    return df


def load_tmdb_catalog(config: dict):
    """Carga el catálogo TMDB desde Catalog o fallback JSONL en S3."""
    s3_path = config['source_s3_paths']['bronze_tmdb_path']
    logger.info("Cargando catálogo TMDB...")
    return read_table_or_path(
        database=SOURCE_TMDB_DATABASE,
        table=SOURCE_TMDB_MOVIES_TABLE,
        s3_fallback_path=s3_path,
        file_format="json"
    )


def get_poster_movie_ids(config: dict):
    """
    Lista los movieIds con poster existente en S3 Silver.
    Los archivos se nombran {movieId}.jpg.
    """
    posters_path = config['source_s3_paths']['silver_posters_path']
    logger.info(f"Listando posters disponibles en: {posters_path}")

    bucket, prefix = _parse_s3_path(posters_path.rstrip('/') + '/')
    movie_ids = []

    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            filename = obj['Key'].split('/')[-1]
            if filename.endswith('.jpg'):
                try:
                    movie_ids.append(int(filename.replace('.jpg', '')))
                except ValueError:
                    continue

    logger.info(f"  → Posters encontrados: {len(movie_ids):,}")

    if movie_ids:
        df_posters = spark.createDataFrame(
            [(mid,) for mid in movie_ids],
            schema=StructType([StructField("movieId", IntegerType(), False)])
        )
    else:
        df_posters = spark.createDataFrame(
            [], schema=StructType([StructField("movieId", IntegerType(), False)])
        )

    return df_posters


# ============================================================
# FUNCIONES DE TRANSFORMACIÓN
# ============================================================
def build_obt_ratings(df_ratings, df_links, df_tmdb_catalog, df_posters):
    """
    Construye el OBT de ratings con triple intersección:
      ratings × links × tmdb_catalog × posters
    Solo interacciones con las 3 fuentes disponibles.
    Equivalente a: obt_tmdb_entrenamiento del notebook.
    """
    logger.info("Construyendo OBT Ratings (triple intersección)...")

    # Paso 1: Ratings + Links → obtener tmdbId por movieId
    df_merged = df_ratings.join(
        df_links.select('movieId', 'tmdbId'), on='movieId', how='inner'
    )
    logger.info(f"  → Tras merge ratings + links: {df_merged.count():,}")

    # Paso 2: Filtrar solo películas con metadata TMDB (texto)
    tmdb_ids = df_tmdb_catalog.select('movieId').distinct()
    df_merged = df_merged.join(tmdb_ids, on='movieId', how='inner')
    logger.info(f"  → Tras intersección con TMDB catalog: {df_merged.count():,}")

    # Paso 3: Filtrar solo películas con poster visual disponible
    df_merged = df_merged.join(df_posters, on='movieId', how='inner')
    logger.info(f"  → Tras intersección con posters: {df_merged.count():,}")

    # Paso 4: Selección final + deduplicación
    df_obt_ratings = df_merged.select(
        F.col('userId').cast(IntegerType()),
        F.col('movieId').cast(IntegerType()),
        F.col('rating').cast(FloatType()),
        F.col('timestamp').cast(LongType()),
    ).dropDuplicates(['userId', 'movieId'])

    n_total = df_obt_ratings.count()
    n_users = df_obt_ratings.select('userId').distinct().count()
    n_movies = df_obt_ratings.select('movieId').distinct().count()

    logger.info(f"  → OBT Ratings final: {n_total:,} interacciones | {n_users:,} usuarios | {n_movies:,} películas")
    return df_obt_ratings


def build_obt_movies(df_tmdb_catalog, df_obt_ratings):
    """
    Catálogo de películas filtrado por los movieIds del OBT ratings.
    Mantiene los arrays nativos (generos, actores, palabras_clave) sin aplanar.
    La textualización para embeddings se hará en SageMaker Processing.
    Equivalente a: df_movies_filtrado del notebook.
    """
    logger.info("Construyendo OBT Movies (catálogo filtrado por ratings válidos)...")

    valid_ids = df_obt_ratings.select('movieId').distinct()
    df_movies = df_tmdb_catalog.join(valid_ids, on='movieId', how='inner')

    # Columnas de interés (se seleccionan las disponibles)
    desired_columns = [
        'movieId', 'tmdbId', 'titulo', 'titulo_original', 'sinopsis',
        'generos', 'director', 'actores', 'palabras_clave',
        'fecha_lanzamiento', 'calificacion_tmdb', 'votos_tmdb',
        'frase_promocional', 'popularidad', 'poster_path',
        'idioma_original', 'duracion_minutos',
    ]
    available = [c for c in desired_columns if c in df_movies.columns]
    df_movies = df_movies.select(*available)

    n_movies = df_movies.count()
    logger.info(f"  → OBT Movies final: {n_movies:,} películas")
    return df_movies


# ============================================================
# FUNCIONES DE ESCRITURA (Apache Iceberg)
# ============================================================
def write_iceberg_table(df, database: str, table_name: str, s3_base_path: str):
    """
    Escribe DataFrame como tabla Apache Iceberg en S3 y registra en Glue Catalog.
    Usa CREATE OR REPLACE para idempotencia (overwrite completo de los datos).
    
    La configuración de Iceberg (catalog, warehouse) ya viene inyectada por
    el módulo Terraform de Glue con add_iceberg_config=true.
    """
    full_table_name = f"glue_catalog.{database}.{table_name}"
    output_path = f"{s3_base_path.rstrip('/')}/{table_name}/"
    logger.info(f"  Escribiendo Iceberg: {full_table_name} → {output_path}")

    # Registrar como tabla temporal para usar SparkSQL
    temp_view = f"tmp_{table_name}"
    df.createOrReplaceTempView(temp_view)

    # CREATE TABLE IF NOT EXISTS + INSERT OVERWRITE (idempotente)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {full_table_name}
        USING iceberg
        LOCATION '{output_path}'
        AS SELECT * FROM {temp_view} WHERE 1=0
    """)

    # Overwrite completo (TRUNCATE + INSERT para idempotencia)
    spark.sql(f"""
        INSERT OVERWRITE {full_table_name}
        SELECT * FROM {temp_view}
    """)

    row_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {full_table_name}").collect()[0]['cnt']
    logger.info(f"  → Tabla Iceberg {full_table_name} escrita: {row_count:,} registros")


# ============================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================
def main():
    inicio = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"Job: {args['JOB_NAME']}")
    logger.info(f"Pipeline ID: {PIPELINE_ID}")
    logger.info(f"Correlation ID: {CORRELATION_ID}")
    logger.info(f"{'='*60}")

    # 1. CONFIGURACIÓN (SSM Parameter Store — solo paths S3)
    logger.info("\n[PASO 1/5] Cargando configuración de paths desde Parameter Store...")
    config = load_config_from_ssm()

    target_s3_base = config['target_s3_path']

    logger.info(f"  Source ML DB: {SOURCE_ML_DATABASE} (tables: {SOURCE_ML_RATINGS_TABLE}, {SOURCE_ML_LINKS_TABLE})")
    logger.info(f"  Source TMDB DB: {SOURCE_TMDB_DATABASE} (table: {SOURCE_TMDB_MOVIES_TABLE})")
    logger.info(f"  Target DB: {TARGET_DATABASE}")
    logger.info(f"  Target Tables: {TARGET_RATINGS_TABLE}, {TARGET_MOVIES_TABLE}")
    logger.info(f"  Target S3: {target_s3_base}")

    # 2. CARGA DE DATOS (Bronze — Catalog-first con fallback S3)
    logger.info("\n[PASO 2/5] Cargando datos fuente (Bronze Layer)...")
    df_ratings = load_ratings(config)
    df_links = load_links(config)
    df_tmdb_catalog = load_tmdb_catalog(config)
    df_posters = get_poster_movie_ids(config)

    # 3. CONSTRUCCIÓN OBT RATINGS
    logger.info("\n[PASO 3/5] Construyendo cleansed_ratings...")
    df_obt_ratings = build_obt_ratings(df_ratings, df_links, df_tmdb_catalog, df_posters)

    # 4. CONSTRUCCIÓN OBT MOVIES
    logger.info("\n[PASO 4/5] Construyendo cleansed_movies...")
    df_obt_movies = build_obt_movies(df_tmdb_catalog, df_obt_ratings)

    # 5. ESCRITURA EN SILVER (Apache Iceberg)
    logger.info("\n[PASO 5/5] Escribiendo tablas Iceberg en Silver Layer...")
    write_iceberg_table(df_obt_ratings, TARGET_DATABASE, TARGET_RATINGS_TABLE, target_s3_base)
    write_iceberg_table(df_obt_movies, TARGET_DATABASE, TARGET_MOVIES_TABLE, target_s3_base)

    # RESUMEN
    duracion = round(time.time() - inicio, 2)
    logger.info(f"\n{'='*60}")
    logger.info(f"Job completado exitosamente en {duracion}s")
    logger.info(f"  {TARGET_RATINGS_TABLE}: {df_obt_ratings.count():,} registros")
    logger.info(f"  {TARGET_MOVIES_TABLE}: {df_obt_movies.count():,} registros")
    logger.info(f"{'='*60}")


# ============================================================
# EJECUCIÓN
# ============================================================
try:
    main()
except Exception as e:
    logger.error(f"Job FALLÓ: {e}")
    raise
finally:
    job.commit()
