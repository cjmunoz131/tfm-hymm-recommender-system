"""
AWS Glue Python Shell Job: TMDB Posters Download (Silver Layer)
================================================================
Descarga los posters de películas desde la CDN pública de TMDB usando el campo
poster_path previamente extraído en la capa Bronze (tmdb_baseline_bronze_job).

Tipo de Job: Python Shell (I/O bound — descarga de binarios desde CDN)
Runtime: Python 3.9 | Max DPU: 1 | Max Concurrency: 1

Flujo:
  1. Lee catálogo TMDB desde Glue Catalog o fallback S3 path
  2. Filtra películas con poster_path válido
  3. Verifica cuáles ya existen en S3 silver (idempotencia)
  4. Descarga concurrente de imágenes desde CDN de TMDB
  5. Sube cada poster como {movieId}.jpg en S3 silver layer

Argumentos Glue (--key value):
  - JOB_NAME
  - config_parameter:      nombre del parámetro SSM con config (paths, url, concurrency, image_size)
  - source_database:       database del Glue Catalog para catálogo TMDB (ej: tmdb_bronze)
  - source_table:          tabla del catálogo TMDB (ej: movies)
  - pipeline_id
  - correlation_id
  - aws_region
"""

import sys
import json
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Set

import aiohttp
import boto3
import pandas as pd
from awsglue.utils import getResolvedOptions

# ============================================================
# ARGUMENTOS
# ============================================================
args = getResolvedOptions(sys.argv, [
    'config_parameter',
    'source_database',
    'source_table',
    'pipeline_id',
    'correlation_id',
    'aws_region',
])

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
JOB_NAME = args.get('JOB_NAME', 'tmdb_posters_silver_job')
CONFIG_PARAMETER = args['config_parameter']
SOURCE_DATABASE = args['source_database']
SOURCE_TABLE = args['source_table']
REGION = args.get('aws_region', 'us-east-1')
PIPELINE_ID = args['pipeline_id']
CORRELATION_ID = args['correlation_id']

# ============================================================
# CLIENTES AWS
# ============================================================
session = boto3.Session(region_name=REGION)
s3_client = session.client('s3')
ssm_client = session.client('ssm')
glue_client = session.client('glue')


# ============================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================
def load_config() -> dict:
    """Lee la configuración completa del job desde SSM Parameter Store."""
    logger.info(f"Cargando configuración desde SSM: {CONFIG_PARAMETER}")
    try:
        response = ssm_client.get_parameter(Name=CONFIG_PARAMETER, WithDecryption=True)
        config = json.loads(response['Parameter']['Value'])
        logger.info("Configuración cargada exitosamente.")
        return config
    except Exception as e:
        logger.error(f"Error leyendo configuración desde SSM: {e}")
        raise


# ============================================================
# FUNCIONES UTILITARIAS
# ============================================================
def _parse_s3_path(s3_path: str) -> tuple:
    """Parsea s3://bucket/prefix en (bucket, prefix)."""
    path = s3_path.replace("s3://", "")
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _table_exists(database: str, table: str) -> bool:
    """Verifica si una tabla existe en el Glue Data Catalog."""
    try:
        glue_client.get_table(DatabaseName=database, Name=table)
        return True
    except glue_client.exceptions.EntityNotFoundException:
        return False
    except Exception as e:
        logger.warning(f"Error verificando tabla {database}.{table}: {e}")
        return False


def _get_table_s3_location(database: str, table: str) -> str:
    """Obtiene la ubicación S3 de una tabla del Catalog."""
    table_info = glue_client.get_table(DatabaseName=database, Name=table)
    return table_info['Table']['StorageDescriptor']['Location']


# ============================================================
# FUNCIONES DE LECTURA (Catalog-first con fallback a S3)
# ============================================================
def read_catalog(s3_fallback_path: str) -> pd.DataFrame:
    """
    Lee el catálogo TMDB desde Glue Catalog o fallback a JSONL en S3.
    Retorna un DataFrame con al menos 'movieId' y 'poster_path'.
    """
    if _table_exists(SOURCE_DATABASE, SOURCE_TABLE):
        logger.info(f"Leyendo desde Catalog: {SOURCE_DATABASE}.{SOURCE_TABLE}")
        s3_location = _get_table_s3_location(SOURCE_DATABASE, SOURCE_TABLE)
        return _read_jsonl_from_s3_prefix(s3_location)
    else:
        logger.info(f"Tabla no encontrada. Fallback a S3: {s3_fallback_path}")
        return _read_jsonl_from_s3_prefix(s3_fallback_path)


def _read_jsonl_from_s3_prefix(s3_prefix: str) -> pd.DataFrame:
    """Lee todos los archivos JSONL bajo un prefijo S3."""
    bucket, prefix = _parse_s3_path(s3_prefix.rstrip('/') + '/')
    paginator = s3_client.get_paginator('list_objects_v2')
    all_records = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.jsonl') or key.endswith('.json'):
                response = s3_client.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read().decode('utf-8')
                for line in content.strip().split('\n'):
                    if line.strip():
                        all_records.append(json.loads(line))

    if not all_records:
        logger.warning(f"No se encontraron registros en: s3://{bucket}/{prefix}")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    logger.info(f"Registros leídos: {len(df)}")
    return df


def get_existing_posters(target_s3_path: str) -> Set[int]:
    """Lista los movieIds de posters ya existentes en S3 (idempotencia)."""
    bucket, prefix = _parse_s3_path(target_s3_path.rstrip('/') + '/')
    existing_ids = set()

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                filename = obj['Key'].split('/')[-1]
                if filename.endswith('.jpg'):
                    try:
                        existing_ids.add(int(filename.replace('.jpg', '')))
                    except ValueError:
                        continue
    except Exception as e:
        logger.warning(f"Error listando posters existentes: {e}")

    logger.info(f"Posters ya existentes en silver: {len(existing_ids)}")
    return existing_ids


def upload_poster_to_s3(image_bytes: bytes, movie_id: int, target_s3_path: str) -> bool:
    """Sube un poster individual a S3 silver layer."""
    bucket, prefix = _parse_s3_path(target_s3_path.rstrip('/'))
    key = f"{prefix}/{movie_id}.jpg"

    try:
        s3_client.put_object(
            Bucket=bucket, Key=key,
            Body=image_bytes,
            ContentType='image/jpeg',
            Metadata={
                'movie_id': str(movie_id),
                'correlation_id': CORRELATION_ID,
                'extraction_date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            }
        )
        return True
    except Exception as e:
        logger.error(f"Error subiendo poster movieId={movie_id}: {e}")
        return False


# ============================================================
# FUNCIONES DE EXTRACCIÓN (LÓGICA DE NEGOCIO)
# ============================================================
async def descargar_poster(
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    image_base_url: str,
    image_size: str,
    movie_id: int,
    poster_path: str,
    max_reintentos: int = 3,
) -> Optional[tuple]:
    """Descarga un poster desde la CDN pública de TMDB."""
    url = f"{image_base_url}/{image_size}{poster_path}"

    async with semaphore:
        for intento in range(max_reintentos):
            try:
                async with http_session.get(url) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        if len(image_bytes) > 1024:
                            return (movie_id, image_bytes)
                        return None
                    elif response.status == 429:
                        await asyncio.sleep(int(response.headers.get('Retry-After', 3)))
                    elif response.status == 404:
                        return None
                    else:
                        if intento < max_reintentos - 1:
                            await asyncio.sleep(2 ** intento)
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if intento < max_reintentos - 1:
                    await asyncio.sleep(2 ** intento)
                else:
                    logger.warning(f"Error descargando poster movieId={movie_id}: {e}")
    return None


async def ejecutar_descarga_posters(
    df_pendientes: pd.DataFrame,
    image_base_url: str,
    image_size: str,
    max_concurrency: int,
    target_s3_path: str,
) -> dict:
    """Orquestador de descarga paralela con upload a S3 por batches."""
    semaphore = asyncio.Semaphore(max_concurrency)
    timeout = aiohttp.ClientTimeout(total=60, connect=10)

    total = len(df_pendientes)
    logger.info(f"Iniciando descarga de {total} posters (concurrencia: {max_concurrency})")

    metricas = {'descargados': 0, 'fallidos': 0, 'subidos': 0}
    BATCH_SIZE = 500
    registros = df_pendientes[['movieId', 'poster_path']].to_dict('records')

    for batch_start in range(0, total, BATCH_SIZE):
        batch = registros[batch_start:batch_start + BATCH_SIZE]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Batch {batch_num}/{total_batches} ({len(batch)} posters)")

        # Connector y session frescos por batch (evita conexiones reutilizadas rotas)
        connector = aiohttp.TCPConnector(limit=max_concurrency, limit_per_host=max_concurrency)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as http_session:
            tareas = [
                descargar_poster(http_session, semaphore, image_base_url, image_size, r['movieId'], r['poster_path'])
                for r in batch
            ]
            resultados = await asyncio.gather(*tareas, return_exceptions=True)

        for resultado in resultados:
            if isinstance(resultado, Exception):
                metricas['fallidos'] += 1
            elif resultado is not None:
                movie_id, image_bytes = resultado
                metricas['descargados'] += 1
                if upload_poster_to_s3(image_bytes, movie_id, target_s3_path):
                    metricas['subidos'] += 1
                else:
                    metricas['fallidos'] += 1
            else:
                metricas['fallidos'] += 1

        logger.info(f"  Acumulado: {metricas['subidos']} subidos, {metricas['fallidos']} fallidos")

    return metricas


# ============================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================
def main():
    inicio = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"Job: {JOB_NAME}")
    logger.info(f"Pipeline ID: {PIPELINE_ID}")
    logger.info(f"Correlation ID: {CORRELATION_ID}")
    logger.info(f"{'='*60}")

    # 1. Configuración desde Parameter Store
    config = load_config()
    image_base_url = config['tmdb_posters_url_template']
    image_size = config.get('image_size', 'w500')
    max_concurrency = int(config.get('max_concurrency', 20))
    source_s3_path = config['source_s3_path']
    target_s3_path = config['target_s3_path']

    logger.info(f"  Image URL: {image_base_url}/{image_size}/...")
    logger.info(f"  Target: {target_s3_path}")

    # 2. Leer catálogo TMDB (Catalog-first con fallback S3)
    df_catalog = read_catalog(s3_fallback_path=source_s3_path)

    if df_catalog.empty:
        logger.warning("Catálogo vacío. Finalizando.")
        return

    # 3. Filtrar películas con poster_path válido
    df_con_poster = df_catalog[
        df_catalog['poster_path'].notna() &
        (df_catalog['poster_path'] != '') &
        (df_catalog['poster_path'] != 'None')
    ].copy()

    logger.info(f"Películas con poster_path: {len(df_con_poster)} / {len(df_catalog)}")

    if df_con_poster.empty:
        logger.warning("No hay películas con poster_path. Finalizando.")
        return

    # 4. Idempotencia: excluir posters ya descargados
    existing_ids = get_existing_posters(target_s3_path)
    df_pendientes = df_con_poster[~df_con_poster['movieId'].isin(existing_ids)].copy()

    logger.info(f"Posters pendientes: {len(df_pendientes)} (ya existentes: {len(existing_ids)})")

    if df_pendientes.empty:
        logger.info("Todos los posters ya están en silver.")
        return

    # 5. Descarga paralela + upload a S3
    metricas = asyncio.run(
        ejecutar_descarga_posters(df_pendientes, image_base_url, image_size, max_concurrency, target_s3_path)
    )

    # 6. Resumen
    duracion = round(time.time() - inicio, 2)
    logger.info(f"{'='*60}")
    logger.info(f"Job finalizado | Subidos: {metricas['subidos']} | Fallidos: {metricas['fallidos']} | {duracion}s")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
