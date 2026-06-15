"""
AWS Glue Python Shell Job: TMDB Baseline Metadata Extraction (Bronze Layer)
===========================================================================
Extrae metadata cinematográfica completa (credits, keywords, poster_path) de la API
de TMDB para todas las películas del dataset MovieLens (links.csv).

Tipo de Job: Python Shell (I/O bound, no requiere procesamiento distribuido)
Runtime: Python 3.9 | Max DPU: 1 | Max Concurrency: 1

Flujo:
  1. Lee links desde Glue Catalog o fallback S3 path (bronze/ml/links/)
  2. Elimina duplicados por tmdbId y movieId
  3. Obtiene API key desde Secrets Manager
  4. Obtiene URL template, paths S3, y config desde Parameter Store
  5. Ejecuta extracción asíncrona paralela (controlada por semáforo)
  6. Escribe resultado como JSON Lines en S3 (bronze/tmdb/movies/)

Argumentos Glue (--key value):
  - JOB_NAME
  - config_parameter:       nombre del parámetro SSM con config completa (paths, url, concurrency)
  - secret_name:            nombre del secreto en Secrets Manager (API key TMDB)
  - source_database:        database del Glue Catalog para links (ej: ml_bronze)
  - source_table:           tabla de links en el Catalog (ej: links)
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
from typing import Optional

import aiohttp
import boto3
import pandas as pd
from awsglue.utils import getResolvedOptions

# ============================================================
# ARGUMENTOS
# ============================================================
args = getResolvedOptions(sys.argv, [
    'config_parameter',
    'secret_name',
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
JOB_NAME = args.get('JOB_NAME', 'tmdb_baseline_bronze_job')
CONFIG_PARAMETER = args['config_parameter']
SECRET_NAME = args['secret_name']
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
secrets_client = session.client('secretsmanager')
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


def get_api_key() -> str:
    """Obtiene la API key de TMDB desde AWS Secrets Manager."""
    try:
        response = secrets_client.get_secret_value(SecretId=SECRET_NAME)
        secret_data = json.loads(response['SecretString'])
        api_key = secret_data.get('api_key') or secret_data.get('API_KEY')
        if not api_key:
            raise KeyError("El secreto no contiene el campo 'api_key' o 'API_KEY'")
        logger.info("API key obtenida exitosamente desde Secrets Manager.")
        return api_key
    except Exception as e:
        logger.error(f"Error obteniendo API key: {e}")
        raise


# ============================================================
# FUNCIONES DE LECTURA (Catalog-first con fallback a S3)
# ============================================================
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


def read_links(s3_fallback_path: str) -> pd.DataFrame:
    """
    Lee el dataframe de links desde Glue Catalog o fallback S3.
    Retorna DataFrame limpio con columnas [movieId, tmdbId].
    """
    if _table_exists(SOURCE_DATABASE, SOURCE_TABLE):
        logger.info(f"Leyendo desde Catalog: {SOURCE_DATABASE}.{SOURCE_TABLE}")
        s3_location = _get_table_s3_location(SOURCE_DATABASE, SOURCE_TABLE)
        df = _read_csv_from_s3_prefix(s3_location)
    else:
        logger.info(f"Tabla no encontrada en Catalog. Fallback a S3: {s3_fallback_path}")
        df = _read_csv_from_s3_prefix(s3_fallback_path)

    registros_originales = len(df)
    logger.info(f"Registros cargados: {registros_originales}")

    # Limpiar: eliminar nulos y duplicados
    df = df.dropna(subset=['tmdbId'])
    df['tmdbId'] = df['tmdbId'].astype(int)
    df['movieId'] = df['movieId'].astype(int)
    df = df.drop_duplicates(subset=['tmdbId', 'movieId'], keep='first')

    logger.info(f"Registros tras deduplicación: {len(df)} (eliminados: {registros_originales - len(df)})")
    return df[['movieId', 'tmdbId']].reset_index(drop=True)


def _read_csv_from_s3_prefix(s3_path: str) -> pd.DataFrame:
    """Lee archivos CSV bajo un prefijo S3 y los combina."""
    bucket, prefix = _parse_s3_path(s3_path.rstrip('/') + '/')
    paginator = s3_client.get_paginator('list_objects_v2')
    frames = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.csv'):
                response = s3_client.get_object(Bucket=bucket, Key=key)
                frames.append(pd.read_csv(response['Body']))

    if not frames:
        # Intenta leer como un solo archivo (path directo)
        try:
            response = s3_client.get_object(Bucket=bucket, Key=prefix.rstrip('/'))
            return pd.read_csv(response['Body'])
        except Exception:
            pass
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _parse_s3_path(s3_path: str) -> tuple:
    """Parsea s3://bucket/prefix en (bucket, prefix)."""
    path = s3_path.replace("s3://", "")
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


# ============================================================
# FUNCIONES DE ESCRITURA
# ============================================================
def write_results_to_s3(resultados: list, target_s3_path: str, language: str) -> str:
    """Escribe resultados como JSON Lines en S3."""
    if not resultados:
        logger.warning("No hay resultados para escribir.")
        return ""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"movies.jsonl"

    bucket, prefix = _parse_s3_path(target_s3_path.rstrip('/'))
    key = f"{prefix}/{filename}"

    jsonl_content = "\n".join(json.dumps(record, ensure_ascii=False) for record in resultados)
    s3_client.put_object(
        Bucket=bucket, Key=key,
        Body=jsonl_content.encode('utf-8'),
        ContentType='application/jsonl'
    )

    output_path = f"s3://{bucket}/{key}"
    logger.info(f"Resultados escritos en: {output_path} ({len(resultados)} registros)")
    return output_path


# ============================================================
# FUNCIONES DE EXTRACCIÓN (LÓGICA DE NEGOCIO)
# ============================================================
def limpiar_texto(texto: Optional[str]) -> str:
    """Limpia caracteres Unicode ocultos y saltos de línea."""
    if not texto:
        return ""
    return texto.replace('\u2028', ' ').replace('\u2029', ' ').replace('\r', '').replace('\n', ' ').strip()


def transformar_respuesta(data: dict, movie_id: int, tmdb_id: int) -> dict:
    """Transforma la respuesta cruda de la API TMDB en el esquema de destino."""
    equipo = data.get('credits', {}).get('crew', [])
    director = next(
        (m['name'] for m in equipo if m.get('job') == 'Director'), "Desconocido"
    )
    elenco = data.get('credits', {}).get('cast', [])
    actores = [actor['name'] for actor in elenco[:5]]
    etiquetas = [kw['name'] for kw in data.get('keywords', {}).get('keywords', [])]

    return {
        'movieId': int(movie_id),
        'tmdbId': int(tmdb_id),
        'titulo': data.get('title', ''),
        'titulo_original': data.get('original_title', ''),
        'sinopsis': limpiar_texto(data.get('overview', '')),
        'generos': [g['name'] for g in data.get('genres', [])],
        'director': director,
        'actores': actores,
        'palabras_clave': etiquetas,
        'fecha_lanzamiento': data.get('release_date', ''),
        'calificacion_tmdb': data.get('vote_average', 0),
        'votos_tmdb': data.get('vote_count', 0),
        'frase_promocional': limpiar_texto(data.get('tagline', '')),
        'popularidad': data.get('popularity', 0),
        'poster_path': data.get('poster_path', ''),
        'backdrop_path': data.get('backdrop_path', ''),
        'idioma_original': data.get('original_language', ''),
        'duracion_minutos': data.get('runtime', 0),
        'presupuesto': data.get('budget', 0),
        'ingresos': data.get('revenue', 0),
        'estado': data.get('status', ''),
        'extraction_timestamp': datetime.now(timezone.utc).isoformat(),
        'correlation_id': CORRELATION_ID,
    }


async def obtener_metadatos(
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url_template: str,
    api_key: str,
    language: str,
    tmdb_id: int,
    movie_id: int,
    max_reintentos: int = 5,
) -> Optional[dict]:
    """
    Consulta asíncrona a la API de TMDB con retry y backoff exponencial.
    5 reintentos para mayor resiliencia en ambientes con NAT Gateway.
    """
    url = url_template.format(tmdb_id=tmdb_id, API_KEY=api_key, LANGUAGE=language)

    async with semaphore:
        for intento in range(max_reintentos):
            try:
                async with http_session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return transformar_respuesta(data, movie_id, tmdb_id)
                    elif response.status == 429:
                        espera = int(response.headers.get('Retry-After', 2))
                        await asyncio.sleep(espera)
                    elif response.status == 404:
                        return None
                    else:
                        if intento < max_reintentos - 1:
                            await asyncio.sleep(2 ** intento)
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if intento < max_reintentos - 1:
                    await asyncio.sleep(2 ** intento)
    return None


async def ejecutar_extraccion(df_links: pd.DataFrame, url_template: str, api_key: str, language: str, max_concurrency: int) -> list:
    """
    Orquestador de extracción por batches con sessions HTTP frescas.
    Procesa en grupos de BATCH_SIZE para evitar que una session degradada
    afecte a todas las peticiones. Todos los resultados se acumulan en
    una sola lista para escribir un único archivo al final.
    """
    BATCH_SIZE = 5000
    semaphore = asyncio.Semaphore(max_concurrency)
    timeout = aiohttp.ClientTimeout(total=60, connect=15)

    registros = df_links[['movieId', 'tmdbId']].to_dict('records')
    total = len(registros)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info(f"Iniciando extracción de {total} películas en {total_batches} batches (concurrencia: {max_concurrency})")

    resultados = []
    errores_totales = 0
    not_found_totales = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = registros[batch_start:batch_start + BATCH_SIZE]
        batch_num = (batch_start // BATCH_SIZE) + 1

        logger.info(f"Batch {batch_num}/{total_batches} ({len(batch)} películas)")

        # Connector y session frescos por batch (evita conexiones reutilizadas rotas)
        connector = aiohttp.TCPConnector(limit=max_concurrency, limit_per_host=max_concurrency)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as http_session:
            tareas = [
                obtener_metadatos(http_session, semaphore, url_template, api_key, language, r['tmdbId'], r['movieId'])
                for r in batch
            ]
            respuestas = await asyncio.gather(*tareas, return_exceptions=True)

        # Procesar resultados del batch
        batch_ok = 0
        batch_err = 0
        for resp in respuestas:
            if isinstance(resp, Exception):
                batch_err += 1
                errores_totales += 1
            elif resp is not None:
                resultados.append(resp)
                batch_ok += 1
            else:
                not_found_totales += 1

        logger.info(f"  → Batch {batch_num}: {batch_ok} éxitos | {len(batch) - batch_ok - batch_err} not found | {batch_err} errores")

    logger.info(
        f"Extracción completada: {len(resultados)} éxitos | "
        f"{not_found_totales} not found | {errores_totales} errores"
    )
    return resultados


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
    url_template = config['tmdb_api_url_template']
    language = config.get('language', 'en-US')
    max_concurrency = int(config.get('max_concurrency', 30))
    source_s3_path = config['source_s3_path']
    target_s3_path = config['target_s3_path']

    # 2. API Key desde Secrets Manager
    api_key = get_api_key()

    # 3. Leer datos fuente (Catalog-first con fallback S3)
    df_links = read_links(s3_fallback_path=source_s3_path)

    if df_links.empty:
        logger.warning("DataFrame vacío. Finalizando sin procesamiento.")
        return

    # 4. Extracción asíncrona
    resultados = asyncio.run(ejecutar_extraccion(df_links, url_template, api_key, language, max_concurrency))

    # 5. Escribir en S3 Bronze
    output_path = write_results_to_s3(resultados, target_s3_path, language)

    # 6. Resumen
    duracion = round(time.time() - inicio, 2)
    logger.info(f"{'='*60}")
    logger.info(f"Job finalizado | {len(resultados)} registros | {duracion}s | Output: {output_path}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
