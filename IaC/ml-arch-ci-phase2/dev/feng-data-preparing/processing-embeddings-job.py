"""
SageMaker Processing Job: Multimodal Embeddings Generation (Bedrock Nova)
==========================================================================
Genera el catálogo de embeddings multimodales (texto + imagen) para cada película
usando Amazon Bedrock Nova. Usa ThreadPoolExecutor para paralelizar las invocaciones.

Arquitectura:
  - Lee el catálogo de películas (obt_movies) desde Silver
  - Construye la super-sinopsis semántica por película
  - Invoca Bedrock Nova en paralelo (ThreadPoolExecutor, 5-10 workers)
  - Cada worker procesa: 1 call texto + 1 call imagen por película
  - Guarda resultado como embeddings_catalog.pkl

Concurrencia:
  - ThreadPoolExecutor es thread-safe con boto3 (cada thread usa su propia session)
  - Bedrock Nova soporta ~50 TPS; con 5 workers nos mantenemos en ~10 TPS (seguro)
  - Retry con backoff exponencial por worker

Inputs:
  - /opt/ml/processing/input/movies/    → Parquet de cleansed_movies (Silver)
  - /opt/ml/processing/input/posters/   → Imágenes JPG ({movieId}.jpg)

Outputs:
  - /opt/ml/processing/output/embeddings/ → embeddings_catalog.pkl

Instance recomendada: ml.t3.medium o ml.m5.large (I/O bound, no CPU bound)
Tiempo estimado (9K películas, 5 workers): ~15-20 min
"""

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "--quiet"])

import argparse
import ast
import base64
import json
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from threading import Lock
from typing import Any, Dict, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTES
# ============================================================
BEDROCK_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
BEDROCK_EMBEDDING_DIM = 1024
DEFAULT_MAX_WORKERS = 5
DEFAULT_MAX_RETRIES = 3


# ============================================================
# CLASE ENCODER (Basada en tu notebook, adaptada para threading)
# ============================================================
class AwsNovaMultimodalEncoder:
    """
    Encoder de embeddings multimodales con Amazon Bedrock Nova.
    Thread-safe: cada invocación usa el cliente boto3 del thread.
    """

    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.model_id = BEDROCK_MODEL_ID
        self.embedding_dim = BEDROCK_EMBEDDING_DIM
        logger.info(f"Encoder inicializado | Region: {region_name} | Model: {self.model_id}")

    def _get_client(self):
        """Crea un cliente Bedrock por thread (thread-safe)."""
        return boto3.client("bedrock-runtime", region_name=self.region_name)

    def _imagen_a_base64(self, ruta_img: str) -> str:
        """Convierte imagen a base64 JPEG (maneja formatos mixtos)."""
        img = Image.open(ruta_img).convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _obtener_vector(self, client, payload: dict, max_retries: int = DEFAULT_MAX_RETRIES) -> np.ndarray:
        """Invoca Bedrock con retry y backoff exponencial."""
        for intento in range(max_retries):
            try:
                response = client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(payload),
                )
                response_body = json.loads(response["body"].read())
                vector = response_body["embeddings"][0]["embedding"]
                return np.array(vector, dtype=np.float32)

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "ThrottlingException" and intento < max_retries - 1:
                    wait_time = 2 ** intento
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                if intento < max_retries - 1:
                    time.sleep(2 ** intento)
                else:
                    raise

    def procesar_pelicula(self, movie_id: int, sinopsis: str, ruta_img: str) -> Optional[Tuple[int, Dict[str, np.ndarray]]]:
        """
        Procesa una película: genera embedding de texto + imagen.
        Cada invocación crea su propio cliente Bedrock (thread-safe).

        Returns:
            Tupla (movieId, {'text_emb': array, 'img_emb': array}) o None si falla.
        """
        client = self._get_client()

        try:
            # 1. Embedding de texto
            payload_txt = {
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_INDEX",
                    "embeddingDimension": self.embedding_dim,
                    "text": {"truncationMode": "END", "value": sinopsis},
                },
            }
            v_txt = self._obtener_vector(client, payload_txt)

            # 2. Embedding de imagen
            if os.path.exists(ruta_img):
                img_b64 = self._imagen_a_base64(ruta_img)
                payload_img = {
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": {
                        "embeddingPurpose": "GENERIC_INDEX",
                        "embeddingDimension": self.embedding_dim,
                        "image": {"format": "jpeg", "source": {"bytes": img_b64}},
                    },
                }
                v_img = self._obtener_vector(client, payload_img)
            else:
                v_img = np.zeros(self.embedding_dim, dtype=np.float32)

            return (movie_id, {"text_emb": v_txt, "img_emb": v_img})

        except FileNotFoundError:
            return None
        except (ClientError, UnidentifiedImageError, Exception):
            return None


# ============================================================
# PREPARACIÓN DE DATOS
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
    """Construye la super-sinopsis enriquecida para el embedding de texto."""
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


def preparar_catalogo(df_movies: pd.DataFrame) -> pd.DataFrame:
    """Prepara el DataFrame con sinopsis semántica para el encoder."""
    columnas_array = ["actores", "palabras_clave", "generos"]
    df = df_movies.copy()
    for col in columnas_array:
        if col in df.columns:
            df[col] = df[col].apply(limpiar_lista_a_string)
    df["sinopsis_semantica"] = df.apply(crear_sinopsis_semantica, axis=1)
    return df


# ============================================================
# GENERACIÓN PARALELA DE EMBEDDINGS
# ============================================================
def generar_embeddings_paralelo(
    df_peliculas: pd.DataFrame,
    posters_path: str,
    region: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Genera embeddings en paralelo usando ThreadPoolExecutor.
    Cada thread crea su propio cliente Bedrock (thread-safe).

    Args:
        df_peliculas: DataFrame con movieId y sinopsis_semantica
        posters_path: Directorio con las imágenes
        region: Región AWS
        max_workers: Número de threads concurrentes

    Returns:
        Diccionario {movieId: {'text_emb': array, 'img_emb': array}}
    """
    encoder = AwsNovaMultimodalEncoder(region_name=region)
    catalogo = {}
    errores = 0
    total = len(df_peliculas)

    # Preparar lista de tareas
    tareas = []
    for _, row in df_peliculas.iterrows():
        movie_id = int(row["movieId"])
        sinopsis = row["sinopsis_semantica"]
        ruta_img = os.path.join(posters_path, f"{movie_id}.jpg")
        tareas.append((movie_id, sinopsis, ruta_img))

    logger.info(f"Iniciando generación paralela: {total:,} películas | {max_workers} workers")

    # Lock para el logging de progreso (thread-safe)
    progress_lock = Lock()
    processed = [0]

    def log_progress():
        with progress_lock:
            processed[0] += 1
            if processed[0] % 500 == 0 or processed[0] == total:
                logger.info(f"  Progreso: {processed[0]:,}/{total:,} | OK: {len(catalogo):,} | Errores: {errores}")

    # Ejecución paralela
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(encoder.procesar_pelicula, mid, sin, img): mid
            for mid, sin, img in tareas
        }

        for future in as_completed(futures):
            movie_id = futures[future]
            try:
                resultado = future.result()
                if resultado is not None:
                    mid, embeddings = resultado
                    catalogo[mid] = embeddings
                else:
                    errores += 1
            except Exception as e:
                errores += 1
                if errores <= 5:
                    logger.error(f"Error en movieId={movie_id}: {e}")

            log_progress()

    logger.info(f"Generación completada: {len(catalogo):,}/{total:,} éxitos | {errores} errores")
    return catalogo


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Threads concurrentes para Bedrock (default: 5)")
    args = parser.parse_args()

    inicio = time.time()
    logger.info("=" * 60)
    logger.info("Processing Job: Multimodal Embeddings Generation")
    logger.info(f"  Region: {args.region}")
    logger.info(f"  Workers: {args.max_workers}")
    logger.info(f"  Model: {BEDROCK_MODEL_ID}")
    logger.info("=" * 60)

    # Paths
    input_movies = "/opt/ml/processing/input/movies"
    input_posters = "/opt/ml/processing/input/posters"
    output_embeddings = "/opt/ml/processing/output/embeddings"

    # 1. Cargar catálogo de películas
    logger.info("\n[PASO 1/3] Cargando catálogo de películas...")
    # Soporta tablas Iceberg (carpeta con metadata/ y data/) o parquet directo
    movies_data_path = os.path.join(input_movies, "data")
    if os.path.isdir(movies_data_path):
        logger.info(f"  Detectada estructura Iceberg, leyendo desde: {movies_data_path}")
        df_movies = pd.read_parquet(movies_data_path)
    else:
        df_movies = pd.read_parquet(input_movies)
    logger.info(f"  → {len(df_movies):,} películas cargadas")

    # 2. Preparar sinopsis semántica
    logger.info("\n[PASO 2/3] Preparando textualización semántica...")
    df_prepared = preparar_catalogo(df_movies)
    logger.info(f"  → Ejemplo: {df_prepared['sinopsis_semantica'].iloc[0][:100]}...")

    # 3. Generar embeddings en paralelo
    logger.info(f"\n[PASO 3/3] Generando embeddings ({args.max_workers} workers paralelos)...")
    catalogo = generar_embeddings_paralelo(
        df_prepared, input_posters, args.region, max_workers=args.max_workers
    )

    # Guardar resultado
    os.makedirs(output_embeddings, exist_ok=True)
    filepath = os.path.join(output_embeddings, "embeddings_catalog.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(catalogo, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    duracion = round(time.time() - inicio, 2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Job completado en {duracion}s ({duracion/60:.1f} min)")
    logger.info(f"  Películas procesadas: {len(catalogo):,}")
    logger.info(f"  Archivo: {filepath} ({size_mb:.1f} MB)")
    logger.info(f"  Dimensión embeddings: {BEDROCK_EMBEDDING_DIM}D (texto + imagen)")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
