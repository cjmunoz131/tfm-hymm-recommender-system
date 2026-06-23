import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

def evaluate_rigorous_ranking_reg_true_negatives(model, df_test, df_train, df_valid, dict_embeddings, device, k=10, num_decoys=99, threshold=3.5, bad_threshold=2.5):
    """
    Evaluación rigurosa de Regresión Pura usando VERDADEROS NEGATIVOS.
    Garantiza que los señuelos sean películas explícitamente mal calificadas (<= bad_threshold)
    en la base de datos, evaluando solo la predicción de estrellas.
    """
    print(f"📐 Iniciando Evaluación REGRESIÓN con VERDADEROS NEGATIVOS (Top-{k} con {num_decoys} señuelos MALOS)...")
    model.eval()

    df_all = pd.concat([df_train, df_valid, df_test])
    item_pool = set(df_all['movieId'].unique())
    global_history = df_all.groupby('userId')['movieId'].apply(set).to_dict()

    df_items = df_all[['movieId', 'movieId_idx', 'genres_multihot']].drop_duplicates(subset=['movieId']).set_index('movieId')

    hits = 0
    ndcg_sum = 0.0
    total_evaluations = 0

    # 1. Positivos del usuario (La aguja de oro que queremos que encuentre)
    df_test_positives = df_test[df_test['rating_scaled'] >= (threshold - 1.0)/4.0]

    # 2. Construimos la piscina global de "Películas Malas"
    df_malas = df_all[df_all['rating_scaled'] <= (bad_threshold - 1.0)/4.0]
    bad_item_pool = set(df_malas['movieId'].unique())

    print(f"   ↳ Piscina de Señuelos Malos detectada: {len(bad_item_pool)} películas.")

    usuarios_test = df_test_positives['userId'].unique()

    with torch.no_grad():
        for user_id in tqdm(usuarios_test, desc="Evaluando Usuarios (Regresión - True Negatives)"):
            user_idx = df_all[df_all['userId'] == user_id]['userId_idx'].iloc[0]
            user_test_movies = df_test_positives[df_test_positives['userId'] == user_id]['movieId'].values

            historial = global_history.get(user_id, set())

            # 3. SELECCIÓN DE SEÑUELOS ESTRICTOS
            posibles_senuelos = list(bad_item_pool - historial)

            for target_movie in user_test_movies:
                # Si por rareza estadística no hay 99 malas disponibles, rellenamos con normales
                if len(posibles_senuelos) < num_decoys:
                    faltantes = num_decoys - len(posibles_senuelos)
                    extra_decoys = list(item_pool - bad_item_pool - historial)
                    decoys = posibles_senuelos + list(np.random.choice(extra_decoys, size=faltantes, replace=False))
                else:
                    decoys = list(np.random.choice(posibles_senuelos, size=num_decoys, replace=False))

                eval_items = [target_movie] + decoys

                batch_users = torch.tensor([user_idx] * len(eval_items), dtype=torch.long).to(device)
                batch_items_idx = torch.tensor(df_items.loc[eval_items]['movieId_idx'].values, dtype=torch.long).to(device)
                batch_genres = torch.tensor(np.vstack(df_items.loc[eval_items]['genres_multihot'].values), dtype=torch.float32).to(device)

                text_embs = [dict_embeddings[i]['text_emb'] for i in eval_items]
                img_embs = [dict_embeddings[i]['img_emb'] for i in eval_items]
                batch_text = torch.tensor(np.vstack(text_embs), dtype=torch.float32).to(device)
                batch_img = torch.tensor(np.vstack(img_embs), dtype=torch.float32).to(device)

                # 🌟 EL CAMBIO: Extraemos solo la predicción de estrellas (y atención)
                pred_rating_scaled, _ = model(batch_users, batch_items_idx, batch_genres, batch_text, batch_img)

                # Aplanamos el tensor
                pred_rat = pred_rating_scaled.view(-1).cpu().numpy()

                # 🌟 EL CAMBIO DE MÉTRICA: Ordenamos directamente por las estrellas
                rankings = (-pred_rat).argsort().argsort()
                target_rank = rankings[0]

                if target_rank < k:
                    hits += 1
                    ndcg_sum += 1.0 / math.log2(target_rank + 2)

                total_evaluations += 1

    hit_rate = hits / total_evaluations if total_evaluations > 0 else 0
    ndcg_final = ndcg_sum / total_evaluations if total_evaluations > 0 else 0

    print("\n" + "-" * 50)
    print(f"🏆 RESULTADOS REGRESIÓN PURA (Top-{k} vs {num_decoys} Verdaderos Negativos)")
    print(f"   ↳ Hit Rate@{k}: {hit_rate:.4f}")
    print(f"   ↳ NDCG@{k}:     {ndcg_final:.4f}")
    print("-" * 50)

    return hit_rate, ndcg_final

import torch
import pandas as pd
import numpy as np

def evaluar_ndcg_microsoft_standard(model, test_loader, device, k=10):
    """
    Implementación del NDCG@K basada en el estándar de Microsoft Recommenders.
    Paso 1: Extrae las predicciones del DataLoader.
    Paso 2: Construye DataFrames y aplica la fórmula matemática del NDCG.
    """
    print("⚙️ Paso 1: Extrayendo predicciones del modelo a DataFrames de Pandas...")
    model.eval()

    users, items, y_true, y_pred = [], [], [], []

    # --- EXTRACCIÓN (PyTorch -> Listas) ---
    with torch.no_grad():
        for batch in test_loader:
            u = batch['user'].to(device)
            i = batch['item'].to(device)
            g = batch['genres'].to(device)
            t_emb = batch['text_emb'].to(device)
            i_emb = batch['img_emb'].to(device)
            rating_real = batch['rating'].to(device)

            rating_pred, _ = model(u, i, g, t_emb, i_emb)

            users.extend(u.cpu().numpy())
            items.extend(i.cpu().numpy())

            # Desescalamos a 1-5 estrellas para la fórmula del NDCG
            y_true.extend((rating_real.cpu().numpy() * 4.0) + 1.0)
            y_pred.extend((rating_pred.view(-1).cpu().numpy() * 4.0) + 1.0)

    # --- CONSTRUCCIÓN DE DATAFRAMES ---
    df_true = pd.DataFrame({'userId': users, 'movieId': items, 'rating': y_true})
    df_pred = pd.DataFrame({'userId': users, 'movieId': items, 'prediction': y_pred})

    print(f"📐 Paso 2: Calculando NDCG@{k} con la formulación estándar (2^rel - 1)...")

    # --- LÓGICA DEL REPOSITORIO (aryan-jadon / Microsoft) ---
    df_merged = pd.merge(df_true, df_pred, on=['userId', 'movieId'], how='inner')

    def calculate_ndcg_per_user(group):
        # 1. ORDEN PREDICHO: Ordenamos lo que el modelo cree que es mejor
        group_pred = group.sort_values(by='prediction', ascending=False).head(k)
        relevance_pred = group_pred['rating'].values  # Cuánto le gustó realmente ese Top K

        # 2. ORDEN IDEAL: Las mejores películas reales de este usuario en el test
        # (Para calcular el IDCG perfecto)
        ideal_relevance = np.sort(group['rating'].values)[::-1][:k]

        # 3. Fórmulas Matemáticas (Estado del Arte)
        # DCG = Sumatoria de (2^relevancia - 1) / log2(rank + 1)
        dcg = np.sum((2 ** relevance_pred - 1) / np.log2(np.arange(2, len(relevance_pred) + 2)))
        idcg = np.sum((2 ** ideal_relevance - 1) / np.log2(np.arange(2, len(ideal_relevance) + 2)))

        # Evitamos división por cero si el IDCG es 0 (ej. puras películas de 0 estrellas)
        return dcg / idcg if idcg > 0 else 0.0

    # Aplicamos la función usuario por usuario y promediamos
    ndcg_scores = df_merged.groupby('userId').apply(calculate_ndcg_per_user)
    ndcg_final = ndcg_scores.mean()

    print("-" * 50)
    print(f"🏆 RESULTADO DE RANKING (Estándar Microsoft Recommenders)")
    print(f"   ↳ NDCG@{k}: {ndcg_final:.4f}")
    print(f"   ↳ Evaluado sobre {len(ndcg_scores)} usuarios.")
    print("-" * 50)

    return ndcg_final


from collections import defaultdict
import torch

def evaluate_top_k_recommender(model, test_loader, device, k=10, threshold=3.8):
    """
    Evalúa Precision@K y Recall@K agrupando los resultados por cada usuario.
    """
    model.eval()

    # Diccionario para agrupar las predicciones: {user_id: [(pred_rating, true_rating), ...]}
    user_ratings_comparison = defaultdict(list)

    print("🧠 Generando predicciones y agrupando por usuario...")
    with torch.no_grad():
        for batch in test_loader:
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            genres = batch['genres'].to(device)
            text_emb = batch['text_emb'].to(device)
            img_emb = batch['img_emb'].to(device)

            # Desescalamos a formato de 1 a 5 estrellas
            rating_real_01 = batch['rating'].to(device)
            rating_real_estrellas = (rating_real_01 * 4.0) + 1.0

            # Predicción del modelo
            rating_pred_01, _ = model(user, item, genres, text_emb, img_emb)
            # Usamos view(-1) en lugar de squeeze() para evitar problemas con batches de tamaño 1
            rating_pred_estrellas = (rating_pred_01.view(-1) * 4.0) + 1.0

            # Guardamos las tuplas (predicción, valor_real) agrupadas por usuario
            for u, p, t in zip(user, rating_pred_estrellas, rating_real_estrellas):
                user_ratings_comparison[u.item()].append((p.item(), t.item()))

    def calculate_precision_recall_at_k(user_ratings, k, threshold):
        # 1. ORDENAMOS de mayor a menor según la predicción del modelo (El Carrusel)
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # 2. n_rel: Cuántos ítems son REALMENTE relevantes para el usuario en todo su historial
        n_rel = sum(true_r >= threshold for _, true_r in user_ratings)

        # 3. n_rec_k: Cuántos ítems RECOMENDAMOS en el Top K (que el modelo cree que superan el umbral)
        n_rec_k = sum(pred >= threshold for pred, _ in user_ratings[:k])

        # 4. n_rel_and_rec_k: Cuántos ACERTAMOS en el Top K
        n_rel_and_rec_k = sum((true_r >= threshold) and (pred >= threshold) for pred, true_r in user_ratings[:k])

        # Cálculo de métricas (Evitando división por cero)
        precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precision, recall, n_rel

    user_precisions = []
    user_recalls = []

    print(f"📐 Calculando Precision@{k} y Recall@{k} con umbral de {threshold} estrellas...")

    for user_id, user_ratings in user_ratings_comparison.items():
        precision, recall, n_rel = calculate_precision_recall_at_k(user_ratings, k, threshold)

        # Solo consideramos usuarios que tienen al menos un ítem relevante en su conjunto de test
        # (Es injusto castigar el Recall del modelo si el usuario no tiene ítems buenos para descubrir)
        if n_rel > 0:
            user_precisions.append(precision)
            user_recalls.append(recall)

    # Promedio global de todos los usuarios
    average_precision = sum(user_precisions) / len(user_precisions)
    average_recall = sum(user_recalls) / len(user_recalls)

    print("-" * 50)
    print(f"📊 RESULTADOS FINALES (Top-{k}):")
    print(f"✅ Precision@{k}: {average_precision:.4f} (De lo recomendado en Top-{k}, cuánto le gustó)")
    print(f"🔍 Recall@{k}:    {average_recall:.4f} (De lo que le gusta, cuánto lograste capturar en el Top-{k})")
    print("-" * 50)

    return average_precision, average_recall

######### EJECUCION ############
hr, ndcg = evaluate_rigorous_ranking_reg_true_negatives(
    model=model,
    df_test=df_test,
    df_train=df_train,
    df_valid=df_valid,
    dict_embeddings=catalogo_final,
    device=device,
    k=10,
    num_decoys=99
)