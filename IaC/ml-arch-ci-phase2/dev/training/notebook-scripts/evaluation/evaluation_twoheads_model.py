import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def evaluate_rigorous_ranking_mtl_true_negatives(model, df_test, df_train, df_valid, dict_embeddings, device, k=10, num_decoys=99, threshold=3.5, bad_threshold=2.5):
    """
    Evaluación rigurosa MTL usando VERDADEROS NEGATIVOS.
    Garantiza que los señuelos sean películas explícitamente mal calificadas (<= bad_threshold)
    en la base de datos, evitando penalizar Falsos Negativos (películas buenas no vistas).
    """
    print(f"📐 Iniciando Evaluación MTL con VERDADEROS NEGATIVOS (Top-{k} con {num_decoys} señuelos MALOS)...")
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

    # 🌟 2. EL CAMBIO MAESTRO: Construimos la piscina global de "Películas Malas"
    # Tomamos todas las películas que alguien calificó con 1 o 2 estrellas.
    df_malas = df_all[df_all['rating_scaled'] <= (bad_threshold - 1.0)/4.0]
    bad_item_pool = set(df_malas['movieId'].unique())

    print(f"   ↳ Piscina de Señuelos Malos detectada: {len(bad_item_pool)} películas.")

    usuarios_test = df_test_positives['userId'].unique()

    with torch.no_grad():
        for user_id in tqdm(usuarios_test, desc="Evaluando Usuarios (MTL - True Negatives)"):
            user_idx = df_all[df_all['userId'] == user_id]['userId_idx'].iloc[0]
            user_test_movies = df_test_positives[df_test_positives['userId'] == user_id]['movieId'].values

            historial = global_history.get(user_id, set())

            # 🌟 3. SELECCIÓN DE SEÑUELOS ESTRICTOS
            # Filtramos la piscina de malas quitando lo que el usuario ya vio
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

                # Extraemos ambas cabezas
                prob_interaction, pred_rating_scaled, _ = model(batch_users, batch_items_idx, batch_genres, batch_text, batch_img)

                # Puntaje Híbrido
                prob_int = prob_interaction.view(-1).cpu().numpy()
                pred_rat = pred_rating_scaled.view(-1).cpu().numpy()
                hybrid_scores = prob_int * pred_rat

                # Métrica
                rankings = (-hybrid_scores).argsort().argsort()
                target_rank = rankings[0]

                if target_rank < k:
                    hits += 1
                    ndcg_sum += 1.0 / math.log2(target_rank + 2)

                total_evaluations += 1

    hit_rate = hits / total_evaluations if total_evaluations > 0 else 0
    ndcg_final = ndcg_sum / total_evaluations if total_evaluations > 0 else 0

    print("\n" + "-" * 50)
    print(f"🏆 RESULTADOS MTL (Top-{k} vs {num_decoys} Verdaderos Negativos)")
    print(f"   ↳ Hit Rate@{k}: {hit_rate:.4f}")
    print(f"   ↳ NDCG@{k}:     {ndcg_final:.4f}")
    print("-" * 50)

    return hit_rate, ndcg_final


def evaluar_clasificacion_recomendador_mtl(model, test_loader, device, umbral=4.0):
    """
    Evalúa la capacidad de la Cabeza de Regresión (MSE) del modelo Multi-Task
    para clasificar correctamente películas de alta calidad (>= umbral).
    """
    model.eval()

    todas_predicciones = []
    todos_reales = []

    print(f"🧠 Generando predicciones de Calidad (Cabeza MSE) para matriz de confusión...")
    with torch.no_grad():
        for batch in test_loader:
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            genres = batch['genres'].to(device)
            text_emb = batch['text_emb'].to(device)
            img_emb = batch['img_emb'].to(device)

            # Rating real de la base de datos (0-1 a 1-5)
            rating_real_01 = batch['rating'].to(device)
            rating_real_estrellas = (rating_real_01 * 4.0) + 1.0

            # 🌟 EL CAMBIO MAESTRO: Desempaquetamos las 3 salidas del modelo MTL
            # Ignoramos la probabilidad de clic y la atención, nos quedamos con el rating
            _, rating_pred_01, _ = model(user, item, genres, text_emb, img_emb)

            # Aplanamos de forma segura con view(-1) y convertimos a estrellas
            rating_pred_estrellas = (rating_pred_01.view(-1) * 4.0) + 1.0

            # Guardamos los resultados
            todas_predicciones.extend(rating_pred_estrellas.cpu().numpy())
            todos_reales.extend(rating_real_estrellas.cpu().numpy())

    # BINARIZACIÓN: >= umbral es "Relevante" (1), < umbral es "Irrelevante" (0)
    reales_binario = [1 if r >= umbral else 0 for r in todos_reales]
    preds_binario = [1 if p >= umbral else 0 for p in todas_predicciones]

    # 1. Reporte de Clasificación (Precision, Recall, F1)
    print(f"\n📊 REPORTE DE CLASIFICACIÓN MTL (Umbral de Relevancia: >= {umbral} Estrellas)")
    print("-" * 65)
    print(classification_report(reales_binario, preds_binario, target_names=[f'Irrelevante (<{umbral})', f'Relevante (>={umbral})']))

    # 2. Matriz de Confusión Visual
    cm = confusion_matrix(reales_binario, preds_binario)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: Irrelevante', 'Pred: Relevante'],
                yticklabels=['Real: Irrelevante', 'Real: Relevante'])
    plt.title('Matriz de Confusión - Cabeza de Calidad (MTL)', fontweight='bold')
    plt.show()

from collections import defaultdict
import torch

def evaluate_top_k_recommender_mtl(model, test_loader, device, k=10, threshold=3.5):
    """
    Evalúa Precision@K y Recall@K para la cabeza de regresión (Multi-Task Learning).
    """
    model.eval()
    user_ratings_comparison = defaultdict(list)

    print("🧠 Generando predicciones (Cabeza MSE) y agrupando por usuario...")
    with torch.no_grad():
        for batch in test_loader:
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            genres = batch['genres'].to(device)
            text_emb = batch['text_emb'].to(device)
            img_emb = batch['img_emb'].to(device)

            rating_real_01 = batch['rating'].to(device)
            rating_real_estrellas = (rating_real_01 * 4.0) + 1.0

            # 🌟 EL CAMBIO MAESTRO: Desempaquetamos 3 variables.
            # Tomamos 'rating_pred_01' (Cabeza 2) y tiramos el resto a '_'
            _, rating_pred_01, _ = model(user, item, genres, text_emb, img_emb)

            rating_pred_estrellas = (rating_pred_01.view(-1) * 4.0) + 1.0

            for u, p, t in zip(user, rating_pred_estrellas, rating_real_estrellas):
                user_ratings_comparison[u.item()].append((p.item(), t.item()))

    def calculate_precision_recall_at_k(user_ratings, k, threshold):
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
        n_rec_k = sum(pred >= threshold for pred, _ in user_ratings[:k])
        n_rel_and_rec_k = sum((true_r >= threshold) and (pred >= threshold) for pred, true_r in user_ratings[:k])

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precision, recall, n_rel

    user_precisions = []
    user_recalls = []

    print(f"📐 Calculando Precision@{k} y Recall@{k} con umbral de {threshold} estrellas...")

    for user_id, user_ratings in user_ratings_comparison.items():
        precision, recall, n_rel = calculate_precision_recall_at_k(user_ratings, k, threshold)
        if n_rel > 0:
            user_precisions.append(precision)
            user_recalls.append(recall)

    average_precision = sum(user_precisions) / len(user_precisions) if user_precisions else 0
    average_recall = sum(user_recalls) / len(user_recalls) if user_recalls else 0

    print("-" * 50)
    print(f"📊 RESULTADOS FINALES (Top-{k} - Calidad MTL):")
    print(f"✅ Precision@{k}: {average_precision:.4f} (De lo que predijo como 'Bueno', cuánto le gustó)")
    print(f"🔍 Recall@{k}:    {average_recall:.4f} (De lo que le gusta, cuánto logró capturar)")
    print("-" * 50)

    return average_precision, average_recall

###### EJECUCIÓN ###
hr, ndcg = evaluate_rigorous_ranking_mtl_true_negatives(
    model=model,
    df_test=df_test,
    df_train=df_train,
    df_valid=df_valid,
    dict_embeddings=catalogo_final,
    device=device,
    k=10,
    num_decoys=99
)