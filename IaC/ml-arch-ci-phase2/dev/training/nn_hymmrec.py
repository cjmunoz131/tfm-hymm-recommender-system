"""
HYMM-REC: Multimodal Explainable GMF Architecture (Shared Module)
=================================================================
Arquitectura Two-Tower con mecanismo de atención multimodal para el sistema
recomendador híbrido. Soporta dos variantes:

  1. MultimodalExplainableGMF (Single-Head):
     - Modo Regresión: Predice rating escalado [0,1] → MSELoss
     - Output: (prediction, attention_weights)

  2. MultimodalExplainableGMF_TwoHeads (Multi-Task):
     - Cabeza 1 (Retrieval): Predice probabilidad de interacción → BCELoss
     - Cabeza 2 (Calidad): Predice rating escalado [0,1] → MSELoss (enmascarado)
     - Entrenamiento conjunto: Loss = BCE + MSE(solo positivos)
     - Output: (prob_interaction, pred_rating, attention_weights)

Componentes compartidos:
  - ModalityAttention: Pesos de atención sobre categoría, texto e imagen
  - UserTower: Embedding + MLP para representación de usuarios
  - ExplainableItemTower: Embedding + Fusión multimodal con atención

Uso:
  # Regresión (single-head)
  from nn_hymmrec import MultimodalExplainableGMF
  model = MultimodalExplainableGMF(num_users, num_items, num_categories)

  # Multi-Task (two-heads)
  from nn_hymmrec import MultimodalExplainableGMF_TwoHeads
  model = MultimodalExplainableGMF_TwoHeads(num_users, num_items, num_categories)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAttention(nn.Module):
    """
    Mecanismo de atención sobre las 3 modalidades de contenido del ítem:
      - Categoría (géneros multi-hot → emb_dim)
      - Texto (embedding Bedrock Nova → aws_dim)
      - Imagen (embedding Bedrock Nova → aws_dim)

    Output: vectores ponderados + pesos de atención (explicabilidad).
    """

    def __init__(self, cat_dim=64, aws_dim=1024):
        super().__init__()
        self.cat_scorer = nn.Linear(cat_dim, 1)
        self.text_scorer = nn.Linear(aws_dim, 1)
        self.img_scorer = nn.Linear(aws_dim, 1)

    def forward(self, cat_vec, text_vec, img_vec):
        score_cat = self.cat_scorer(cat_vec)      # [Batch, 1]
        score_text = self.text_scorer(text_vec)    # [Batch, 1]
        score_img = self.img_scorer(img_vec)      # [Batch, 1]

        scores = torch.cat([score_cat, score_text, score_img], dim=1)  # [Batch, 3]
        weights = F.softmax(scores, dim=1)                              # [Batch, 3]

        w_cat = weights[:, 0:1]
        w_text = weights[:, 1:2]
        w_img = weights[:, 2:3]

        cat_weighted = cat_vec * w_cat
        text_weighted = text_vec * w_text
        img_weighted = img_vec * w_img

        return cat_weighted, text_weighted, img_weighted, weights


class UserTower(nn.Module):
    """Torre de usuario: Embedding → MLP → representación de usuario."""

    def __init__(self, num_users, emb_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

    def forward(self, user_id):
        return self.mlp(self.user_embedding(user_id))


class ExplainableItemTower(nn.Module):
    """
    Torre de ítem con fusión multimodal explicable:
      1. Item embedding (collaborative filtering signal)
      2. Categoría multi-hot → embedding
      3. Atención sobre (categoría, texto, imagen)
      4. Fusión de contenido ponderado
      5. Fusión final: item CF + item content
    """

    def __init__(self, num_items, num_categories, emb_dim=64, aws_dim=1024, dropout=0.3):
        super().__init__()

        # --- Entradas Sparse ---
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

        self.cat_mlp = nn.Sequential(
            nn.Linear(num_categories, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

        # Módulo de atención multimodal
        self.attention_layer = ModalityAttention(cat_dim=emb_dim, aws_dim=aws_dim)

        # --- Fusión de Contenido ---
        content_input_dim = emb_dim + aws_dim + aws_dim

        self.content_mlp = nn.Sequential(
            nn.Linear(content_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
        )

        # --- Fusión Final del Ítem ---
        final_item_dim = emb_dim + 256

        self.final_mlp = nn.Sequential(
            nn.Linear(final_item_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
        )

    def forward(self, item_id, cat_id, text_emb, img_emb):
        emb_i = self.item_mlp(self.item_embedding(item_id))
        emb_c = self.cat_mlp(cat_id)

        # Atención multimodal
        emb_c_w, text_emb_w, img_emb_w, attention_weights = self.attention_layer(
            emb_c, text_emb, img_emb
        )

        # Fusión de contenido ponderado
        content_concat = torch.cat([emb_c_w, text_emb_w, img_emb_w], dim=1)
        item_content_vector = self.content_mlp(content_concat)

        # Fusión final
        final_concat = torch.cat([emb_i, item_content_vector], dim=1)
        mf_item_vector = self.final_mlp(final_concat)

        return mf_item_vector, attention_weights


class MultimodalExplainableGMF(nn.Module):
    """
    Generalized Matrix Factorization con Two-Tower multimodal explicable.
    Single-Head: una sola cabeza de predicción (regresión de rating).

    Args:
        num_users: Número total de usuarios (vocabulario embedding)
        num_items: Número total de ítems (vocabulario embedding)
        num_categories: Dimensión del vector multi-hot de géneros
        emb_dim: Dimensión de embeddings internos (default: 64)
        aws_dim: Dimensión de embeddings Bedrock Nova (default: 1024)
        dropout: Dropout rate base (default: 0.3)

    Output:
        (prediction, attention_weights)
        - prediction: sigmoid(logits) ∈ [0, 1] (rating escalado)
        - attention_weights: [batch, 3] → importancia de (categoría, texto, imagen)
    """

    def __init__(self, num_users, num_items, num_categories, emb_dim=64, aws_dim=1024, dropout=0.3):
        super().__init__()
        self.user_tower = UserTower(num_users, emb_dim)
        self.item_tower = ExplainableItemTower(
            num_items, num_categories, emb_dim, aws_dim, dropout
        )
        self.gmf_layer = nn.Linear(emb_dim, 1)

    def forward(self, user_id, item_id, cat_id, text_emb, img_emb):
        mf_user_vector = self.user_tower(user_id)
        mf_item_vector, attention_weights = self.item_tower(item_id, cat_id, text_emb, img_emb)

        # Interacción GMF (producto elemento a elemento)
        interaction = mf_user_vector * mf_item_vector

        # Predicción
        logits = self.gmf_layer(interaction)
        prediction = torch.sigmoid(logits)

        return prediction, attention_weights


class MultimodalExplainableGMF_TwoHeads(nn.Module):
    """
    Multi-Task Learning: Two-Heads sobre la misma representación GMF.

    Cabeza 1 (Retrieval/Ranking):
      - Predice probabilidad de interacción ∈ [0, 1]
      - Loss: BCELoss sobre TODOS los datos (positivos + negativos)
      - Objetivo: Aprender qué recomendar (ranking)

    Cabeza 2 (Calidad/Rating):
      - Predice rating escalado ∈ [0, 1]
      - Loss: MSELoss ENMASCARADO (solo sobre interacciones positivas)
      - Objetivo: Aprender cuánto le gustará (quality estimation)

    Entrenamiento conjunto:
      - Total Loss = BCE(head1) + MSE(head2, solo positivos)
      - El gradiente fluye por ambas cabezas hacia el backbone compartido
      - Las torres aprenden representaciones que sirven para ambos objetivos

    Args:
        num_users: Número total de usuarios
        num_items: Número total de ítems
        num_categories: Dimensión del vector multi-hot de géneros
        emb_dim: Dimensión de embeddings internos (default: 64)
        aws_dim: Dimensión de embeddings Bedrock Nova (default: 1024)
        dropout: Dropout rate base (default: 0.3)

    Output:
        (prob_interaction, pred_rating, attention_weights)
        - prob_interaction: sigmoid ∈ [0, 1] → probabilidad de clic/interacción
        - pred_rating: sigmoid ∈ [0, 1] → rating escalado predicho
        - attention_weights: [batch, 3] → explicabilidad multimodal
    """

    def __init__(self, num_users, num_items, num_categories, emb_dim=64, aws_dim=1024, dropout=0.3):
        super().__init__()
        self.user_tower = UserTower(num_users, emb_dim)
        self.item_tower = ExplainableItemTower(
            num_items, num_categories, emb_dim, aws_dim, dropout
        )

        # Dos cabezas independientes sobre el mismo vector de interacción
        self.head_interaction = nn.Linear(emb_dim, 1)  # Retrieval: P(interacción)
        self.head_rating = nn.Linear(emb_dim, 1)       # Calidad: rating escalado

    def forward(self, user_id, item_id, cat_id, text_emb, img_emb):
        mf_user_vector = self.user_tower(user_id)
        mf_item_vector, attention_weights = self.item_tower(item_id, cat_id, text_emb, img_emb)

        # Interacción GMF (producto elemento a elemento) — backbone compartido
        interaction_vector = mf_user_vector * mf_item_vector

        # Cabeza 1: Probabilidad de Interacción (Ranking/Retrieval)
        logits_interaction = self.head_interaction(interaction_vector)
        prob_interaction = torch.sigmoid(logits_interaction)

        # Cabeza 2: Predicción de Rating (Calidad)
        logits_rating = self.head_rating(interaction_vector)
        pred_rating = torch.sigmoid(logits_rating)

        return prob_interaction, pred_rating, attention_weights
