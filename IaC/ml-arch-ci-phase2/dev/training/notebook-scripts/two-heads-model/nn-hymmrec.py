import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityAttention(nn.Module):
    def __init__(self, cat_dim=64, aws_dim=1024):
        super().__init__()
        # Evaluadores independientes para cada tipo de dato
        self.cat_scorer = nn.Linear(cat_dim, 1)
        self.text_scorer = nn.Linear(aws_dim, 1)
        self.img_scorer = nn.Linear(aws_dim, 1)

    def forward(self, cat_vec, text_vec, img_vec):
        # 1. ¿Qué tan importante es cada vector para esta película?
        score_cat = self.cat_scorer(cat_vec)    # [Batch, 1]
        score_text = self.text_scorer(text_vec) # [Batch, 1]
        score_img = self.img_scorer(img_vec)    # [Batch, 1]

        # 2. Unimos los scores y aplicamos Softmax (para que sumen 1.0)
        scores = torch.cat([score_cat, score_text, score_img], dim=1) # [Batch, 3]
        weights = F.softmax(scores, dim=1)                            # [Batch, 3]

        # 3. Extraemos los pesos individuales (Explicabilidad)
        w_cat = weights[:, 0:1]
        w_text = weights[:, 1:2]
        w_img = weights[:, 2:3]

        # 4. Multiplicamos (ponderamos) los vectores originales por su peso
        # Si el texto tiene 80% de peso, su vector pasará casi intacto.
        # Si la imagen tiene 5% de peso, sus valores se reducirán casi a cero.
        cat_weighted = cat_vec * w_cat
        text_weighted = text_vec * w_text
        img_weighted = img_vec * w_img

        return cat_weighted, text_weighted, img_weighted, weights
    
class UserTower(nn.Module):
    def __init__(self, num_users, emb_dim=64):
        super().__init__()
        # Sparse Vector -> Embedding Product ID
        self.user_embedding = nn.Embedding(num_users, emb_dim)

        # MLP Layer + ReLU
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim), # Normalización de la capa
            nn.ReLU()
        )

    def forward(self, user_id):
        return self.mlp(self.user_embedding(user_id))
    
class ExplainableItemTower(nn.Module):
    def __init__(self, num_items, num_categories, emb_dim=64, aws_dim=1024):
        super().__init__()

        # --- Entradas Sparse ---
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU()
        )

        self.cat_mlp = nn.Sequential(
            nn.Linear(num_categories, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU()
        )

        # Instanciamos el módulo de atención
        self.attention_layer = ModalityAttention(cat_dim=emb_dim, aws_dim=aws_dim)

        # --- Fusión de Contenido (Símbolo ⨁ inferior) ---
        # Concatena: Categoría (emb_dim) + Texto (1024) + Imagen (1024)
        content_input_dim = emb_dim + aws_dim + aws_dim

        self.content_mlp = nn.Sequential(
            # Capa 1: Transición inicial (2112 -> 1024)
            nn.Linear(content_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Capa 2: Transición media (1024 -> 512)
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Capa 3: Salida final madura (512 -> 256)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # --- Fusión Final del Ítem (Símbolo ⨁ central) ---
        # Concatena: Embedding Product ID (emb_dim) + Item Content Vector (256)
        final_item_dim = emb_dim + 256

        self.final_mlp = nn.Sequential(
            nn.Linear(final_item_dim, emb_dim),
            nn.BatchNorm1d(emb_dim), # Normalización final
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, item_id, cat_id, text_emb, img_emb):
        emb_i = self.item_mlp(self.item_embedding(item_id))
        emb_c = self.cat_mlp(cat_id)

        # Evaluamos y ponderamos antes de concatenar
        emb_c_w, text_emb_w, img_emb_w, attention_weights = self.attention_layer(emb_c, text_emb, img_emb)

        # Usamos los vectores PONDERADOS para la concatenación
        content_concat = torch.cat([emb_c_w, text_emb_w, img_emb_w], dim=1)

        # Pasa por tu embudo sin problemas
        item_content_vector = self.content_mlp(content_concat)

        final_concat = torch.cat([emb_i, item_content_vector], dim=1)
        mf_item_vector = self.final_mlp(final_concat)

        # Devolvemos el vector del ítem Y los pesos de explicabilidad
        return mf_item_vector, attention_weights
    
class MultimodalExplainableGMF(nn.Module):
    def __init__(self, num_users, num_items, num_categories, emb_dim=64, aws_dim=1024):
        super().__init__()
        self.user_tower = UserTower(num_users, emb_dim)
        self.item_tower = ExplainableItemTower(num_items, num_categories, emb_dim, aws_dim)

        # 🌟 DOS CABEZAS (Multi-Task Learning)
        self.head_interaction = nn.Linear(emb_dim, 1) # Predice Probabilidad (Clic)
        self.head_rating = nn.Linear(emb_dim, 1)      # Predice Estrellas (Calidad)

    def forward(self, user_id, item_id, cat_id, text_emb, img_emb):
        mf_user_vector = self.user_tower(user_id)
        mf_item_vector, attention_weights = self.item_tower(item_id, cat_id, text_emb, img_emb)

        # Fusión base
        interaction_vector = mf_user_vector * mf_item_vector

        # Cabeza 1: Probabilidad de Interacción (Sigmoide)
        logits_int = self.head_interaction(interaction_vector)
        prob_interaction = torch.sigmoid(logits_int)

        # Cabeza 2: Predicción de Rating (Sigmoide porque rating_scaled va de 0 a 1)
        logits_rat = self.head_rating(interaction_vector)
        pred_rating = torch.sigmoid(logits_rat)

        # Ahora devolvemos 3 cosas
        return prob_interaction, pred_rating, attention_weights