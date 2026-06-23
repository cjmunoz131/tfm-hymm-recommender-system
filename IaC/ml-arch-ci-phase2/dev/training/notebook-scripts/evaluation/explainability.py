import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random

def explicar_recomendacion(model, user_id_idx, movie_id_idx, dict_aws, df_peliculas, device):
    """
    df_peliculas: Tu dataframe original de Test que tiene los mapeos y nombres.
    """
    model.eval()
    with torch.no_grad():
        # 1. Buscar la fila exacta de esta película en el DataFrame
        fila_pelicula = df_peliculas[df_peliculas['movieId_idx'] == movie_id_idx].iloc[0]

        # 2. EL PUENTE (Traducción de IDs)
        original_movie_id = fila_pelicula['movieId']
        nombre_pelicula = fila_pelicula['title'] if 'title' in fila_pelicula else f"Película {original_movie_id}"

        # 3. Extraer los Géneros desde el DataFrame (como lo hace tu Dataset)
        generos_array = fila_pelicula['genres_multihot']

        # 4. Preparar tensores
        u = torch.tensor([user_id_idx]).to(device)
        i = torch.tensor([movie_id_idx]).to(device)

        # 5. Cargar tensores multimodales (Usando el ID Original para el dict_aws)
        c = torch.tensor(generos_array).unsqueeze(0).float().to(device)
        t = torch.tensor(dict_aws[original_movie_id]['text_emb']).unsqueeze(0).float().to(device)
        img = torch.tensor(dict_aws[original_movie_id]['img_emb']).unsqueeze(0).float().to(device)

        # 6. Inferencia
        pred, pesos = model(u, i, c, t, img)

        # Como tu modelo entrena con MSE directo escalado de 0 a 1, si quieres ver de 1 a 5 estrellas:
        pred_estrellas = pred.item() * 4.0 + 1.0

        # 7. Mostrar Resultados en Consola
        print(f"🎬 Película: {nombre_pelicula}")
        print(f"⭐ Predicción para el Usuario {user_id_idx}: {pred_estrellas:.2f} Estrellas")
        print("-" * 30)
        print(f"🧠 Pesos de Decisión (Explicabilidad):")
        print(f"   - Categoría: {pesos[0][0].item()*100:.1f}%")
        print(f"   - Sinopsis:  {pesos[0][1].item()*100:.1f}%")
        print(f"   - Póster:    {pesos[0][2].item()*100:.1f}%")

        # 8. Gráfico de barras rápido para tu TFM
        nombres = ['Géneros (Tabular)', 'Sinopsis (Texto Nova)', 'Póster (Imagen Nova)']
        valores = [pesos[0][0].item(), pesos[0][1].item(), pesos[0][2].item()]

        plt.figure(figsize=(8, 4))
        sns.barplot(x=valores, y=nombres, palette="viridis")
        plt.title(f"Explicabilidad de Recomendación:\n{nombre_pelicula}", fontweight='bold')
        plt.xlabel("Peso de Importancia (α)")
        plt.xlim(0, 1)

        # Añadir las etiquetas de porcentaje en las barras
        for idx_val, val in enumerate(valores):
            plt.text(val + 0.01, idx_val, f'{val*100:.1f}%', va='center', fontsize=10)

        plt.tight_layout()
        plt.show()
        
########## EJECUCIÓN ###########
indice_aleatorio = random.randint(0, len(df_test) - 1)
fila_prueba = df_test.iloc[indice_aleatorio]

usuario_prueba = int(fila_prueba['userId_idx'])
pelicula_prueba = int(fila_prueba['movieId_idx'])

# Aseguramos el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OJO: Aquí asumo que la variable de tu diccionario se llama 'catalogo_final'
# Si en tu código se llama 'dict_aws', cámbialo en los argumentos.
explicar_recomendacion(
    model=model,
    user_id_idx=usuario_prueba,
    movie_id_idx=pelicula_prueba,
    dict_aws=catalogo_final, # <-- Asegúrate de que este es el nombre de tu diccionario de embeddings
    df_peliculas=df_test,    # Le pasamos el df_test para intentar buscar el nombre
    device=device
)