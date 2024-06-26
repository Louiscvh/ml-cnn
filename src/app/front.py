import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Fonction pour dessiner un chiffre manuscrit
def draw_digit():
    st.sidebar.title("Dessinez un chiffre")

    # Créer un canvas pour le dessin
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Couleur de remplissage du canvas (orange transparent)
        stroke_width=10,  # Largeur du trait du pinceau
        stroke_color="#000000",  # Couleur du trait du pinceau (noir)
        background_color="#FFFFFF",  # Couleur de fond du canvas (blanc)
        update_streamlit=True,
        height=500,  # Hauteur du canvas en pixels
        width=500,  # Largeur du canvas en pixels
        drawing_mode="freedraw",  # Mode de dessin libre
        key="canvas",
    )

    return canvas_result

# Fonction pour envoyer l'image à l'API pour prédiction
def send_prediction_request(image):
    # Préparation de l'image pour l'envoi
    img = Image.fromarray(image.astype('uint8'))  # Convertir l'image numpy en image PIL

    img = img.resize((28, 28))  # Redimensionner l'image au format attendu par le modèle

    # Convertir l'image en bytes pour l'envoi via l'API
    byte_io = io.BytesIO()
    img.save(byte_io, format='PNG')
    byte_io.seek(0)

    # URL de l'API FastAPI pour la prédiction (remplacez par votre propre URL)
    api_url = 'http://127.0.0.1:8000/api/v1/predict'

    # Envoyer la requête POST à l'API
    files = {'file': byte_io}
    response = requests.post(api_url, files=files)

    # Gérer la réponse de l'API
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.sidebar.success(f"Prédiction: {prediction}")
    else:
        st.sidebar.error("Erreur lors de la prédiction. Veuillez réessayer.")

# Fonction principale pour exécuter l'interface
def main():
    st.title("Application de reconnaissance de chiffres manuscrits")
    st.sidebar.title("Paramètres")

    # Dessiner un chiffre
    canvas_result = draw_digit()

    # Bouton pour prédire basé sur le dessin
    if st.sidebar.button("Prédire") and canvas_result is not None:
        # Convertir le canvas en image pour l'envoi à l'API
        image_array = np.array(canvas_result.image_data)
        if image_array.max() > 0:  # Vérifier si le canvas contient des données de dessin
            send_prediction_request(image_array)
        else:
            st.sidebar.warning("Veuillez dessiner un chiffre avant de prédire.")

    st.sidebar.info("Dessinez un chiffre dans le canvas pour commencer.")

if __name__ == "__main__":
    main()
