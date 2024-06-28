import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from streamlit_drawable_canvas import st_canvas

def draw_digit():
    st.sidebar.title("Dessinez un chiffre")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=10,
        stroke_color="#000000", 
        background_color="#FFFFFF",
        update_streamlit=True,
        height=500,
        width=500,
        drawing_mode="freedraw",
        key="canvas",
    )

    return canvas_result

def send_prediction_request(image):
    img = Image.fromarray(image.astype('uint8'))

    img = img.resize((28, 28)) 

    byte_io = io.BytesIO()
    img.save(byte_io, format='PNG')
    byte_io.seek(0)

    api_url = 'http://backend:8000/api/v1/predict'    

    files = {'file': byte_io}
    response = requests.post(api_url, files=files)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.sidebar.success(f"Prédiction: {prediction}")
    else:
        st.sidebar.error("Erreur lors de la prédiction. Veuillez réessayer.")

def main():
    st.title("Application de reconnaissance de chiffres manuscrits")
    st.sidebar.title("Paramètres")

    canvas_result = draw_digit()

    if st.sidebar.button("Prédire") and canvas_result is not None:
        image_array = np.array(canvas_result.image_data)
        if image_array.max() > 0:
            send_prediction_request(image_array)
        else:
            st.sidebar.warning("Veuillez dessiner un chiffre avant de prédire.")

    st.sidebar.info("Dessinez un chiffre dans le canvas pour commencer.")

if __name__ == "__main__":
    main()
