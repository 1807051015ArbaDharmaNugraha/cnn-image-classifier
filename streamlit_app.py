import streamlit as st
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Klasifikasi Gambar - Dummy", layout="centered")
st.title("Klasifikasi Gambar (Dummy Model)")
st.write("Demo app menggunakan dummy model. Labels: Mobil, Motor, Sepeda")

MODEL_PATH = os.path.join('models', 'model_cnn.h5')
LABELS_PATH = os.path.join('models', 'class_indices.npy')

if not os.path.exists(MODEL_PATH):
    st.error("Model belum ada. Silakan taruh file models/model_cnn.h5 di repo.")
    st.stop()

model = None
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.warning("Model dummy tidak bisa diload di runtime ini, akan pakai prediksi acak. Error: " + str(e))

labels_map = {0: "Mobil", 1: "Motor", 2: "Sepeda"}

uploaded_file = st.file_uploader("Upload gambar (.jpg/.png)", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar input', use_column_width=True)

    if st.button("Predict"):
        if model is not None:
            img = image.resize((64, 64))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            preds = model.predict(arr)[0]
            top_idx = int(np.argmax(preds))
            label = labels_map[top_idx]
            st.success(f"Prediksi: **{label}** ({preds[top_idx]*100:.2f}% confidence)")
            st.write("Probabilitas tiap kelas:")
            for i, p in enumerate(preds):
                st.write(f"- {labels_map[i]}: {p*100:.2f}%")
        else:
            import random
            top_idx = random.randint(0, 2)
            label = labels_map[top_idx]
            prob = np.round(np.random.uniform(0.7, 0.99), 2)
            st.success(f"Prediksi (dummy): **{label}** ({prob*100:.2f}% confidence)")
            st.info("Ini hasil acak karena model dummy belum aktif.")