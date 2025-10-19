import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ===============================================
# Judul dan deskripsi aplikasi
# ===============================================
st.title("Klasifikasi Gambar Kendaraan (Model CNN Asli)")
st.write("""
Aplikasi ini menggunakan model CNN hasil training untuk mengklasifikasikan gambar **Mobil**, **Motor**, dan **Sepeda**.
Upload gambar dan lihat hasil prediksinya!
""")

# ===============================================
# Load model dan label class
# ===============================================
@st.cache_resource
def load_my_model():
    try:
        model = load_model("models/model_cnn.h5")
        class_indices = np.load("models/class_indices.npy", allow_pickle=True).item()
        labels = list(class_indices.keys())
        st.success("[V] Model berhasil dimuat.")
        return model, labels
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

model, labels = load_my_model()

# ===============================================
# Upload gambar
# ===============================================
uploaded_file = st.file_uploader(
    "Upload gambar (.jpg/.jpeg/.png)",
    type=["jpg", "jpeg", "png"]
)

# ===============================================
# Proses prediksi
# ===============================================
if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang di-upload", use_column_width=True)

    if model is not None:
        # Preprocessing gambar agar cocok dengan model
        img = img.resize((128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi
        preds = model.predict(img_array)
        pred_label = labels[np.argmax(preds)]
        confidence = np.max(preds) * 100

        # Tampilkan hasil
        st.subheader("🔍 Hasil Prediksi")
        st.write(f"**Label:** {pred_label}")
        st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")
    else:
        st.warning("Model belum dimuat, tidak bisa melakukan prediksi.")
else:
    st.info("Silakan upload gambar terlebih dahulu.")
