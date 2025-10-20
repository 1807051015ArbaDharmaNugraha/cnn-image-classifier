import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ===============================================
# Judul dan deskripsi aplikasi
# ===============================================
st.set_page_config(page_title="CNN Image Classifier", layout="wide")
st.title("Klasifikasi Gambar Kendaraan (Model CNN Otomatis)")
st.write("""
Pilih gambar di sidebar dan aplikasi akan menyesuaikan preprocessing dengan input model secara otomatis.
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
# Sidebar untuk upload gambar
# ===============================================
st.sidebar.header("Pilih Gambar")
uploaded_file = st.sidebar.file_uploader(
    "Upload gambar (.jpg/.jpeg/.png)",
    type=["jpg", "jpeg", "png"]
)

# ===============================================
# Lakukan prediksi
# ===============================================
if uploaded_file is not None and model is not None:
    # Buka gambar
    img = Image.open(uploaded_file)

    # ===============================================
    # Ambil input shape dari model
    # ===============================================
    input_shape = model.input_shape  # contoh: (None, 32, 32, 3) atau (None, 2304)
    
    if len(input_shape) == 4:
        # Model CNN 2D
        _, height, width, channels = input_shape

        # Sesuaikan warna
        if channels == 1:
            img = img.convert("L")  # grayscale
        else:
            img = img.convert("RGB")  # RGB

        # Resize sesuai model
        img_resized = img.resize((width, height))

        # Konversi ke array dan normalisasi
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

    elif len(input_shape) == 2:
        # Model dense yang menerima 1D input
        _, input_len = input_shape
        img = img.convert("L")  # biasnya grayscale
        # resize agar total pixel sesuai
        side = int(np.sqrt(input_len))
        img_resized = img.resize((side, side))
        img_array = image.img_to_array(img_resized).flatten()
        img_array = np.expand_dims(img_array, axis=0) / 255.0
    else:
        st.error("Model memiliki input shape tidak dikenal!")
        st.stop()

    # Tampilkan gambar di halaman utama
    st.image(img, caption="Gambar yang di-upload", use_container_width=True)

    # Prediksi
    try:
        preds = model.predict(img_array)
        pred_label = labels[np.argmax(preds)]
        confidence = np.max(preds) * 100

        st.subheader("Hasil Prediksi")
        st.write(f"**Label:** {pred_label}")
        st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
else:
    st.info("Silakan pilih gambar terlebih dahulu di sidebar dan pastikan model berhasil dimuat.")
