
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Batik Vision",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. HTML RENDER FUNCTION
# ==========================================

def render_html(content):
    # Menghapus indentasi pada setiap baris HTML
    clean_content = "\n".join(
        line.lstrip() for line in content.splitlines()
    )

    st.markdown(
        clean_content,
        unsafe_allow_html=True
    )


# ==========================================
# 3. CUSTOM CSS
# ==========================================

render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

.stApp {
    background-color: #F7F2EA;
    font-family: 'DM Sans', sans-serif;
    color: #49372E;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #55382C !important;
}

.hero {
    background: linear-gradient(120deg, #E9D8C5, #F1E8DC);
    border: 1px solid #E2D0BD;
    padding: 35px;
    border-radius: 22px;
    margin-bottom: 28px;
}

.hero-label {
    color: #A15C42;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 38px;
    font-weight: 700;
    color: #55382C;
    line-height: 1.2;
    margin: 12px 0;
}

.hero-desc {
    color: #765F50;
    font-size: 15px;
    line-height: 1.7;
    max-width: 650px;
}

.section-label {
    color: #A15C42;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.panel {
    background: #FFFCF7;
    border: 1px solid #E9DED1;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
}

.result-card {
    background: #FFFCF7;
    border: 1px solid #E7D8C8;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
}

.result-name {
    font-family: 'Playfair Display', serif;
    font-size: 29px;
    color: #55382C;
    font-weight: 700;
}

.confidence {
    color: #A15C42;
    font-size: 38px;
    font-weight: 700;
    margin: 8px 0;
}

.status-good {
    background: #E4EEE3;
    color: #416A49;
    padding: 9px 14px;
    border-radius: 9px;
    display: inline-block;
    font-size: 13px;
    font-weight: 600;
}

.status-unknown {
    background: #F5E6D8;
    color: #9A5738;
    padding: 9px 14px;
    border-radius: 9px;
    display: inline-block;
    font-size: 13px;
    font-weight: 600;
}

.stButton > button {
    background: #92543D;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover {
    background: #75412F;
    color: white;
    border: none;
}

div[data-testid="stFileUploader"] {
    background: #F9F4ED;
    border: 1px dashed #B9957D;
    border-radius: 14px;
    padding: 15px;
}

div[data-testid="stProgressBar"] > div > div {
    background-color: #A15C42;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""")


# ==========================================
# 4. MODEL CONFIGURATION
# ==========================================

class_names = [
    "batik_kawung",
    "batik_parang",
    "batik_sidomukti",
    "batik_truntum"
]

THRESHOLD = 60.0

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_batik_mobilenetv2.keras"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()


# ==========================================
# 5. SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("## 🧵 BATIK VISION")
    st.caption("Indonesian Heritage Recognition")

    st.divider()

    st.markdown("### Tentang Aplikasi")
    st.write(
        "Batik Vision merupakan aplikasi klasifikasi "
        "empat motif batik Jawa Tengah berbasis Computer Vision."
    )

    st.markdown("### Teknologi")
    st.markdown("""
    - Computer Vision
    - Convolutional Neural Network
    - MobileNetV2
    - Transfer Learning
    - TensorFlow
    - Streamlit
    """)

    st.divider()

    st.markdown("### Kategori Motif")
    st.write("01 · Batik Kawung")
    st.write("02 · Batik Parang")
    st.write("03 · Batik Sidomukti")
    st.write("04 · Batik Truntum")

    st.divider()
    st.caption("Batik Vision © 2026")


# ==========================================
# 6. HERO SECTION
# ==========================================

render_html("""
<div class="hero">
<div class="hero-label">
Indonesian Cultural Heritage
</div>
<div class="hero-title">
Kenali Motif Batik Jawa Tengah.
</div>
<div class="hero-desc">
Unggah gambar batik dan temukan jenis motifnya
menggunakan teknologi kecerdasan buatan.
Sistem ini mengenali empat motif batik Jawa Tengah,
yaitu Kawung, Parang, Sidomukti, dan Truntum.
</div>
</div>
""")


# ==========================================
# 7. MAIN CONTENT
# ==========================================

left, right = st.columns([1.05, 0.95], gap="large")


with left:
    render_html("""
    <div class="section-label">
    01 — IMAGE INPUT
    </div>
    """)

    st.markdown("### Unggah Gambar Batik")

    st.write(
        "Pilih gambar motif batik dari perangkat kamu "
        "untuk memulai proses identifikasi."
    )

    uploaded_file = st.file_uploader(
        "Masukkan gambar batik",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Gambar yang akan dianalisis",
                use_container_width=True
            )

            analyze = st.button(
                "✦ Analisis Motif Batik",
                type="primary"
            )

        except Exception:
            st.error("Gambar tidak dapat dibaca.")
            image = None
            analyze = False

    else:
        image = None
        analyze = False

        render_html("""
        <div class="panel" style="text-align:center; padding:45px 20px;">
        <div style="font-size:45px;">🖼️</div>
        <h3>Area Upload Gambar</h3>
        <p style="color:#8B7667;">
        Unggah gambar untuk melihat hasil klasifikasi.
        </p>
        <small style="color:#A18A78;">
        Format JPG, JPEG, PNG
        </small>
        </div>
        """)


# ==========================================
# 8. IMAGE CLASSIFICATION
# ==========================================

prediction = None
confidence = None
predicted_class = None
display_name = None

with right:
    render_html("""
    <div class="section-label">
    02 — CLASSIFICATION
    </div>
    """)

    st.markdown("### Hasil Identifikasi")

    if analyze and image is not None:
        try:
            with st.spinner("Sedang menganalisis motif batik..."):

                resized = image.resize(
                    (224, 224),
                    Image.Resampling.NEAREST
                )

                image_array = np.array(
                    resized,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                image_array = preprocess_input(image_array)

                prediction = model.predict(
                    image_array,
                    verbose=0
                )

                predicted_index = int(
                    np.argmax(prediction[0])
                )

                predicted_class = class_names[predicted_index]

                confidence = float(
                    prediction[0][predicted_index] * 100
                )

            display_name = predicted_class.replace(
                "batik_", ""
            ).replace("_", " ").title()

            if confidence >= THRESHOLD:
                render_html(f"""
                <div class="result-card">
                <div class="section-label">
                DETECTED MOTIF
                </div>
                <div class="result-name">
                {display_name}
                </div>
                <div class="confidence">
                {confidence:.2f}%
                </div>
                <div class="status-good">
                ✓ Motif berhasil dikenali
                </div>
                </div>
                """)

            else:
                render_html(f"""
                <div class="result-card">
                <div class="section-label">
                IDENTIFICATION RESULT
                </div>
                <div class="result-name">
                Motif Tidak Dikenali
                </div>
                <div class="confidence">
                {confidence:.2f}%
                </div>
                <div class="status-unknown">
                ! Di luar kriteria pengenalan
                </div>
                <p style="color:#765F50; margin-top:15px;">
                Tingkat keyakinan model belum mencapai
                batas minimum pengenalan.
                </p>
                </div>
                """)

            st.caption(f"Prediksi asli model: {display_name}")
            st.caption(f"Skor probabilitas model: {confidence:.2f}%")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

    else:
        render_html("""
        <div class="panel" style="min-height:230px; display:flex;
        flex-direction:column; justify-content:center;
        align-items:center; text-align:center;">
        <div style="font-size:42px;">✦</div>
        <h3>Menunggu Gambar</h3>
        <p style="color:#8B7667;">
        Hasil klasifikasi akan muncul di sini
        setelah gambar dianalisis.
        </p>
        </div>
        """)


# ==========================================
# 9. PROBABILITY DETAILS
# ==========================================

if prediction is not None:
    st.markdown("---")

    render_html("""
    <div class="section-label">
    03 — MODEL ANALYSIS
    </div>
    """)

    st.markdown("### Distribusi Probabilitas")

    st.write(
        "Berikut tingkat probabilitas prediksi model "
        "untuk setiap kategori motif."
    )

    for i, name in enumerate(class_names):
        probability = float(prediction[0][i])

        display_name = name.replace(
            "batik_", ""
        ).replace("_", " ").title()

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write(f"**{display_name}**")

        with col2:
            st.write(f"**{probability * 100:.2f}%**")

        st.progress(
            min(max(probability, 0.0), 1.0)
        )


# ==========================================
# 10. INFORMATION FOOTER
# ==========================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**COMPUTER VISION**")
    st.caption("Image Classification")

with col2:
    st.markdown("**CNN ARCHITECTURE**")
    st.caption("MobileNetV2")

with col3:
    st.markdown("**MODEL TYPE**")
    st.caption("Transfer Learning")


render_html("""
<div style="text-align:center; padding:25px 0 5px 0;
color:#9A8372; font-size:12px;">
BATIK VISION · Preserving Heritage Through Technology
</div>
""")
