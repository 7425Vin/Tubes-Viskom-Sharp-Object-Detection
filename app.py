import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np


# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Deteksi Benda Tajam - YOLO 11",
    layout="wide"
)

# ======================
# HEADER
# ======================
st.title("Deteksi Benda Tajam Menggunakan YOLO 11")

st.markdown("""
**Nama  : Vincentius Setyawan Widyahadi**  
**NIM   : 24060122120006**  

Aplikasi ini merupakan sistem **deteksi objek berbasis Computer Vision**
yang mampu mengenali **benda tajam** seperti:
- Gunting
- Pisau
- Cutter  

""")

st.divider()

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.selectbox(
    "Pilih Mode Deteksi",
    ["Gambar", "Video", "Realtime Webcam"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5
)

# ======================
# MODE GAMBAR
# ======================
if menu == "Gambar":
    st.subheader("Deteksi dari Gambar")

    uploaded = st.file_uploader(
        "Upload gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar Asli", use_column_width=True)

        results = model(img, conf=confidence)
        result_img = results[0].plot()

        st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

# ======================
# MODE VIDEO
# ======================
elif menu == "Video":
    st.subheader("Deteksi dari Video")

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)
            frame = results[0].plot()
            stframe.image(frame, channels="BGR")

        cap.release()

# ======================
# MODE REALTIME WEBCAM
# ======================
elif menu == "Realtime Webcam":
    st.subheader("Realtime Webcam Detection")

    start = st.checkbox("Start Webcam")
    stframe = st.image([])

    if start:
        cap = cv2.VideoCapture(0)

        while start:
            ret, frame = cap.read()
            if not ret:
                st.error("Tidak dapat mengakses webcam")
                break

            results = model(frame, conf=confidence)
            frame = results[0].plot()
            stframe.image(frame, channels="BGR")

        cap.release()
