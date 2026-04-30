import streamlit as st
import requests
from PIL import Image
import io
import json

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = "DEIN_HF_TOKEN"

API_URL = "https://api-inference.huggingface.co/models/keremberke/yolov8n-object-detection"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# -----------------------------
# API CALL
# -----------------------------
def query(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    response = requests.post(
        API_URL,
        headers=headers,
        data=img_bytes
    )

    return response.json()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🧠 YOLOv8 Object Detection (Hugging Face API)")
st.write("Upload ein Bild – KI erkennt Objekte ohne lokale Modelle")

uploaded_file = st.file_uploader(
    "Bild auswählen...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Originalbild", use_column_width=True)

    st.write("🔄 Analysiere mit Hugging Face YOLOv8...")

    result = query(image)

    # -----------------------------
    # Debug optional
    # -----------------------------
    # st.write(result)

    st.subheader("🔎 Erkannte Objekte:")

    if isinstance(result, list) and len(result) > 0:

        for obj in result:
            label = obj.get("label", "unknown")
            score = obj.get("score", 0) * 100
            box = obj.get("box", {})

            st.write(f"**{label}** — {score:.2f}%")
            st.json(box)

    else:
        st.write("Keine Objekte erkannt oder API lädt noch (erste Anfrage kann 10–20s dauern).")
