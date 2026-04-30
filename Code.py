import streamlit as st
import requests
from PIL import Image
import io
import time

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = "DEIN_HF_TOKEN"

API_URL = "https://api-inference.huggingface.co/models/keremberke/yolov8n-object-detection"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# -----------------------------
# SAFE API CALL
# -----------------------------
def query(image: Image.Image, retries=3):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    for i in range(retries):
        response = requests.post(
            API_URL,
            headers=headers,
            data=img_bytes
        )

        # Debug optional
        # st.write(response.status_code, response.text)

        # Erfolgreich
        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                return {"error": True, "text": response.text}

        # Model lädt noch → warten
        if response.status_code == 503:
            time.sleep(3)
            continue

        # anderer Fehler
        return {
            "error": True,
            "status_code": response.status_code,
            "text": response.text
        }

    return {"error": True, "text": "Max retries exceeded"}

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🧠 YOLOv8 Objekt-Erkennung (Hugging Face API)")
st.write("Lade ein Bild hoch – KI erkennt Objekte automatisch.")

uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="📷 Originalbild", use_column_width=True)

    st.write("🔄 Analysiere Bild...")

    result = query(image)

    st.subheader("🔎 Ergebnisse:")

    # -----------------------------
    # Fehlerfall
    # -----------------------------
    if isinstance(result, dict) and result.get("error"):
        st.error("API Fehler")
        st.write(result.get("text"))

    # -----------------------------
    # Erfolgreiche Detection
    # -----------------------------
    elif isinstance(result, list):

        if len(result) == 0:
            st.write("Keine Objekte erkannt.")
        else:
            for obj in result:
                label = obj.get("label", "unknown")
                score = obj.get("score", 0) * 100
                box = obj.get("box", {})

                st.write(f"**{label}** — {score:.2f}%")
                st.json(box)

    # -----------------------------
    # Fallback
    # -----------------------------
    else:
        st.write(result)
