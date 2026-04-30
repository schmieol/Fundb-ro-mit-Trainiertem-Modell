import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import time

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = "DEIN_HF_TOKEN"

# stabiler YOLOv8 Endpoint (HF Inference)
API_URL = "https://api-inference.huggingface.co/models/hustvl/yolos-tiny"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# -----------------------------
# API CALL (robust)
# -----------------------------
def query(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    for _ in range(3):
        response = requests.post(
            API_URL,
            headers=headers,
            data=img_bytes
        )

        # Model lädt noch
        if response.status_code == 503:
            time.sleep(2)
            continue

        if response.status_code != 200:
            return {"error": True, "text": response.text}

        try:
            return response.json()
        except:
            return {"error": True, "text": response.text}

    return {"error": True, "text": "Timeout / Model not ready"}

# -----------------------------
# DRAW YOLO BOXES
# -----------------------------
def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)

    for det in detections:
        label = det.get("label", "objekt")
        score = det.get("score", 0)

        box = det.get("box", {})
        xmin = box.get("xmin", 0)
        ymin = box.get("ymin", 0)
        xmax = box.get("xmax", 0)
        ymax = box.get("ymax", 0)

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin), f"{label} {score:.2f}", fill="red")

    return image

# -----------------------------
# UI
# -----------------------------
st.title("🧠 YOLO Object Detection (Hugging Face API)")
st.write("Upload ein Bild – YOLO erkennt Objekte mit Bounding Boxes")

uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="📷 Originalbild", use_column_width=True)

    st.write("🔄 Analysiere mit YOLO API...")

    result = query(image)

    st.subheader("🔎 Ergebnisse:")

    # -----------------------------
    # ERROR HANDLING
    # -----------------------------
    if isinstance(result, dict) and result.get("error"):
        st.error("API Fehler")
        st.write(result["text"])

    # -----------------------------
    # YOLO DETECTIONS
    # -----------------------------
    elif isinstance(result, list):

        if len(result) == 0:
            st.write("Keine Objekte erkannt.")
        else:
            for obj in result:
                st.write(f"**{obj.get('label')}** — {obj.get('score',0)*100:.2f}%")

            # Bild mit Boxen anzeigen
            img_with_boxes = draw_boxes(image.copy(), result)
            st.image(img_with_boxes, caption="📦 YOLO Ergebnisse", use_column_width=True)

    else:
        st.write(result)
