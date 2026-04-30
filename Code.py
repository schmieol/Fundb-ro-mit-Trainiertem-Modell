import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# ---------------------------
# YOLOv8 Modell laden
# ---------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # leicht & schnell
    return model

model = load_model()

# ---------------------------
# Prediction Funktion
# ---------------------------
def predict(image):
    results = model(image)
    return results[0]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🧠 YOLOv8 Produkt-Erkennung")
st.write("Lade ein Bild hoch – das Modell erkennt Objekte automatisch.")

uploaded_file = st.file_uploader(
    "Bild auswählen...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="📷 Originalbild", use_column_width=True)

    st.write("🔄 Analysiere mit YOLOv8...")

    # Prediction
    result = predict(image)

    # Bild mit Bounding Boxes
    plotted_img = result.plot()

    st.image(plotted_img, caption="📦 Erkanntes Ergebnis", use_column_width=True)

    # ---------------------------
    # Ergebnisse anzeigen
    # ---------------------------
    st.subheader("🔎 Erkannte Objekte:")

    if len(result.boxes) == 0:
        st.write("Keine Objekte erkannt.")
    else:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            st.write(f"**{label}** — {conf * 100:.2f}%")
