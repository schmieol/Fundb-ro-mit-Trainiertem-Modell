from ultralytics import YOLO

# Modell laden
model = YOLO("yolov8n.pt")

# Fundliste vorbereiten
if not os.path.exists("fundliste.csv"):
    df = pd.DataFrame(columns=["Zeit", "Objekt", "Confidence", "Bild"])
    df.to_csv("fundliste.csv", index=False)

# Kamera starten
cap = cv2.VideoCapture(0)

print("Fundkisten-System gestartet. Drücke 'q' zum Beenden.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Objekterkennung
    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            # Nur relevante Fundkisten-Objekte speichern
            if label in ["backpack", "handbag", "cell phone", "bottle"]:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                img_name = f"fund_{timestamp}.jpg"

                # Bild speichern
                cv2.imwrite(img_name, frame)

                # In CSV speichern
                new_entry = pd.DataFrame([{
                    "Zeit": timestamp,
                    "Objekt": label,
                    "Confidence": conf,
                    "Bild": img_name
                }])

                new_entry.to_csv("fundliste.csv", mode="a", header=False, index=False)

                print(f"Fund erkannt: {label} ({conf:.2f})")

            # Bounding Box anzeigen
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        2)

    # Live-Anzeige
    cv2.imshow("Fundkiste Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
