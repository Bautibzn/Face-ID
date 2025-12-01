from ultralytics import YOLO
import cv2
import os
import numpy as np
import insightface

# Cargar detector YOLO
detector = YOLO("yolov11n-face.pt")

# Cargar InsightFace
modelo = insightface.app.FaceAnalysis(name="buffalo_l")
modelo.prepare(ctx_id=0, det_size=(128, 128))  # más rápido y estable

nombre = input("Nombre de la persona: ")
os.makedirs("rostros_guardados", exist_ok=True)

cap = cv2.VideoCapture(0)

print("Presiona S para capturar rostro")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = detector(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # RECORTE DEL ROSTRO
        recorte = frame[y1:y2, x1:x2]

        if recorte.size == 0:
            continue

        cv2.imshow("Rostro detectado", recorte)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            caras = modelo.get(recorte)

            if len(caras) == 0:
                print("❌ InsightFace no encontró un rostro en el recorte. Reintentar.")
                continue

            embedding = caras[0].embedding
            np.save(f"rostros_guardados/{nombre}.npy", embedding)
            print(f"✔️ Rostro guardado como rostros_guardados/{nombre}.npy")

            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.imshow("Camara", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
