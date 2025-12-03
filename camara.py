from ultralytics import YOLO
import cv2
import numpy as np
import os
import insightface
import time
from numpy.linalg import norm
from collections import deque

detector = YOLO("yolov11n-face.pt")

modelo = insightface.app.FaceAnalysis(name="buffalo_l")
modelo.prepare(ctx_id=0, det_size=(640, 640))


embeddings = {}
carpeta = "rostros_guardados"

for archivo in os.listdir(carpeta):
    if archivo.endswith(".npy"):
        nombre = archivo.replace(".npy", "")
        emb = np.load(os.path.join(carpeta, archivo))
        embeddings[nombre] = emb / norm(emb)

if not embeddings:
    print("No hay rostros registrados.")
    exit()

print("Embeddings cargados:", list(embeddings.keys()))

def distancia_cosine(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

def reconocer_emb(embedding_input):
    mejor_nombre = None
    mejor_distancia = 999

    for nombre, emb_guardado in embeddings.items():
        d = distancia_cosine(embedding_input, emb_guardado)
        if d < mejor_distancia:
            mejor_distancia = d
            mejor_nombre = nombre

    if mejor_distancia < 0.40:
        return mejor_nombre, mejor_distancia

    return "Desconocido", mejor_distancia


os.makedirs("desconocidos", exist_ok=True)
ultimo_embedding_desconocido = None
umbral_cambio = 0.28
contador_desconocidos = 1

buffer_nombres = deque(maxlen=5)

# NUEVO: temporizador para desconocidos
tiempo_inicio_desconocido = None
segundos_para_guardar = 3

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 360)

print("Reconociendo...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    resultados = detector(frame)[0]

    # Ejecutar InsightFace SOLO UNA VEZ POR FRAME → MUCHO MÁS FLUIDO
    analisis = modelo.get(frame)

    for box in resultados.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        rostro = frame[y1:y2, x1:x2]

        if rostro.size == 0:
            continue

        if len(analisis) == 0:
            continue

        cara = min(analisis, key=lambda c: abs(c.bbox[0]-x1)+abs(c.bbox[1]-y1))

        emb = cara.embedding
        emb = emb / norm(emb)

        nombre, dist = reconocer_emb(emb)

        buffer_nombres.append(nombre)
        nombre_estable = max(set(buffer_nombres), key=buffer_nombres.count)

        # =============================
        #      SISTEMA DE TIMER
        # =============================
        if nombre_estable == "Desconocido":
            if tiempo_inicio_desconocido is None:
                tiempo_inicio_desconocido = time.time()

            tiempo_transcurrido = time.time() - tiempo_inicio_desconocido

            if tiempo_transcurrido >= segundos_para_guardar:
                guardar = False

                if ultimo_embedding_desconocido is None:
                    guardar = True
                else:
                    d = distancia_cosine(emb, ultimo_embedding_desconocido)
                    if d > umbral_cambio:
                        guardar = True

                if guardar:
                    archivo = f"desconocidos/desconocido_{contador_desconocidos:03d}.jpg"
                    cv2.imwrite(archivo, rostro)
                    print(f"[OK] Desconocido guardado → {archivo}")

                    ultimo_embedding_desconocido = emb.copy()
                    contador_desconocidos += 1

                tiempo_inicio_desconocido = None  

        else:
            tiempo_inicio_desconocido = None

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{nombre_estable} ({dist:.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
