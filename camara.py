from ultralytics import YOLO
import cv2
import numpy as np
import os
import insightface
from numpy.linalg import norm
from collections import deque

# ============================
# 1) CONFIGURACIÓN Y MODELOS
# ============================
detector = YOLO("yolov11n-face.pt")

modelo = insightface.app.FaceAnalysis(name="buffalo_l")
modelo.prepare(ctx_id=0, det_size=(320, 320))  # mayor calidad

# ============================
# 2) CARGAR EMBEDDINGS
# ============================
embeddings = {}
carpeta = "rostros_guardados"

for archivo in os.listdir(carpeta):
    if archivo.endswith(".npy"):
        nombre = archivo.replace(".npy", "")
        emb = np.load(os.path.join(carpeta, archivo))
        embeddings[nombre] = emb / norm(emb)  # normalizar
        

if len(embeddings) == 0:
    print("❌ No hay rostros registrados. Ejecuta registro.py primero.")
    exit()

print("Embeddings cargados:", embeddings.keys())


# ============================
# 3) FUNCIONES
# ============================

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

    # Umbral dinámico (más robusto)
    if mejor_distancia < 0.45:
        return mejor_nombre, mejor_distancia

    return "Desconocido", mejor_distancia


# ============================
# 4) BUFFER DE ESTABILIDAD
# ============================
buffer_nombres = deque(maxlen=5)


# ============================
# 5) INICIAR CÁMARA
# ============================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Reconociendo... Presiona Q para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    resultados = detector(frame)[0]

    for box in resultados.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        rostro = frame[y1:y2, x1:x2]

        # evitar errores de InsightFace
        if rostro.size == 0:
            continue
        
        # InsightFace requiere tamaño mínimo
        rostro_resized = cv2.resize(rostro, (256, 256))

        caras = modelo.get(rostro_resized)

        if len(caras) == 0:
            continue

        # tomar solo la cara más grande
        cara = max(caras, key=lambda c: c.bbox[2] - c.bbox[0])

        emb = cara.embedding
        
        nombre, dist = reconocer_emb(emb)

        buffer_nombres.append(nombre)  # agregar al historial

        # nombre por mayoría (evita saltos)
        nombre_estable = max(set(buffer_nombres), key=buffer_nombres.count)

        # dibujar resultados
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{nombre_estable} ({dist:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
