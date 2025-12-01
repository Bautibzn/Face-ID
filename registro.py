import cv2
import os
import numpy as np
import insightface
from numpy.linalg import norm

modelo = insightface.app.FaceAnalysis(name="buffalo_l")
modelo.prepare(ctx_id=0, det_size=(640, 640))

nombre = input("Nombre de la persona: ")

os.makedirs("rostros_guardados", exist_ok=True)

cap = cv2.VideoCapture(0)

print("\n=== Registro de rostro multi-ángulo ===")
print("Mira al frente y presioná S para comenzar\n")

embeddings_capturados = []

pasos = [
    "Mira al frente",
    "Girate un poco a la izquierda",
    "Girate un poco a la derecha",
    "Mira un poco hacia arriba",
    "Mira un poco hacia abajo",
]

paso_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    mensaje = pasos[paso_idx] if paso_idx < len(pasos) else "Finalizando..."
    cv2.putText(frame, mensaje, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Registro", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord('s'):

        caras = modelo.get(frame)

        if len(caras) == 0:
            print("No se detectó una cara clara. Intentá de nuevo.")
            continue

        cara = max(caras, key=lambda c: (c.bbox[2]-c.bbox[0])*(c.bbox[3]-c.bbox[1]))

        embedding = cara.embedding
        embeddings_capturados.append(embedding)

        print(f"Capturada posición: {pasos[paso_idx]}")

        paso_idx += 1

        if paso_idx >= len(pasos):
            break

    if tecla == ord('q'):
        exit()

cap.release()
cv2.destroyAllWindows()

if len(embeddings_capturados) == 0:
    print("No se capturaron rostros. Abortando.")
    exit()

promedio = np.mean(embeddings_capturados, axis=0)
promedio = promedio / norm(promedio)

ruta = f"rostros_guardados/{nombre}.npy"
np.save(ruta, promedio)

print(f"\n=== Registro completado ===")
print(f"Se guardó el embedding mejorado en: {ruta}")
