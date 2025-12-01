from ultralytics import YOLO
import cv2
import os

model = YOLO('runs/detect/FaceId4/weights/best.pt')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

carpeta_salida = "pred_cam"
os.makedirs(carpeta_salida, exist_ok=True)

contador = 0

print("Detectando... presiona Q para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame)[0]
    img_pred = results.plot()

    cv2.imshow("Camara - Detección", img_pred)
    
    nombre_img = f"{carpeta_salida}/frame_{contador}.jpg"
    cv2.imwrite(nombre_img, img_pred)

    nombre_txt = f"{carpeta_salida}/frame_{contador}.txt"
    with open(nombre_txt, "w") as f:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x, y, w, h = box.xywh[0]

            f.write(f"{cls} {conf:.4f} {x:.4f} {y:.4f} {w:.4f} {h:.4f}\n")

    contador += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Listo. Resultados guardados en:", carpeta_salida)

