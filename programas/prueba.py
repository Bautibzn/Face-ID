from ultralytics import YOLO
import cv2
import os
import random

model = YOLO('runs/detect/FaceId4/weights/best.pt')

carpeta = 'data/images/train'

todas = [os.path.join(carpeta, f) for f in os.listdir(carpeta)
         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

imagenes_random = random.sample(todas, min(10, len(todas)))

print("Imágenes seleccionadas:")
for img in imagenes_random:
    print(" →", img)

preds = model(imagenes_random)

for pred in preds:
    img = pred.plot()

    cv2.imshow("Predicción", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

model.predict(source=imagenes_random, save=True, save_txt=True)

