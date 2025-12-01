from ultralytics import YOLO
import cv2
import os
import random

# Cargar modelo
model = YOLO('runs/detect/FaceId4/weights/best.pt')

# Carpeta con imágenes
carpeta = 'data/images/train'

# Obtener todas las imágenes
todas = [os.path.join(carpeta, f) for f in os.listdir(carpeta)
         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Elegir solo 10 al azar
imagenes_random = random.sample(todas, min(10, len(todas)))

print("Imágenes seleccionadas:")
for img in imagenes_random:
    print(" →", img)

# Ejecutar YOLO sobre esas 10 imágenes
preds = model(imagenes_random)

# Mostrar resultados
for pred in preds:
    img = pred.plot()

    cv2.imshow("Predicción", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

# Guardar resultados en carpeta y txt
model.predict(source=imagenes_random, save=True, save_txt=True)
