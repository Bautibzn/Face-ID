import os
import cv2
from ultralytics import YOLO

MODEL_PATH = "yolov11n-face.pt"

IMAGES_TRAIN = "data/images/train"
IMAGES_VAL   = "data/images/val"

LABELS_TRAIN = "data/labels/train"
LABELS_VAL   = "data/labels/val"

os.makedirs(LABELS_TRAIN, exist_ok=True)
os.makedirs(LABELS_VAL, exist_ok=True)

model = YOLO(MODEL_PATH)

def autolabel(img_folder, label_folder):
    images = [img for img in os.listdir(img_folder)
              if img.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"[INFO] Procesando {len(images)} imágenes en: {img_folder}")

    for img_name in images:
        img_path = os.path.join(img_folder, img_name)
        label_path = os.path.join(
            label_folder,
            img_name.rsplit(".", 1)[0] + ".txt"
        )

        img = cv2.imread(img_path)
        if img is None:
            print("[WARN] No se pudo leer:", img_name)
            continue

        h, w = img.shape[:2]

        
        results = model(img, verbose=False)[0]

        yolo_lines = []

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            line = f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
            yolo_lines.append(line)

        with open(label_path, "w") as f:
            f.writelines(yolo_lines)

    print("[OK] Etiquetado finalizado:", label_folder)

autolabel(IMAGES_TRAIN, LABELS_TRAIN)
autolabel(IMAGES_VAL, LABELS_VAL)

