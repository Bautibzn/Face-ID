from ultralytics import YOLO

model = YOLO("yolov11n-face.pt")

results = model.train(
    data="data.yaml",
    epochs=15,          # más estable que 15 (no es mucho más lento)
    imgsz=384,          # ideal para iGPU
    batch=4,            # seguro para tu VRAM
    name='IdFace',
    amp=True,
    cos_lr=True,
    cache='disk',       # evita saturar RAM
    workers=0,

    mosaic=0.1,         # augment suave bueno para rostros
    mixup=0.0,
    hsv_h=0.015,        # augment muy controlado
    hsv_s=0.7,
    hsv_v=0.4,

    freeze=10,          # acelera entrenamiento
    warmup_epochs=2,
)

# cargar mejor modelo
model = YOLO("runs/detect/FaceId4/weights/best.pt")
