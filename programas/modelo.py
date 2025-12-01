from ultralytics import YOLO

model = YOLO("yolov11n-face.pt")

results = model.train(
    data="data.yaml",
    epochs=15,          
    imgsz=384,          
    batch=4,           
    name='IdFace',
    amp=True,
    cos_lr=True,
    cache='disk',       
    workers=0,
    mosaic=0.1,         
    mixup=0.0,
    hsv_h=0.015,        
    hsv_s=0.7,
    hsv_v=0.4,
    freeze=10,         
    warmup_epochs=2,
)

model = YOLO("runs/detect/FaceId4/weights/best.pt")

