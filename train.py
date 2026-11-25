from ultralytics import YOLO

# Point to your previous run's best weights
model = YOLO("/home/lenovo6/Desktop/Interns/detection/runs/detect/train2/weights/best.pt")

model.train(
    data="dataset.yaml",
    epochs=75,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    patience=20
)
