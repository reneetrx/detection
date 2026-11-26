from ultralytics import YOLO

# Point to your previous run's best weights
model = YOLO("/home/lenovo6/Desktop/Interns/detection/runs/detect/train4/weights/last.pt")

model.train(
    data="dataset.yaml",
    epochs=75,
    imgsz=640,
    batch=32,
    device=0,
    workers=8,
    patience=20
)
