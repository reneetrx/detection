from ultralytics import YOLO

model = YOLO("yolo26m.pt")

model.train(
    data="dataset.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    device=0,
    workers=4   
)