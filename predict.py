import os
from ultralytics import YOLO
from pathlib import Path
import subprocess

if __name__ == "__main__":
    output_dir = "detection/output"
    os.makedirs(output_dir, exist_ok=True)
    
    weights_path = model_path = "/home/lenovo6/Desktop/Interns/detection/runs/detect/train6/weights/best.pt"
    # os.path.join("detection", "runs", "detect", "train2", "weights", "best.pt")

    model = YOLO(weights_path)
    
    image_path = "https://media.istockphoto.com/id/1011441936/photo/traffic-jam-on-multilane-road.jpg?s=612x612&w=0&k=20&c=bcKWjPgkBa7RR0boHkikbcRUkizeyDYlEd0Dek-L_F8="
    results = model.predict(
        source = image_path,
        imgsz = 640,  
        conf = 0.25,
        iou = 0.45,
        device = 0,
        save = True,
        project = output_dir,
        name = "predictions",
        exist_ok = True,
    )
    
    for r in results:
        if len(r.boxes) == 0:
            print("No objects detected.")
        for box in r.boxes:
            cls_id = model.names[int(box.cls[0])]
            
            conf = float(box.conf[0]) 
            os.makedirs(output_dir, exist_ok=True)
    
            xyxy = box.xyxy[0].cpu().numpy()
            print(f"Detected class {cls_id} with confidence {conf:.2f} at {xyxy}.")
            
saved_dir = Path(results[0].save_dir)
saved_files = list(saved_dir.rglob("*.*"))
if saved_files:
    print("Opening:", saved_files[0])
    subprocess.run(["open", str(saved_files[0])])