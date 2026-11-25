import subprocess
from pathlib import Path
from ultralytics import YOLO

weight_files=sorted(
    Path(runs)
)