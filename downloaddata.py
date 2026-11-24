import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import kagglehub

# Replace with your new Kaggle dataset slug
download_path = Path(kagglehub.dataset_download("redzapdos123/aerial-view-of-vehicles-and-humans-dataset-yolo"))

# Index all images (.jpg / .jpeg / .png)
all_images = (
    list(download_path.rglob("*.jpg"))
    + list(download_path.rglob("*.jpeg"))
    + list(download_path.rglob("*.png"))
)
img_by_stem = {img.stem: img for img in all_images}

xml_files = list(download_path.rglob("*.xml"))
txt_files = [
    f for f in download_path.rglob("*.txt")
    if f.name.lower() not in {"classes.txt", "train.txt", "val.txt", "test.txt", "readme.txt"}
]

annotations = {}
class_names = []

# --- If the dataset uses Pascal VOC .xml files ---
if xml_files:
    class_names_set = set()
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall("object"):
            name_el = obj.find("name")
            if name_el is not None and name_el.text:
                class_names_set.add(name_el.text.strip())

    class_names = sorted(class_names_set)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    print(f"Found {len(class_names)} classes: {class_names}")

    for xml_path in xml_files:
        img_path = img_by_stem.get(xml_path.stem)
        if not img_path:
            continue

        with Image.open(img_path) as im:
            w, h = im.size
        if w <= 0 or h <= 0:
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        lines = []
        for obj in root.findall("object"):
            name_el = obj.find("name")
            box = obj.find("bndbox")
            if name_el is None or not name_el.text or box is None:
                continue

            cls_id = class_to_id[name_el.text.strip()]

            if box.find("xmin") is not None:
                xmin, ymin = float(box.find("xmin").text), float(box.find("ymin").text)
                xmax, ymax = float(box.find("xmax").text), float(box.find("ymax").text)
            else:
                xs = [float(box.find(f"x{i}").text) for i in range(4) if box.find(f"x{i}") is not None]
                ys = [float(box.find(f"y{i}").text) for i in range(4) if box.find(f"y{i}") is not None]
                if not xs or not ys:
                    continue
                xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

            xc = max(0.0, min(1.0, ((xmin + xmax) / 2.0) / w))
            yc = max(0.0, min(1.0, ((ymin + ymax) / 2.0) / h))
            bw = max(0.0, min(1.0, (xmax - xmin) / w))
            bh = max(0.0, min(1.0, (ymax - ymin) / h))
            if bw > 0 and bh > 0:
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            annotations[img_path] = lines

# --- If the dataset already has YOLO .txt files ---
elif txt_files:
    max_id = 0
    for txt_path in txt_files:
        img_path = img_by_stem.get(txt_path.stem)
        if not img_path:
            continue
        lines = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
        if lines:
            annotations[img_path] = lines
            for l in lines:
                max_id = max(max_id, int(l.split()[0]))

    classes_file = next(download_path.rglob("classes.txt"), None)
    if classes_file:
        class_names = [c.strip() for c in classes_file.read_text().splitlines() if c.strip()]
    else:
        class_names = [f"class_{i}" for i in range(max_id + 1)]

print(f"Successfully prepared {len(annotations)} labeled images.")

# Sample 6,000 train and 1,500 val images (7,500 total)
items = list(annotations.items())
random.seed(42)
random.shuffle(items)
train_items = items[:6000]
val_items = items[6000:7500]

def save_split(split_items, split):
    img_dest = Path(f"dataset/images/{split}")
    lbl_dest = Path(f"dataset/labels/{split}")
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)
    for img_path, yolo_lines in split_items:
        shutil.copy(img_path, img_dest / img_path.name)
        (lbl_dest / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines))

save_split(train_items, "train")
save_split(val_items, "val")

# Automatically write dataset.yaml
yaml_lines = [
    "path: ./dataset",
    "train: images/train",
    "val: images/val",
    "",
    "names:"
]
for idx, name in enumerate(class_names):
    yaml_lines.append(f"  {idx}: {name}")

Path("dataset.yaml").write_text("\n".join(yaml_lines) + "\n")
print(f"Done! Copied {len(train_items)} train and {len(val_items)} val images.")
