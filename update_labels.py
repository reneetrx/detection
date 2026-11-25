import torch

# new_names = {
#     0: "small vehicle",
#     1: "large vehicle",
#     2: "human",
# }


# weights_path = "/home/lenovo6/Desktop/Interns/detection/runs/detect/train2/weights/best.pt"
# ckpt = torch.load(weights_path, map_location=0, weights_only=False)
# ckpt["model"].names.update(new_names)
# torch.save(ckpt, weights_path)
# print("Updated class names in best.pt", ckpt["model"].names)

weights_path="/home/lenovo6/Desktop/Interns/detection/runs/detect/train2/weights/best.pt"
ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
new_names = {
    0: "small vehicle",
    1: "large vehicle",
    2: "human",
}

ckpt["model"].names = new_names

torch.save(ckpt, weights_path)
print("Updated class names in best.pt", ckpt["model"].names)