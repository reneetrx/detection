import shutil
import kagglehub

# Downloads to cache, then copies straight into a 'dataset' folder in your project
cache_path = kagglehub.dataset_download("redzapdos123/aerial-view-of-vehicles-and-humans-dataset-yolo")
shutil.copytree(cache_path, "./dataset", dirs_exist_ok=True)

print("Done! Check the 'dataset' folder in VS Code.")
