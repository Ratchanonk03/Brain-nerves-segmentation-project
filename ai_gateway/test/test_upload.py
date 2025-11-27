import requests
from pathlib import Path

url = "http://localhost:8000/infer"
cwd = Path.cwd()
dataset_path = cwd.parent / "dataset" / "images"
print("CWD:", cwd)
print(f"Dataset Path: {dataset_path} Exists: {dataset_path.exists()}")
file_path = dataset_path / "1.png"

with open(file_path, "rb") as f:
    files = {"payload": (file_path.name, f, "image/png")}
    resp = requests.post(url, files=files)

print("Status:", resp.status_code)
print("Response:", resp.json())