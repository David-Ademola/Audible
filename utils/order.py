import shutil
import os

DATA_DIR = "data"
FAKE_DIR = os.path.join(DATA_DIR, "fake")
os.makedirs(FAKE_DIR, exist_ok=True)

for root, dirs, files in os.walk(DATA_DIR):
    if root == DATA_DIR or root == FAKE_DIR or root.endswith("real"):
        continue

    for file in os.listdir(root):
        if file.endswith(".wav"):
            new_file_name = root.split("/")[-1] + "_" + file
            print(f"Moving {file} to {new_file_name}")
            shutil.move(os.path.join(root, file), os.path.join(FAKE_DIR, new_file_name))
