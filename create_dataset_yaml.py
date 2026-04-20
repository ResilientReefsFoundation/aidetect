#!/usr/bin/env python3
"""
Run this in D:\CotsImages\train\ to create a data.yaml for the CSIRO dataset.
Usage: python create_dataset_yaml.py
"""
import os, yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, "images")
labels_dir = os.path.join(script_dir, "labels")

# Count images and non-empty labels
img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
non_empty = sum(1 for f in label_files if os.path.getsize(os.path.join(labels_dir, f)) > 0)

print(f"Images: {img_count}")
print(f"Label files: {len(label_files)}")
print(f"Non-empty labels (have detections): {non_empty}")
print(f"Empty labels (negative samples): {len(label_files) - non_empty}")

# Check first non-empty label to get class names
classes = ["cots-detection"]  # default
for f in label_files:
    path = os.path.join(labels_dir, f)
    if os.path.getsize(path) > 0:
        with open(path) as fp:
            line = fp.readline().strip()
            if line:
                parts = line.split()
                print(f"\nSample label line: {line}")
                print(f"Class index: {parts[0]}")
        break

data = {
    "train": images_dir.replace("\\", "/"),
    "val":   images_dir.replace("\\", "/"),
    "nc":    1,
    "names": classes,
}

yaml_path = os.path.join(script_dir, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data, f)

print(f"\ndata.yaml created at {yaml_path}")
print("Now zip the entire train/ folder and upload to the Train tab.")
