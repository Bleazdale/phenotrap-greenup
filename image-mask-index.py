# This script has been written to link images to annotated masks within a csv index for training a semantic segmentation model

# === Import Dependencies ===
import os
import re
import json
import csv

# === Configure Paths ===
json_path = r"\\path\to\your\file.json"
mask_folder = r"\\path\to\your\masks"
output_file = r"\\path\to\your\output.csv"

# === Load Label Studio JSON ===
with open(json_path, "r", encoding="utf-8") as f:
    tasks = json.load(f)

id_to_image = {}
for task in tasks:
    task_id = task.get("id")
    file_upload = task.get("file_upload", "")
    if task_id and file_upload:
        clean_name = file_upload.split("-", 1)[-1]   # remove LS prefix
        clean_name = os.path.splitext(clean_name)[0] # remove extension
        id_to_image[str(task_id)] = clean_name

# === Collect merged masks ===
mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(".png")]
mask_files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])

# === Write CSV linking image with mask ===
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image-name", "mask-name"])

    for mask_file in mask_files:
        # Expect mask_###.png
        match = re.match(r"mask(\d+)\.png$", mask_file, flags=re.IGNORECASE)
        if not match:
            continue

        task_id = match.group(1)
        base_image = id_to_image.get(task_id, "Unknown")
        image_name = base_image + ".JPG" if base_image != "Unknown" else "Unknown"

        writer.writerow([image_name, mask_file])

print(f"Saved index file: {output_file}")
