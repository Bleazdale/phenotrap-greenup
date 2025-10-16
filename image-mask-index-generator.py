# Image-Mask Index Generator
#-----------------------------------------------------------------------------------------------
# This script produces an index that maps images, masks and classes together for the purposes of semantic segmentation training
# Image names are derived from a JSON file that was also used to produce the mask PNG files.
# The output of this script is a CSV file that is to be incorporated in a semantic segmentation script


# Import dependencies
import os
import re
import json
import csv

# Define paths
json_path = r"path/to/your/file.json" # Replace with the path to your JSON file
mask_folder = r"path/to/your/mask_folder" # Replace with the path to your maks folder
output_file = r"path/to/your/output_index.csv" # Replace with the desired path and name of your index CSV file

# Build lookup dict: task_id -> true image name
with open(json_path, "r", encoding="utf-8") as f:
    tasks = json.load(f)

id_to_image = {}
for task in tasks:
    task_id = task.get("id")
    file_upload = task.get("file_upload", "")
    if task_id and file_upload:
        clean_name = file_upload.split("-", 1)[-1]      # remove Label Studio prefix
        clean_name = os.path.splitext(clean_name)[0]    # drop extension (.JPG etc.)
        id_to_image[str(task_id)] = clean_name

# Collect all mask files
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

mask_files = [
    f for f in os.listdir(mask_folder)
    if f.lower().endswith(".png")
]
mask_files.sort(key=natural_sort_key)


# Create CSV file

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image-name", "mask-name", "class-name"])

    for mask_file in mask_files:
        match = re.match(r"mask(\d+)_(.+?)_(\d+)\.png", mask_file, flags=re.IGNORECASE)
        if not match:
            continue

        task_id, label, _ = match.groups()
        image_name = id_to_image.get(task_id, "Unknown")
        writer.writerow([image_name, mask_file, label])

print(f"Saved index file to {output_file}")
