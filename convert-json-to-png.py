# JSON to PNG Mask Converter
# -----------------------------------------------------------------------------
# This script converts polygon annotations exported from a labeling tool (Label Studio) into binary mask PNGs.
# Each annotation polygon is drawn as a white region (255) on a black background (0).
# Output images are single-channel (grayscale) and lossless PNG files.

#Import dependencies
import json
from PIL import Image, ImageDraw
import os

# Create a folder to store converted png masks if one does not exist
#def json_to_convert(json_path, output_dir=r"path/to/your/directory"): # Change to create directory in desired location
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        tasks = json.load(f)  # This is a list of annotation tasks

    # Iterate through each annotated task
    for task in tasks:
        task_id = task.get("id", "unknown")
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        # Extract polygon results from the first annotation entry
        results = annotations[0].get("result", [])
        for i, result in enumerate(results):
            if 'value' not in result or 'points' not in result['value']:
                continue

            # Get original image dimensions
            width = result['original_width']
            height = result['original_height']
            points = result['value']['points']

            # Polygon coordinates are stored as percentages (0–100) convert them to absolute pixel coordinates
            polygon = [(x / 100 * width, y / 100 * height) for x, y in points]

            # Create blank mask and draw polygon
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(polygon, outline=255, fill=255)

            # Extract label (class name) from JSON
            labels = result['value'].get('polygonlabels', [])
            label = labels[0] if labels else 'unlabeled'

            # Sanitise label for safe filename
            label_clean = "".join(c if c.isalnum() else "_" for c in label)

            # Build filename: image ID (NOT ORIGINAL FILENAME) + class label + polygon index
            filename = f"Image{task_id}_{label_clean}_{i}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Save lossless PNG
            mask.save(output_path, format='PNG', compress_level=0)
            
            # Log saved File
            print(f"Saved: {output_path}") 

# Link to JSON file ready for conversion
json_to_convert(r"path/to/your/file.json")  # Change to yor specified JSON file
