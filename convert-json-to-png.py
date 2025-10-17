# JSON to PNG Mask Converter
# -----------------------------------------------------------------------------
# This script converts polygon annotations exported from a labeling tool (Label Studio) into binary mask PNGs.
# Each annotation polygon is drawn as a white region (255) on a black background (0). Also creates a background mask (the inverse of all combined masks)
# Output images are single-channel (grayscale) and lossless PNG files.

#Import dependencies
import json
from PIL import Image, ImageDraw, ImageChops
import os

# Create a folder to store converted png masks if one does not exist
#def json_to_convert(json_path, output_dir=r"path/to/your/directory"): # Change to create directory in desired location
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        tasks = json.load(f)  # List of annotation tasks

    for task in tasks:
        task_id = task.get("id", "unknown")
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        # Collect all polygon annotations for this task
        results = annotations[0].get("result", [])
        if not results:
            continue

        # Use the first result to get dimensions
        width = results[0].get("original_width")
        height = results[0].get("original_height")

        # Base blank image for background calculation
        combined_mask = Image.new('L', (width, height), 0)

        for i, result in enumerate(results):
            value = result.get("value", {})
            if 'points' not in value:
                continue

            points = value["points"]
            polygon = [(x / 100 * width, y / 100 * height) for x, y in points]

            # Create per-class binary mask
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(polygon, outline=255, fill=255)

            # Add this polygon to the combined mask (for background inversion later)
            combined_mask = ImageChops.lighter(combined_mask, mask)

            # Extract class label
            labels = value.get("polygonlabels", [])
            label = labels[0] if labels else "unlabeled"
            label_clean = "".join(c if c.isalnum() else "_" for c in label)

            # Save per-class polygon mask
            filename = f"mask{task_id}_{label_clean}_{i}.png"
            output_path = os.path.join(output_dir, filename)
            mask.save(output_path, format='PNG', compress_level=0)
            print(f"Saved class mask: {output_path}")

        # Create background mask = inverse of combined foreground
        background_mask = ImageChops.invert(combined_mask)
        background_filename = f"mask{task_id}_background.png"
        background_output = os.path.join(output_dir, background_filename)
        background_mask.save(background_output, format='PNG', compress_level=0)
        print(f"Saved background mask: {background_output}")

# Link to JSON file ready for conversion
#json_to_convert(r"path/to/your/file.json")  # Change to yor specified JSON file
