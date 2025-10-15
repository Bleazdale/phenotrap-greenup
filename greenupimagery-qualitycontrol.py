# Quality Control Procedure for Camera Trap Imagery used in Vegetation Green-up Monitoring
# Developed and tested using Python 3.12
# Link to folder (root directory) located at the end of the script

# Import Dependencies
import os
import cv2
import numpy as np
import shutil

# === Function Toggles (on/off switches) ===
RUN_NIR_DETECTION = True
RUN_BLUR_DETECTION = True
RUN_DEEP_SNOW_DETECTION = True
RUN_LOW_SATURATION_DETECTION = False
RUN_BLUE_SNOW_DETECTION = True

# Function to separate RGB and NIR imagery through greyscale and low saturation detection
def separate_nir_images(root_dir):
    os.makedirs(nir_dir, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

    for subdir, _, files in os.walk(root_dir):
        if nir_dir in subdir:
            continue  # Skip already moved NIR images

        for file in files:
            if not file.lower().endswith(image_extensions):
                continue

            file_path = os.path.join(subdir, file)
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            if image is None:
                print(f"Skipping unreadable file: {file_path}")
                continue

            # Greyscale images go to NIR
            if len(image.shape) == 2:
                dest_path = os.path.join(nir_dir, file)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(file_path, dest_path)
                print(f"Classified grayscale {file} as NIR.")
                continue

            # Extract  horizontal center row

            height, width, _ = image.shape
            center_y = height // 2
            start_x = max(0, (width - 500) // 2)
            end_x = min(width, start_x + 500)


            center_strip = image[center_y:center_y+1, start_x:end_x]
            hsv_strip = cv2.cvtColor(center_strip, cv2.COLOR_BGR2HSV)
            saturation = hsv_strip[:, :, 1]

            if (saturation > 20).any():   # Remove any imagery that does not have a saturation greater than 20
                print(f"Image {file} appears to be RGB. Leaving in place.")
            else:
                dest_path = os.path.join(nir_dir, file) # or keep in the current folder
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(file_path, dest_path)
                print(f"Classified {file} as NIR based on low saturation.")


# Function to separate blurred and dark imagery from clear via Laplacian
def is_blurry(image, threshold=150): # threshold=50 for high-threshold blurring, threshold=200 for low-threshold blurring and darkening.  Lower threshold = stricter blur detection.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


# Function to separate deep snow imagery from original dataset

def detect_deep_snow(
    image,
    std_threshold=12, # How flat the region must be -limited variance in pixels
    flat_row_std=12, # Row standard threshold to count as flat
    flat_row_ratio=0.6, # % of flat rows needed
    min_brightness=50, # Do not separate imagery with heavy shadows across lower image
    green_ratio_threshold=0.5 # Do not separate imagery if over half the pixels in the area are green
):

    height = image.shape[0]
    bottom = image[int(height * 0.9):, :, :] # Only consider the bottom 10% of imagery

    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    mean_brightness = np.mean(gray)
    row_std_devs = np.std(gray, axis=1)
    flat_rows_ratio = np.sum(row_std_devs < flat_row_std) / len(row_std_devs)

    # Check green pixel dominance in BGR
    b, g, r = cv2.split(bottom)
    green_mask = (g > r + 20) & (g > b + 20)
    green_ratio = np.sum(green_mask) / green_mask.size

    if green_ratio > green_ratio_threshold:
        return False  # Skip — likely vegetation

    return (
        std_dev < std_threshold and
        flat_rows_ratio > flat_row_ratio and
        mean_brightness > min_brightness
    )

# Function to separate imagery with low saturation - Dark, heavy snow, minimal foliage or live vegetation

def is_low_saturation(
    image,
    saturation_threshold=100, # Remove images with a saturation lower than 100
    low_sat_ratio_threshold=0.75, # Remove images if 75% of pixels are lower than 100
    green_hue_range=(35, 85), # Range of pixels considered green
    green_allowance=0.5 # Do not remove image if 50% of pixels are green
):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)

    # Mask for low-saturation pixels
    low_sat_mask = s < saturation_threshold
    low_sat_ratio = np.sum(low_sat_mask) / s.size

    if low_sat_ratio < low_sat_ratio_threshold:
        return False  # Not gray enough

    # Of the low-saturation pixels, how many are green?
    green_mask = (h >= green_hue_range[0]) & (h <= green_hue_range[1])
    green_in_low_sat = green_mask & low_sat_mask

    if np.sum(low_sat_mask) == 0:
        return False  # Avoid divide-by-zero

    green_ratio_in_low_sat = np.sum(green_in_low_sat) / np.sum(low_sat_mask)

    return green_ratio_in_low_sat < green_allowance

# Function to separate imagery with in heavy snow based on blue-dominant pixels, accounting for camera white balance

def detect_snow_by_blue_pixels(
    image,
    blue_ratio_threshold=0.4, # Remove images that contain 40% blue pixels
    green_hue_range=(35, 85), # Range of pixels considered green
    green_top_threshold=0.5, # Do not remove image if 50% of pixels are green
    verbose=False
):


    height = image.shape[0]
    bottom_50 = image[int(height * 0.5):, :, :] # Only consider the bottom 50% of imagery
    b, g, r = cv2.split(bottom_50)

    # Detect blue-dominant pixels
    blue_mask = (b > r + 20) & (b > g + 20)
    blue_ratio = np.sum(blue_mask) / blue_mask.size if blue_mask.size > 0 else 0

    if verbose:
        print(f"Blue ratio (bottom 50%): {blue_ratio:.3f}")

    if blue_ratio <= blue_ratio_threshold:
        return False

    # Check for green dominance in the top half
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    top_half = hsv[:height // 2, :, :]
    h_top, s_top, _ = cv2.split(top_half)

    green_mask = (h_top >= green_hue_range[0]) & (h_top <= green_hue_range[1]) & (s_top > 30)
    green_ratio_top = np.sum(green_mask) / green_mask.size if green_mask.size > 0 else 0

    if verbose:
        print(f"Green ratio (top half): {green_ratio_top:.3f}")

    if green_ratio_top > green_top_threshold:
        return False

    return True


def root_directory(root_dir):
    quality_control_dir = os.path.join(root_dir, 'Quality_Control') # Set destination location (Currently within Destination folder - This can be set to a different location manually by removing root_dir,)
    os.makedirs(quality_control_dir, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if not file.lower().endswith(image_extensions):
                continue

            file_path = os.path.join(dirpath, file)

            # Skip files already in Quality_Control
            if quality_control_dir in file_path:
                continue

            image = cv2.imread(file_path)
            if image is None:
                print(f"Could not read: {file_path}")
                continue

            print(f"Checked image: {file_path}")

            # Check all conditions
            should_move = False
            reasons = []

            # Run NIR Detection
            if RUN_NIR_DETECTION:
                if len(image.shape) == 2:
                    should_move = True
                    reasons.append("NIR_grayscale")
                else:
                    height, width, _ = image.shape
                    center_y = height // 2
                    start_x = max(0, (width - 500) // 2)
                    end_x = min(width, start_x + 500)

                    center_strip = image[center_y:center_y + 1, start_x:end_x]
                    hsv_strip = cv2.cvtColor(center_strip, cv2.COLOR_BGR2HSV)
                    saturation = hsv_strip[:, :, 1]

                    if not (saturation > 20).any():
                        should_move = True
                        reasons.append("NIR_low_saturation")

            # Run Blur Detection
            if RUN_BLUR_DETECTION and is_blurry(image):
                should_move = True
                reasons.append("blur")

            # Run Deep Snow Detection
            if RUN_DEEP_SNOW_DETECTION and detect_deep_snow(image):
                should_move = True
                reasons.append("snow_blockage")

            # Run Low Saturation (Limited Foliage) Detection
            if RUN_LOW_SATURATION_DETECTION and is_low_saturation(image):
                should_move = True
                reasons.append("low_saturation")

            # Run Blue (Heavy Snow) Detection
            if RUN_BLUE_SNOW_DETECTION and detect_snow_by_blue_pixels(image):
                should_move = True
                reasons.append("blue_snow")



# Establish link to your folder
root_directory(r"path/to/your/dataset") #Enter your desired folder here
