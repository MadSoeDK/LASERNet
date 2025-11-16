import os
import shutil


INPUT_DIR = "data/laser_raw"
OUTPUT_DIR = "data/temp"
END_WITH = "_t_1.tiff"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

for folder in os.listdir(INPUT_DIR):
    if os.path.isdir(os.path.join(INPUT_DIR, folder)):
        for file in os.listdir(os.path.join(INPUT_DIR, folder)):
            if file.endswith(END_WITH):
                shutil.copy(
                    os.path.join(INPUT_DIR, folder, file),
                    os.path.join(OUTPUT_DIR, file),
                )
                print(f"Copied {file} to {OUTPUT_DIR}")
