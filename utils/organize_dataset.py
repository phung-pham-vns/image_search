import os
import shutil
import json
from uuid import uuid4
from pathlib import Path

# Input and output folders
input_root = Path(
    "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/split_dataset/train/pest"
)
output_root = Path(
    "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/split_dataset/train_processed/pest"
)

output_images = output_root / "images"
output_jsons = output_root / "labels"

output_images.mkdir(parents=True, exist_ok=True)
output_jsons.mkdir(parents=True, exist_ok=True)

image_suffixes = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
disease_stats = {}

# Loop through disease folders (e.g., 1/, 2/, ...)
for disease_folder in input_root.iterdir():
    if disease_folder.is_dir():
        image_paths = []
        for image_suffix in image_suffixes:
            image_paths += list(disease_folder.glob(f"**/*{image_suffix}"))

        disease_stats[disease_folder.stem] = len(image_paths)

        json_paths = list(disease_folder.glob(f"**/*.json"))

        # Load info.json once
        with open(json_paths[0], "r", encoding="utf-8") as f:
            info_data = json.load(f)

        # Iterate over each image file
        for image_file in image_paths:
            # Generate new UUID filename
            uid = str(uuid4())
            new_image_path = output_images / f"{uid}{image_file.suffix}"
            new_json_path = output_jsons / f"{uid}.json"

            # Copy image
            shutil.copy2(image_file, new_image_path)

            # Write duplicated info.json with new name
            with open(new_json_path, "w", encoding="utf-8") as out_json:
                json.dump(info_data, out_json, indent=2, ensure_ascii=False)

# Sort by the integer prefix before the underscore
# sorted_stats = dict(
#     sorted(disease_stats.items(), key=lambda x: int(x[0].split("_")[0]))
# )
print(f"Disease Stats: {json.dumps(disease_stats, indent=4, ensure_ascii=False)}")
print("✅ Done: All images and JSONs have been flattened and renamed with UUIDs.")
