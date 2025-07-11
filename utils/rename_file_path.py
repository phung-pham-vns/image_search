import os
import uuid
from pathlib import Path
from PIL import Image

# Root dataset folder
root_dir = Path(
    "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/updated_dataset/pest"
)

# Allowed image file extensions
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".avif"]

# Walk through each disease folder
for disease_folder in root_dir.iterdir():
    images_dir = disease_folder / "images"
    if images_dir.is_dir():
        for img_file in images_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in valid_extensions:
                try:
                    # Generate new UUID-based filename
                    new_name = f"{uuid.uuid4()}.jpg"
                    new_path = images_dir / new_name

                    # Open image and save as .jpg to ensure correct format
                    with Image.open(img_file) as img:
                        img.convert("RGB").save(new_path, "JPEG")

                    # Delete original image
                    img_file.unlink()

                    print(f"✅ Renamed {img_file.name} -> {new_name}")
                except Exception as e:
                    print(f"❌ Failed to process {img_file}: {e}")
