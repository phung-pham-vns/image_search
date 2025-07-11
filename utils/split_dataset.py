import os
import shutil
import random
from pathlib import Path

# Set paths
main_category = "disease"
SOURCE_DIR = Path(
    f"/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/updated_dataset/{main_category}"
)
SAVE_DIR = Path(
    "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/split_dataset"
)
SPLIT_RATIO = 0.7  # 70% train, 30% val
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Ensure destination directories exist
train_dir = SAVE_DIR / "train" / main_category
valid_dir = SAVE_DIR / "valid" / main_category


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


stats = {}
# Traverse each disease subfolder
for category_name in os.listdir(SOURCE_DIR):
    images_dir = SOURCE_DIR / category_name / "images"
    json_paths = list(Path(SOURCE_DIR / category_name).glob("*.json"))
    if not len(json_paths):
        print(f"No json file found for {category_name}")

    if not os.path.isdir(images_dir):
        continue

    # List all image files
    all_images = [f for f in os.listdir(images_dir) if is_image_file(f)]
    random.shuffle(all_images)

    # Split
    split_idx = int(len(all_images) * SPLIT_RATIO)
    train_images = all_images[:split_idx]
    valid_images = all_images[split_idx:]

    # Create corresponding output dirs
    train_img_dir = train_dir / category_name / "images"
    valid_img_dir = valid_dir / category_name / "images"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    valid_img_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy(os.path.join(images_dir, img), os.path.join(train_img_dir, img))
    for img in valid_images:
        shutil.copy(os.path.join(images_dir, img), os.path.join(valid_img_dir, img))

    if len(json_paths) > 1:
        print(f"Multiple json files found for {category_name}")
    else:
        shutil.copy(json_paths[0], train_dir / category_name / f"{category_name}.json")
        shutil.copy(json_paths[0], valid_dir / category_name / f"{category_name}.json")

    print(f"{category_name}: {len(train_images)} train, {len(valid_images)} valid")

print("Split complete.")
