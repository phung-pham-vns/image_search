import json
from pathlib import Path
from collections import defaultdict

data_dir = Path("dataset/images/updated_dataset")

image_paths = list(data_dir.glob("**/*.jpg"))

stats = dict()
for image_path in image_paths:
    category = image_path.parents[2].stem
    if category not in stats:
        stats[category] = defaultdict(int)
    stats[category][image_path.parents[1].stem] += 1

print(json.dumps(stats, indent=4))
