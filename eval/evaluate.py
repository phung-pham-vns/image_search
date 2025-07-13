import re
import json
import time
from PIL import Image
from pathlib import Path
from natsort import natsorted

from src.config import Config
from src.constants import EMBEDDING_MODELS
from src.core.image_search import ImageSearcher


disease_classes = {
    v: k for k, v in json.load(open("eval/classes/diseases.json")).items()
}
# pest_classes = {v: k for k, v in json.load(open("eval/classes/pests.json")).items()}
pest_classes = {}


def short_model_name(model_name_or_path: str) -> str:
    """Generates a normalized, short model name for naming collections."""
    name = model_name_or_path.lower()
    name = re.sub(r"^.*[\\/]|openai/|google/|facebook/", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def evaluate(
    input_folder: str = "dataset/images/250610_dataset/valid_processed",
    output_folder: str = "dataset/images/250610_dataset/valid_topK",
    categories: list[str] = ["pest", "disease"],
    list_top_k: list[int] = [1, 3, 5, 10],
    embedding_name: str = "SigLIP2 Base",
    collection_name_prefix: str = "durian_v2",
) -> list[dict]:
    # Initialize the Iamge Searcher
    if "tulip" in embedding_name.lower():
        embedding_model_path = embedding_name
    else:
        embedding_model_path = EMBEDDING_MODELS[embedding_name]["model_path"]
    config = Config(
        MODEL_NAME_OR_PATH=embedding_model_path,
        COLLECTION_NAME_PREFIX=collection_name_prefix,
    )
    searcher = ImageSearcher(config)

    save_dir = Path(output_folder)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for category in categories:
        input_image_folders = Path(input_folder) / category / "images"
        input_label_folders = Path(input_folder) / category / "labels"

        image_paths = natsorted(list(input_image_folders.glob("*.jpg")))
        label_paths = natsorted(list(input_label_folders.glob("*.json")))

        pair_paths = []
        for image_path, label_path in zip(image_paths, label_paths):
            if image_path.stem != label_path.stem:
                raise ValueError(
                    f"Image and label file names do not match: {image_path.stem} != {label_path.stem}"
                )
            pair_paths.append((image_path, label_path))

        print(f"The number of samples of {category} is {len(pair_paths)}")

        for image_path, label_path in pair_paths:
            with open(label_path, "r") as f:
                label = json.load(f)

            classe_name = label["english_translation"]
            image = Image.open(image_path)

            if category == "pest":
                class_name = pest_classes.get(classe_name, classe_name)
            elif category == "disease":
                class_name = disease_classes.get(classe_name, classe_name)

            for top_k in list_top_k:
                t1 = time.time()
                hits = searcher.search(image, category, top_k)
                t2 = time.time()

                processed_hits = []
                for hit in hits:
                    english_name = hit.get("payload", {}).get("english_translation")
                    if category == "pest":
                        pred_class = pest_classes.get(english_name, english_name)
                    elif category == "disease":
                        pred_class = disease_classes.get(english_name, english_name)
                    processed_hits.append(
                        {"score": hit.get("score"), "class_name": pred_class}
                    )

                results.append(
                    {
                        "model_path": embedding_model_path,
                        "category": category,
                        "class_name": class_name,
                        "topk": top_k,
                        "time_second": t2 - t1,
                        "hits": processed_hits,
                    }
                )

    with open(save_dir / f"{short_model_name(embedding_model_path)}.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    import json

    embedding_names = [
        # "SigLIP2 Base",
        # "SigLIP2 Large",
        # "CLIP ViT-L/14",
        # "CLIP ViT-B/32",
        # "DINOv2 ViT-B/14",
        # "DINOv2 ViT-L/14",
        "TULIP-B-16-224",
        "TULIP-so400m-14-384",
    ]

    for embedding_name in embedding_names:
        results = evaluate(
            input_folder="dataset/images/250610_dataset/valid_processed",
            output_folder="dataset/images/250610_dataset/valid_topK",
            categories=["pest", "disease"],
            list_top_k=[1, 3, 5, 10],
            embedding_name=embedding_name,
            collection_name_prefix="durian_v2",
        )

        print(json.dumps(results, indent=4))
