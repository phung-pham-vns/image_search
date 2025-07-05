import json
from pathlib import Path
from config import Config
from embeddings.image_embedding import ImageEmbedding
from vector_stores.qdrant import QdrantVectorStore


def load_image_paths(image_dir: Path, extensions: tuple = ("*.jpg", "*.jpeg", "*.png")):
    """Recursively load image file paths with given extensions."""
    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.rglob(ext))
    return image_paths


def main():
    cfg = Config()
    embedder = ImageEmbedding(model_name_or_path=cfg.EMBEDDING_NAME, device=cfg.DEVICE)
    store = QdrantVectorStore(uri=cfg.QDRANT_URI)

    store.create_collection(
        collection_name=cfg.COLLECTION_NAME,
        embedding_size=cfg.EMEBDDING_DIM,
        distance="cosine",
    )

    image_dir = Path(
        "/Users/mac/Documents/PHUNGPX/MLOps_practice/durian_retrieval/dataset/processed_pests/images"
    )
    label_dir = Path(
        "/Users/mac/Documents/PHUNGPX/MLOps_practice/durian_retrieval/dataset/processed_pests/labels"
    )
    image_paths = load_image_paths(image_dir)

    if not image_paths:
        print("No image files found. Please check the dataset path.")
        return

    vectors, payloads, ids = [], [], []

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.json"
        if not label_path.exists():
            print(f"Warning: Label file not found for {image_path.name}")
            continue

        vector = embedder.embed(str(image_path))
        with label_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
            metadata["image_name"] = image_path.name

        if vector is None:
            print(str(image_path))
            continue

        vectors.append(vector)
        payloads.append(metadata)
        ids.append(image_path.stem)

    print(f"[INFO] Total images embedded: {len(vectors)}")

    if vectors:
        store.add_embeddings(
            collection_name=cfg.COLLECTION_NAME,
            embeddings=vectors,
            payloads=payloads,
            ids=ids,
        )
        print("[INFO] Embeddings successfully added to Qdrant.")
    else:
        print("[INFO] No valid embeddings to upload.")


if __name__ == "__main__":
    main()
