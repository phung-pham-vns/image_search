import re
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import Config
from src.core.embeddings import ImageEmbedding
from src.core.vector_store import QdrantVectorStore


def short_model_name(model_name_or_path: str) -> str:
    name = model_name_or_path.lower()
    name = re.sub(r"^.*[\\/]|openai/|google/|facebook/", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


if __name__ == "__main__":
    cfg = Config()
    embedder = ImageEmbedding(
        model_name_or_path=cfg.MODEL_NAME_OR_PATH,
        device=cfg.DEVICE,
    )
    store = QdrantVectorStore(uri=cfg.QDRANT_URI)

    # Note: This image path is hardcoded and may need to be updated or passed as an argument.
    image_path = "dataset/images/testing/1/images/Phytophthora_1.jpeg"
    vector = embedder.embed(image_path)

    model_part = short_model_name(cfg.MODEL_NAME_OR_PATH)
    collection_name = f"{cfg.COLLECTION_NAME_PREFIX}_disease_{model_part}"

    print(f"Querying collection '{collection_name}'...")
    hits = store.query(
        collection_name=collection_name,
        query_vector=vector,
        top_k=cfg.TOP_K,
    )

    print(f"Found {len(hits)} results:")
    for hit in hits:
        print(f"  - ID: {hit.id}, Score: {hit.score:.4f}")
        if hit.payload:
            print("    Payload:", hit.payload)
        else:
            print("    No payload.")
