from config import Config
from embeddings.image_embedding import ImageEmbedding
from vector_stores.qdrant import QdrantVectorStore
import re


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

    image_path = "durian_retrieval/dataset/disease_v2/14_Cercospora_Leaf_Spot/14_Cercospora_Leaf_Spot_eng/image-1.jpg"
    vector = embedder.embed(image_path)

    model_part = short_model_name(cfg.MODEL_NAME_OR_PATH)
    collection_name = f"{cfg.COLLECTION_NAME_PREFIX}_disease_{model_part}"

    hits = store.query(
        collection_name=collection_name,
        query_vector=vector,
        top_k=cfg.TOP_K,
    )

    for hit in hits:
        if hit.payload and "english_translation" in hit.payload:
            print(hit.payload["english_translation"], "score:", hit.score)
        else:
            print("No translation available", "score:", hit.score)
