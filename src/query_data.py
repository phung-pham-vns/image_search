from config import Config
from embeddings.image_embedding import ImageEmbedding
from vector_stores.qdrant import QdrantVectorStore


if __name__ == "__main__":
    cfg = Config()
    embedder = ImageEmbedding(model_name_or_path=cfg.EMBEDDING_NAME, device=cfg.DEVICE)
    store = QdrantVectorStore(uri=cfg.QDRANT_URI)

    image_path = "durian_retrieval/dataset/disease_v2/14_Cercospora_Leaf_Spot/14_Cercospora_Leaf_Spot_eng/image-1.jpg"
    vector = embedder.embed(image_path)

    hits = store.query(
        collection_name=cfg.COLLECTION_NAME,
        query_vector=vector,
        top_k=cfg.TOP_K,
    )

    for hit in hits:
        print(hit.payload["english_translation"], "score:", hit.score)
