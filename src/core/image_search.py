import re
import logging
from typing import List, Dict, Optional, Any

from PIL import Image

from src.config import Config
from src.constants import EMBEDDING_MODELS
from src.core.embeddings import ImageEmbedding
from src.core.vector_store import QdrantVectorStore


class ImageSearcher:
    """Handles image similarity search using embeddings and vector storage."""

    def __init__(self, config: Config):
        """
        Initializes the ImageSearcher with configuration, embedder, and vector store.

        Args:
            config (Config): Configuration settings for embedding and vector storage.
        """
        self.config = config
        self.embedder = ImageEmbedding(
            model_name_or_path=self.config.MODEL_NAME_OR_PATH,
            device=self.config.DEVICE,
        )
        self.store = QdrantVectorStore(uri=self.config.QDRANT_URI)

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():  # Avoid duplicate handlers
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info("ImageSearcher initialized.")

    def _short_model_name(self, model_name_or_path: str) -> str:
        """Generates a normalized, short model name for naming collections."""
        name = model_name_or_path.lower()
        name = re.sub(r"^.*[\\/]|openai/|google/|facebook/", "", name)
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return name.strip("_")

    def search(
        self,
        query_image: Image.Image,
        category: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches for similar images using the query image.

        Args:
            query_image (Image.Image): Input image to search with.
            category (str): Category label (used for collection selection).
            top_k (Optional[int]): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: A list of matched results.
        """
        top_k = top_k or self.config.TOP_K
        self.logger.info(f"Starting search in category '{category}' with top_k={top_k}")

        query_embedding = self.embedder.embed(query_image)
        if query_embedding is None:
            self.logger.warning("Failed to generate embedding for query image.")
            return []

        model_key = self._short_model_name(self.config.MODEL_NAME_OR_PATH)
        collection_name = f"{self.config.COLLECTION_NAME_PREFIX}_{category}_{model_key}"

        try:
            hits = self.store.query(
                collection_name=collection_name,
                query_vector=query_embedding,
                top_k=top_k,
            )
            self.logger.info(f"Search returned {len(hits)} result(s).")
        except Exception as e:
            self.logger.error(f"Vector store query failed: {e}", exc_info=True)
            return []

        return self._process_results(hits, category)

    def _process_results(
        self,
        hits: List[Any],
        category: str,
    ) -> List[Dict[str, Any]]:
        """Processes raw vector hits into structured results."""
        processed = []
        image_dir = self.config.PROCESSED_DATASET_DIR / category / "images"

        for hit in hits:
            payload = hit.payload or {}
            image_name = payload.get("image_name")
            if not image_name:
                self.logger.warning("Hit missing image_name in payload; skipping.")
                continue

            image_path = image_dir / image_name
            if image_path.exists():
                processed.append(
                    {
                        "image_path": image_path,
                        "score": hit.score,
                        "payload": payload,
                    }
                )
            else:
                self.logger.warning(f"Image not found at path: {image_path}")

        self.logger.info(f"{len(processed)} results processed and validated.")
        return processed


def main(
    query_image: Image.Image,
    category: str,
    top_k: int = 5,
    embedding_name: str = "SigLIP2 Base",
    collection_name_prefix: str = "durian",
) -> List[Dict[str, Any]]:
    assert category in ["disease", "pest"], "Category must be either disease or pest"

    embedding_model = EMBEDDING_MODELS[embedding_name]
    embedding_model_path = embedding_model["model_path"]

    config = Config()
    config.MODEL_NAME_OR_PATH = embedding_model_path
    config.COLLECTION_NAME_PREFIX = collection_name_prefix

    searcher = ImageSearcher(config)

    try:
        hits = searcher.search(query_image, category, top_k)
        return hits
    except Exception as e:
        logging.error(f"Failed to complete ingestion: {e}")
        raise


if __name__ == "__main__":
    query_image = Image.open(
        "dataset/images/250610_dataset/valid/disease/algal_leaf_spot_red_rust/images/8e3e0636-6ce0-4eb6-948b-cf4381449809.jpg"
    )
    category = "disease"
    hits = main(query_image, category, collection_name_prefix="test")
    print(hits)
