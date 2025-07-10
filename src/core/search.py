import re
from pathlib import Path
from typing import List, Dict, Optional, Any

from PIL import Image

from .config import Config
from .embeddings import ImageEmbedding
from .vector_store import QdrantVectorStore


class ImageSearcher:
    """Encapsulates the logic for image similarity search."""

    def __init__(self, config: Config):
        """
        Initializes the searcher with a configuration.

        Args:
            config: A Config object with all necessary settings.
        """
        self.config = config
        self.embedder = ImageEmbedding(
            model_name_or_path=self.config.MODEL_NAME_OR_PATH,
            device=self.config.DEVICE,
        )
        self.store = QdrantVectorStore(uri=self.config.QDRANT_URI)

    def _short_model_name(self, model_name_or_path: str) -> str:
        """Sanitizes and shortens a model name for collection naming."""
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
        Performs a similarity search for a given query image.

        Args:
            query_image: The image to search with.
            category: The data category to search in (e.g., 'disease').
            top_k: The number of results to return. Uses config default if None.

        Returns:
            A list of dictionaries, each representing a search result.
        """
        if top_k is None:
            top_k = self.config.TOP_K

        # Generate embedding for the query image
        query_embedding = self.embedder.embed(query_image)
        if query_embedding is None:
            # Handle embedding failure
            return []

        # Determine the collection name
        model_part = self._short_model_name(self.config.MODEL_NAME_OR_PATH)
        collection_name = (
            f"{self.config.COLLECTION_NAME_PREFIX}_{category}_{model_part}"
        )

        # Query the vector store
        hits = self.store.query(
            collection_name=collection_name,
            query_vector=query_embedding,
            top_k=top_k,
        )

        # Process and return the results
        return self._process_results(hits, category)

    def _process_results(self, hits: List[Any], category: str) -> List[Dict[str, Any]]:
        """Formats the raw search hits into a structured list."""
        processed_results = []
        image_dir = self.config.PROCESSED_DATASET_DIR / category / "images"

        for hit in hits:
            payload = hit.payload or {}
            img_name = payload.get("image_name")
            if not img_name:
                continue

            result_img_path = image_dir / img_name
            if result_img_path.exists():
                processed_results.append(
                    {
                        "image_path": result_img_path,
                        "score": hit.score,
                        "payload": payload,
                    }
                )
        return processed_results
