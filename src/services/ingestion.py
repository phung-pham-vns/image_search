import re
import json
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union, cast
from dataclasses import dataclass

from src.config import Config
from src.constants import EMBEDDING_MODELS
from src.core.embeddings import ImageEmbedding
from src.core.vector_store import QdrantVectorStore


@dataclass
class ImageData:
    """Data class to hold image processing results."""

    image_path: Path
    label_path: Path
    vector: Optional[List[float]]
    metadata: Dict[str, Any]
    image_id: str


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""

    pass


class DataIngester:
    """Handles the ingestion of image data into vector storage."""

    def __init__(self, config: Config):
        """
        Initialize the data ingester.

        Args:
            config: Configuration object containing all necessary settings
        """
        self.config = config
        self.logger = self._setup_logging()
        self.embedder = None
        self.store = None
        self._validate_config()
        self._initialize_components()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=self.config.LOG_LEVEL,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.LOG_FILE),
            ],
        )
        return logging.getLogger(__name__)

    def _validate_config(self) -> None:
        """Validate configuration settings."""
        if not self.config.PROCESSED_DATASET_DIR:
            raise DataIngestionError("PROCESSED_DATASET_DIR is not configured")

        if not self.config.QDRANT_URI:
            raise DataIngestionError("QDRANT_URI is not configured")

        if not self.config.MODEL_NAME_OR_PATH:
            raise DataIngestionError("MODEL_NAME_OR_PATH is not configured")

    def _short_model_name(self, model_name_or_path: str) -> str:
        """Sanitize and shorten model name for use in collection name."""
        name = model_name_or_path.lower()
        name = re.sub(r"^.*[\\/]|openai/|google/|facebook/", "", name)
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return name.strip("_")

    def _initialize_components(self) -> None:
        """Initialize embedding model and vector store."""
        try:
            self.embedder = ImageEmbedding(
                model_name_or_path=self.config.MODEL_NAME_OR_PATH,
                device=self.config.DEVICE,
            )
            self.store = QdrantVectorStore(uri=self.config.QDRANT_URI)
            self.logger.info(
                "Successfully initialized embedding model and vector store"
            )
        except Exception as e:
            raise DataIngestionError(f"Failed to initialize components: {e}")

    def load_image_paths(self, image_dir: Path) -> List[Path]:
        """
        Recursively load image file paths with given extensions.
        """
        if not image_dir.exists():
            raise DataIngestionError(f"Image directory does not exist: {image_dir}")

        image_paths = []
        for ext in self.config.VALID_IMAGE_EXTENSIONS:
            image_paths.extend(image_dir.rglob(f"*{ext}"))

        if not image_paths:
            raise DataIngestionError(f"No image files found in {image_dir}")

        self.logger.info(f"Found {len(image_paths)} image files in {image_dir}")
        return sorted(image_paths)

    def _load_metadata(self, label_path: Path, image_name: str) -> Dict[str, Any]:
        """
        Load metadata from label file.
        """
        try:
            with label_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
                metadata["image_name"] = image_name
                return metadata
        except FileNotFoundError:
            raise DataIngestionError(f"Label file not found: {label_path}")
        except json.JSONDecodeError as e:
            raise DataIngestionError(f"Invalid JSON in label file {label_path}: {e}")
        except Exception as e:
            raise DataIngestionError(f"Error reading label file {label_path}: {e}")

    def _process_single_image(
        self,
        image_path: Path,
        label_dir: Path,
    ) -> Optional[ImageData]:
        """
        Process a single image and its metadata.
        """
        try:
            label_path = label_dir / f"{image_path.stem}.json"
            metadata = self._load_metadata(label_path, image_path.name)
            vector = self._get_embedding(image_path)

            if vector is None:
                self.logger.warning(
                    f"Failed to generate embedding for {image_path.name}"
                )
                return None

            return ImageData(
                image_path=image_path,
                label_path=label_path,
                vector=vector.tolist(),
                metadata=metadata,
                image_id=image_path.stem,
            )
        except DataIngestionError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error processing {image_path.name}: {e}")
            return None

    def _get_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """Generate embedding for a single image."""
        if self.embedder is None:
            raise DataIngestionError("Embedder not initialized")
        return self.embedder.embed(str(image_path))

    def _process_images_batch(
        self, image_paths: List[Path], label_dir: Path
    ) -> List[ImageData]:
        """
        Process images in batches for better performance and memory management.
        """
        processed_data = []
        batch_size = self.config.BATCH_SIZE

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i : i + batch_size]
            batch_data = []

            for image_path in batch_paths:
                try:
                    image_data = self._process_single_image(image_path, label_dir)
                    if image_data:
                        batch_data.append(image_data)
                except DataIngestionError as e:
                    self.logger.error(f"Skipping {image_path.name}: {e}")
                    continue

            processed_data.extend(batch_data)

        return processed_data

    def _prepare_upload_data(
        self, processed_data: List[ImageData]
    ) -> Tuple[List[List[float]], List[Dict[str, Any]], List[str]]:
        """
        Prepare data for upload to vector store.
        """
        vectors = [data.vector for data in processed_data if data.vector is not None]
        payloads = [data.metadata for data in processed_data if data.vector is not None]
        ids = [data.image_id for data in processed_data if data.vector is not None]

        return vectors, payloads, ids

    def _collection_name(self, category: str) -> str:
        """Generate a unique collection name including model info."""
        model_part = self._short_model_name(self.config.MODEL_NAME_OR_PATH)
        return f"{self.config.COLLECTION_NAME_PREFIX}_{category}_{model_part}"

    def _create_collection(self, category: str) -> None:
        """
        Create vector store collection for the given category.
        """
        collection_name = self._collection_name(category)

        try:
            if self.store is None:
                raise DataIngestionError("Vector store not initialized")
            self.store.create_collection(
                collection_name=collection_name,
                embedding_size=self.config.EMEBDDING_DIM,
                distance=self.config.DISTANCE_METRIC,
            )
            self.logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            raise DataIngestionError(
                f"Failed to create collection {collection_name}: {e}"
            )

    def _upload_embeddings(
        self,
        category: str,
        vectors: List[List[float]],
        payloads: List[Dict],
        ids: List[str],
    ) -> None:
        """
        Upload embeddings to vector store.
        """
        collection_name = self._collection_name(category)

        try:
            if self.store is None:
                raise DataIngestionError("Vector store not initialized")
            numpy_vectors = [np.array(vector) for vector in vectors]
            typed_ids = cast(List[Union[str, int]], ids)
            self.store.add_embeddings(
                collection_name=collection_name,
                embeddings=numpy_vectors,
                payloads=payloads,
                ids=typed_ids,
            )
            self.logger.info(
                f"Successfully uploaded {len(vectors)} embeddings to {collection_name}"
            )
        except Exception as e:
            raise DataIngestionError(
                f"Failed to upload embeddings to {collection_name}: {e}"
            )

    def ingest_category(self, category: str) -> None:
        """
        Ingest all images for a specific category.
        """
        self.logger.info(f"Starting ingestion for category: {category}")

        image_dir = self.config.PROCESSED_DATASET_DIR / category / "images"
        label_dir = self.config.PROCESSED_DATASET_DIR / category / "labels"

        image_paths = self.load_image_paths(image_dir)
        self._create_collection(category)
        processed_data = self._process_images_batch(image_paths, label_dir)

        if not processed_data:
            self.logger.warning(f"No valid images processed for category {category}")
            return

        vectors, payloads, ids = self._prepare_upload_data(processed_data)
        self._upload_embeddings(category, vectors, payloads, ids)

        self.logger.info(
            f"Successfully ingested {len(processed_data)} images for category {category}"
        )

    def run(self, categories: List[str]) -> None:
        """
        Run the complete ingestion process for multiple categories.
        """
        self.logger.info("Starting data ingestion process")
        try:
            for category in categories:
                try:
                    self.ingest_category(category)
                except DataIngestionError as e:
                    self.logger.error(f"Failed to ingest category {category}: {e}")
                    continue
            self.logger.info("Data ingestion process completed")
        except Exception as e:
            self.logger.error(f"Fatal error during ingestion: {e}")
            raise


def main(
    categories: Optional[List[str]] = None,
    embedding_name: str = "SigLIP2 Base",
    collection_name_prefix: str = "durian",
) -> None:
    """Main function to run the data ingestion process."""
    if categories is None:
        categories = ["disease", "pest"]

    embedding_model = EMBEDDING_MODELS[embedding_name]
    embedding_model_path = embedding_model["model_path"]
    embedding_size = embedding_model["embedding_size"]

    config = Config()
    config.MODEL_NAME_OR_PATH = embedding_model_path
    config.EMEBDDING_DIM = embedding_size
    config.COLLECTION_NAME_PREFIX = collection_name_prefix

    ingester = DataIngester(config)

    try:
        ingester.run(categories)
    except Exception as e:
        logging.error(f"Failed to complete ingestion: {e}")
        raise


if __name__ == "__main__":
    embedding_names = [
        # "SigLIP2 Base",
        "SigLIP2 Large",
        # "CLIP ViT-B/32",
        # "DINOv2 ViT-B/14",
        # "DINOv2 ViT-L/14",
    ]
    for embedding_name in embedding_names:
        main(
            embedding_name=embedding_name,
            collection_name_prefix="durian_v2",
        )
