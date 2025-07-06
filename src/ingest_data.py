import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union, cast
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import re

from .config import Config
from .embeddings.image_embedding import ImageEmbedding
from .vector_stores.qdrant import QdrantVectorStore


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
        # Validate configuration
        self._validate_config()
        # Initialize components
        self._initialize_components()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler("ingestion.log")],
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
        # Remove common path/prefixes and non-alphanumeric
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

    def load_image_paths(
        self,
        image_dir: Path,
        extensions: Tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png"),
    ) -> List[Path]:
        """
        Recursively load image file paths with given extensions.

        Args:
            image_dir: Directory containing images
            extensions: Tuple of file extensions to include

        Returns:
            List of image file paths

        Raises:
            DataIngestionError: If image directory doesn't exist or is empty
        """
        if not image_dir.exists():
            raise DataIngestionError(f"Image directory does not exist: {image_dir}")

        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.rglob(ext))

        if not image_paths:
            raise DataIngestionError(f"No image files found in {image_dir}")

        self.logger.info(f"Found {len(image_paths)} image files in {image_dir}")
        return sorted(image_paths)

    def _load_metadata(self, label_path: Path, image_name: str) -> Dict[str, Any]:
        """
        Load metadata from label file.

        Args:
            label_path: Path to the label JSON file
            image_name: Name of the image file

        Returns:
            Dictionary containing metadata with image_name added

        Raises:
            DataIngestionError: If label file is invalid or missing
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
        self, image_path: Path, label_dir: Path
    ) -> Optional[ImageData]:
        """
        Process a single image and its metadata.

        Args:
            image_path: Path to the image file
            label_dir: Directory containing label files

        Returns:
            ImageData object if processing successful, None otherwise
        """
        try:
            label_path = label_dir / f"{image_path.stem}.json"

            # Load metadata
            metadata = self._load_metadata(label_path, image_path.name)

            # Generate embedding
            if self.embedder is None:
                raise DataIngestionError("Embedder not initialized")
            vector = self.embedder.embed(str(image_path))
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
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error processing {image_path.name}: {e}")
            return None

    def _process_images_batch(
        self, image_paths: List[Path], label_dir: Path, batch_size: int = 32
    ) -> List[ImageData]:
        """
        Process images in batches for better performance and memory management.

        Args:
            image_paths: List of image paths to process
            label_dir: Directory containing label files
            batch_size: Number of images to process in each batch

        Returns:
            List of successfully processed ImageData objects
        """
        processed_data = []

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

            # Log progress
            if (i + batch_size) % (batch_size * 5) == 0:
                self.logger.info(f"Processed {len(processed_data)} images so far")

        return processed_data

    def _prepare_upload_data(
        self, processed_data: List[ImageData]
    ) -> Tuple[List[List[float]], List[Dict[str, Any]], List[str]]:
        """
        Prepare data for upload to vector store.

        Args:
            processed_data: List of processed ImageData objects

        Returns:
            Tuple of (vectors, payloads, ids) for upload
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

        Args:
            category: Category name for the collection
        """
        collection_name = self._collection_name(category)

        try:
            if self.store is None:
                raise DataIngestionError("Vector store not initialized")
            self.store.create_collection(
                collection_name=collection_name,
                embedding_size=self.config.EMEBDDING_DIM,
                distance="cosine",
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

        Args:
            category: Category name for the collection
            vectors: List of embedding vectors
            payloads: List of metadata payloads
            ids: List of image IDs
        """
        collection_name = self._collection_name(category)

        try:
            if self.store is None:
                raise DataIngestionError("Vector store not initialized")
            # Convert vectors to numpy arrays
            numpy_vectors = [np.array(vector) for vector in vectors]
            # Cast ids to the expected type
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

    def ingest_category(self, category: str, batch_size: int = 32) -> None:
        """
        Ingest all images for a specific category.

        Args:
            category: Category name (e.g., 'disease', 'pest')
            batch_size: Number of images to process in each batch
        """
        self.logger.info(f"Starting ingestion for category: {category}")

        # Set up paths
        image_dir = Path(f"{self.config.PROCESSED_DATASET_DIR}/{category}/images")
        label_dir = Path(f"{self.config.PROCESSED_DATASET_DIR}/{category}/labels")

        # Load image paths
        image_paths = self.load_image_paths(image_dir)

        # Create collection
        self._create_collection(category)

        # Process images
        processed_data = self._process_images_batch(image_paths, label_dir, batch_size)

        if not processed_data:
            self.logger.warning(f"No valid images processed for category {category}")
            return

        # Prepare and upload data
        vectors, payloads, ids = self._prepare_upload_data(processed_data)
        self._upload_embeddings(category, vectors, payloads, ids)

        self.logger.info(
            f"Successfully ingested {len(processed_data)} images for category {category}"
        )

    def run(self, categories: List[str], batch_size: int = 32) -> None:
        """
        Run the complete ingestion process for multiple categories.

        Args:
            categories: List of category names to process
            batch_size: Number of images to process in each batch
        """
        self.logger.info("Starting data ingestion process")

        try:
            # Process each category
            for category in categories:
                try:
                    self.ingest_category(category, batch_size)
                except DataIngestionError as e:
                    self.logger.error(f"Failed to ingest category {category}: {e}")
                    continue

            self.logger.info("Data ingestion process completed")

        except Exception as e:
            self.logger.error(f"Fatal error during ingestion: {e}")
            raise


def main(categories: Optional[List[str]] = None, batch_size: int = 32) -> None:
    """
    Main function to run the data ingestion process.

    Args:
        categories: List of categories to process. If None, defaults to ['disease', 'pest']
        batch_size: Number of images to process in each batch
    """
    if categories is None:
        categories = ["disease", "pest"]

    config = Config()
    ingester = DataIngester(config)

    try:
        ingester.run(categories, batch_size)
    except Exception as e:
        logging.error(f"Failed to complete ingestion: {e}")
        raise


if __name__ == "__main__":
    main()
