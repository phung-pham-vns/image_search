import torch
from pathlib import Path


class Config:
    # Paths
    PROCESSED_DATASET_DIR = Path("dataset/images/processed_dataset")

    # Model settings
    MODEL_NAME_OR_PATH = "google/siglip2-base-patch16-224"  # Default model
    TULIP_MODEL_NAME = "TULIP-so400m-14-384"  # TULIP model option
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EMEBDDING_DIM = 512  # Default embedding dimension
    EMBEDDING_BATCH_SIZE = 32
    NORMALIZE_EMBEDDINGS = True

    # Vector store settings
    QDRANT_URI = "http://localhost:6333"
    COLLECTION_NAME_PREFIX = "durian"
    DISTANCE_METRIC = "cosine"  # Options: "cosine", "dot", "euclidean"
    TOP_K = 5  # Default number of results to return

    # Processing settings
    BATCH_SIZE = 32  # Batch size for data ingestion
    VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "durian.log"

    def __init__(self, **kwargs):
        """
        Initialize configuration with optional overrides.

        Args:
            **kwargs: Override any configuration parameter
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
