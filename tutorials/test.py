"""
Test script for validating the image retrieval system components.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.embeddings.image_embedding import ImageEmbedding
from src.vector_stores.qdrant import QdrantVectorStore


def test_embedding_models():
    """Test different embedding models."""
    print("\n=== Testing Embedding Models ===")

    # Test images
    test_image = "figures/logo.png"  # Update path as needed
    models = [
        "google/siglip2-base-patch16-224",  # SigLIP
        "openai/clip-vit-base-patch32",  # CLIP
        "facebook/dinov2-base",  # DINOv2
        "TULIP-so400m-14-384",  # TULIP
    ]

    for model_name in models:
        try:
            print(f"\nTesting {model_name}...")
            embedder = ImageEmbedding(model_name_or_path=model_name)
            embedding = embedder.embed(test_image)
            print(f"✓ Successfully generated embedding: shape={embedding.shape}")
            print(f"✓ L2 norm: {np.linalg.norm(embedding):.4f}")
        except Exception as e:
            print(f"✗ Failed: {str(e)}")


def test_vector_store():
    """Test vector store operations."""
    print("\n=== Testing Vector Store ===")

    try:
        # Initialize components
        cfg = Config()
        store = QdrantVectorStore(uri=cfg.QDRANT_URI)

        # Test collection creation
        collection_name = "test_collection"
        embedding_size = 512
        print(f"\nCreating collection '{collection_name}'...")
        store.create_collection(
            collection_name=collection_name,
            embedding_size=embedding_size,
            distance="cosine",
        )
        print("✓ Collection created successfully")

        # Test adding embeddings
        print("\nTesting embedding insertion...")
        test_embeddings = [np.random.randn(embedding_size) for _ in range(3)]
        test_payloads = [{"id": i, "test": True} for i in range(3)]
        store.add_embeddings(
            collection_name=collection_name,
            embeddings=test_embeddings,
            payloads=test_payloads,
        )
        print("✓ Added test embeddings successfully")

        # Test querying
        print("\nTesting vector search...")
        query_vector = np.random.randn(embedding_size)
        results = store.query(
            collection_name=collection_name, query_vector=query_vector, top_k=2
        )
        print(f"✓ Retrieved {len(results)} results")

    except Exception as e:
        print(f"✗ Vector store test failed: {str(e)}")


def test_end_to_end():
    """Test complete image retrieval workflow."""
    print("\n=== Testing End-to-End Workflow ===")

    try:
        # Initialize components
        cfg = Config()
        embedder = ImageEmbedding(model_name_or_path=cfg.MODEL_NAME_OR_PATH)
        store = QdrantVectorStore(uri=cfg.QDRANT_URI)

        # Test image
        test_image = "figures/logo.png"  # Update path as needed
        collection_name = "test_e2e"

        print("\n1. Creating test collection...")
        store.create_collection(
            collection_name=collection_name, embedding_size=cfg.EMEBDDING_DIM
        )

        print("\n2. Generating embeddings...")
        embedding = embedder.embed(test_image)

        print("\n3. Adding to vector store...")
        store.add_embeddings(
            collection_name=collection_name,
            embeddings=[embedding],
            payloads=[{"path": test_image}],
        )

        print("\n4. Testing retrieval...")
        results = store.query(
            collection_name=collection_name, query_vector=embedding, top_k=1
        )

        if results and len(results) > 0:
            print("✓ Successfully retrieved the test image")
            print(f"  Score: {results[0].score:.4f}")
        else:
            print("✗ Failed to retrieve the test image")

    except Exception as e:
        print(f"✗ End-to-end test failed: {str(e)}")


if __name__ == "__main__":
    # Run all tests
    test_embedding_models()
    test_vector_store()
    test_end_to_end()
    print("\nTests completed!")
