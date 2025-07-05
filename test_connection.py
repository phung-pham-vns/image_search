#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.vector_stores.qdrant import QdrantVectorStore
from src.ingest_data import DataIngester


def test_qdrant_connection():
    """Test Qdrant connection"""
    print("Testing Qdrant connection...")
    try:
        store = QdrantVectorStore(uri="http://localhost:6333/")
        print("✅ Qdrant connection successful")
        return store
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return None


def test_data_ingester():
    """Test DataIngester initialization"""
    print("\nTesting DataIngester initialization...")
    try:
        config = Config()
        config.QDRANT_URI = "http://localhost:6333/"
        config.COLLECTION_NAME_PREFIX = "test"
        config.EMBEDDING_NAME = "google/siglip2-base-patch16-224"
        config.DEVICE = "cpu"

        ingester = DataIngester(config)
        print("✅ DataIngester initialization successful")
        print(f"   - Store: {ingester.store}")
        print(f"   - Embedder: {ingester.embedder}")
        return ingester
    except Exception as e:
        print(f"❌ DataIngester initialization failed: {e}")
        return None


def test_collection_creation():
    """Test collection creation"""
    print("\nTesting collection creation...")
    try:
        store = QdrantVectorStore(uri="http://localhost:6333/")
        store.create_collection(
            collection_name="test_collection", embedding_size=768, distance="cosine"
        )
        print("✅ Collection creation successful")
        return True
    except Exception as e:
        print(f"❌ Collection creation failed: {e}")
        return False


if __name__ == "__main__":
    print("🔍 Testing Image Retrieval System Components\n")

    # Test 1: Qdrant connection
    store = test_qdrant_connection()

    # Test 2: Collection creation
    if store:
        test_collection_creation()

    # Test 3: DataIngester initialization
    ingester = test_data_ingester()

    print("\n🏁 Testing completed!")
