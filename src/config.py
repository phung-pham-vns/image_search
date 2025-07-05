class Config:
    QDRANT_URI = ""
    COLLECTION_NAME = "durian_pests"
    LLM_PROVIDER = "openai"
    LLM_MODEL_NAME = "gpt-4o-mini"
    TOP_K = 3
    DEVICE = "cpu"
    EMBEDDING_NAME = "google/siglip2-base-patch16-224"
    EMEBDDING_DIM = 768
    DATASET_DIR = (
        "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/processed_dataset"
    )
