import streamlit as st
from src.core.config import Config

# Available embedding models with detailed specifications
EMBEDDING_MODELS = {
    "SigLIP2 Base": {
        "model_path": "google/siglip2-base-patch16-224",
        "description": "Google's SigLIP2 Base model for image-text understanding",
        "embedding_size": 768,
        "model_size": "~1.2GB",
        "purpose": "General-purpose image embedding with strong zero-shot capabilities",
        "architecture": "Vision Transformer (ViT-B/16)",
    },
    "SigLIP2 Large": {
        "model_path": "google/siglip2-large-patch16-224",
        "description": "Google's SigLIP2 Large model for advanced image understanding",
        "embedding_size": 1024,
        "model_size": "~2.5GB",
        "purpose": "High-performance image embedding with superior accuracy",
        "architecture": "Vision Transformer (ViT-L/16)",
    },
    "CLIP ViT-B/32": {
        "model_path": "openai/clip-vit-base-patch32",
        "description": "OpenAI's CLIP model with ViT-B/32 architecture",
        "embedding_size": 512,
        "model_size": "~150MB",
        "purpose": "Efficient image-text understanding and similarity",
        "architecture": "Vision Transformer (ViT-B/32)",
    },
    "CLIP ViT-L/14": {
        "model_path": "openai/clip-vit-large-patch14",
        "description": "OpenAI's CLIP model with ViT-L/14 architecture",
        "embedding_size": 768,
        "model_size": "~1.7GB",
        "purpose": "High-quality image understanding and retrieval",
        "architecture": "Vision Transformer (ViT-L/14)",
    },
    "DINOv2 ViT-B/14": {
        "model_path": "facebook/dinov2-base",
        "description": "Facebook's DINOv2 self-supervised vision model",
        "embedding_size": 768,
        "model_size": "~1.1GB",
        "purpose": "Self-supervised image representation learning",
        "architecture": "Vision Transformer (ViT-B/14)",
    },
    "DINOv2 ViT-L/14": {
        "model_path": "facebook/dinov2-large",
        "description": "Facebook's DINOv2 large self-supervised vision model",
        "embedding_size": 1024,
        "model_size": "~2.4GB",
        "purpose": "High-capacity self-supervised image understanding",
        "architecture": "Vision Transformer (ViT-L/14)",
    },
    "TULIP": {
        "model_path": "TULIP-so400m-14-384",
        "description": "TULIP model for specialized embeddings",
        "embedding_size": 512,
        "model_size": "N/A",
        "purpose": "Specialized peptide-protein interaction embeddings",
        "architecture": "Transformer-based",
    },
}


def check_database_status():
    """Check and display database status"""
    try:
        from src.core.vector_store import QdrantVectorStore

        store = QdrantVectorStore(uri=st.session_state.qdrant_uri)
        collections = store.client.get_collections()

        st.session_state.db_status = {}
        for collection in collections.collections:
            collection_info = store.client.get_collection(collection.name)
            st.session_state.db_status[collection.name] = {
                "points": collection_info.points_count,
                "vectors": collection_info.vectors_count,
                "status": "active" if collection_info.status == "green" else "inactive",
            }
    except Exception as e:
        st.error(f"Failed to check database status: {e}")


def create_sidebar_settings():
    """Create comprehensive settings in the sidebar"""
    st.sidebar.title("⚙️ Settings")

    # Vector Store Configuration
    st.sidebar.subheader("🗄️ Vector Store")
    qdrant_uri = st.sidebar.text_input(
        "Qdrant URI",
        value=st.session_state.qdrant_uri,
        help="URL of the Qdrant vector database server",
    )
    st.session_state.qdrant_uri = qdrant_uri

    collection_prefix = st.sidebar.text_input(
        "Collection Name Prefix",
        value=st.session_state.collection_name_prefix,
        help="Prefix for collection names (e.g., 'image_retrieval')",
    )
    st.session_state.collection_name_prefix = collection_prefix

    # Embedding Model Selection
    st.sidebar.subheader("🤖 Embedding Model")
    selected_model_name = st.sidebar.selectbox(
        "Choose Model",
        options=list(EMBEDDING_MODELS.keys()),
        index=list(EMBEDDING_MODELS.keys()).index(st.session_state.selected_model),
        help="Select the embedding model for image processing",
    )

    if selected_model_name != st.session_state.selected_model:
        st.session_state.selected_model = selected_model_name
        # Trigger a rerun to reload the model in the main app logic
        st.rerun()

    # Model Specifications
    if selected_model_name in EMBEDDING_MODELS:
        model_specs = EMBEDDING_MODELS[selected_model_name]

        with st.sidebar.expander("📋 Model Specifications", expanded=False):
            st.markdown(f"**Description:** {model_specs.get('description', 'N/A')}")
            st.markdown(f"**Purpose:** {model_specs.get('purpose', 'N/A')}")
            st.markdown(f"**Architecture:** {model_specs.get('architecture', 'N/A')}")
            st.markdown(
                f"**Embedding Size:** {model_specs.get('embedding_size', 'N/A')} dimensions"
            )
            st.markdown(f"**Model Size:** {model_specs.get('model_size', 'N/A')}")

    # Device Configuration
    st.sidebar.subheader("💻 Hardware")
    device = st.sidebar.selectbox(
        "Device",
        options=["cpu", "cuda", "mps"],
        index=["cpu", "cuda", "mps"].index(st.session_state.device),
        help="Device to run the embedding model on",
    )
    st.session_state.device = device

    # Search Configuration
    st.sidebar.subheader("🔍 Search Settings")
    top_k = st.sidebar.slider(
        "Top K Results",
        min_value=1,
        max_value=20,
        value=st.session_state.top_k,
        help="Number of similar images to return",
    )
    st.session_state.top_k = top_k

    show_metadata = st.sidebar.toggle(
        "Show Metadata",
        value=st.session_state.display_metadata,
        help="Display labels and similarity scores",
    )
    st.session_state.display_metadata = show_metadata

    # Database Status
    st.sidebar.subheader("📊 Database Status")
    if st.sidebar.button("🔄 Check Database Status", use_container_width=True):
        check_database_status()

    if "db_status" in st.session_state:
        for collection, info in st.session_state.db_status.items():
            with st.sidebar.expander(f"📁 {collection}", expanded=False):
                st.markdown(f"**Points:** {info.get('points', 'N/A')}")
                st.markdown(f"**Vectors:** {info.get('vectors', 'N/A')}")
                st.markdown(f"**Status:** {info.get('status', 'N/A')}")


def create_navigation():
    """Create navigation sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")

    page_options = [
        "Image Search",
        "Ask with Image",
        "Data Ingestion",
        "Data Investigation",
    ]
    current_page = st.sidebar.selectbox(
        "Select Page",
        options=page_options,
        index=page_options.index(st.session_state.current_page),
    )

    if current_page != st.session_state.current_page:
        st.session_state.current_page = current_page
        st.rerun()
