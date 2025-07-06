import streamlit as st
from src.config import Config

# Available embedding models with detailed specifications
EMBEDDING_MODELS = {
    "SigLIP2 Base": {
        "model_path": "google/siglip2-base-patch16-224",
        "description": "Google's SigLIP2 Base model for image-text understanding",
        "embedding_size": 768,
        "model_size": "~1.2GB",
        "purpose": "General-purpose image embedding with strong zero-shot capabilities",
        "architecture": "Vision Transformer (ViT-B/16)",
        "training_data": "400M image-text pairs",
        "performance": "Excellent for image similarity and retrieval tasks",
    },
    "SigLIP2 Large": {
        "model_path": "google/siglip2-large-patch16-224",
        "description": "Google's SigLIP2 Large model for advanced image understanding",
        "embedding_size": 1024,
        "model_size": "~2.5GB",
        "purpose": "High-performance image embedding with superior accuracy",
        "architecture": "Vision Transformer (ViT-L/16)",
        "training_data": "400M image-text pairs",
        "performance": "Best performance for complex image analysis tasks",
    },
    "CLIP ViT-B/32": {
        "model_path": "openai/clip-vit-base-patch32",
        "description": "OpenAI's CLIP model with ViT-B/32 architecture",
        "embedding_size": 512,
        "model_size": "~150MB",
        "purpose": "Efficient image-text understanding and similarity",
        "architecture": "Vision Transformer (ViT-B/32)",
        "training_data": "400M image-text pairs",
        "performance": "Good balance of speed and accuracy",
    },
    "CLIP ViT-L/14": {
        "model_path": "openai/clip-vit-large-patch14",
        "description": "OpenAI's CLIP model with ViT-L/14 architecture",
        "embedding_size": 768,
        "model_size": "~1.7GB",
        "purpose": "High-quality image understanding and retrieval",
        "architecture": "Vision Transformer (ViT-L/14)",
        "training_data": "400M image-text pairs",
        "performance": "Excellent accuracy for image similarity tasks",
    },
    "DINOv2 ViT-B/14": {
        "model_path": "facebook/dinov2-base",
        "description": "Facebook's DINOv2 self-supervised vision model",
        "embedding_size": 768,
        "model_size": "~1.1GB",
        "purpose": "Self-supervised image representation learning",
        "architecture": "Vision Transformer (ViT-B/14)",
        "training_data": "142M images",
        "performance": "Strong performance on visual tasks without labels",
    },
    "DINOv2 ViT-L/14": {
        "model_path": "facebook/dinov2-large",
        "description": "Facebook's DINOv2 large self-supervised vision model",
        "embedding_size": 1024,
        "model_size": "~2.4GB",
        "purpose": "High-capacity self-supervised image understanding",
        "architecture": "Vision Transformer (ViT-L/14)",
        "training_data": "142M images",
        "performance": "Best performance for self-supervised visual tasks",
    },
}


def check_database_status():
    """Check and display database status"""
    try:
        from src.vector_stores.qdrant import QdrantVectorStore

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
        help="Prefix for collection names (e.g., 'durian' creates 'durian_disease', 'durian_pest')",
    )
    st.session_state.collection_name_prefix = collection_prefix

    # Embedding Model Selection
    st.sidebar.subheader("🤖 Embedding Model")
    selected_model = st.sidebar.selectbox(
        "Choose Model",
        options=list(EMBEDDING_MODELS.keys()),
        index=list(EMBEDDING_MODELS.keys()).index(st.session_state.selected_model),
        help="Select the embedding model for image processing",
    )

    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.embedder = None  # Reset embedder to reload model

    # Model Specifications
    if selected_model in EMBEDDING_MODELS:
        model_specs = EMBEDDING_MODELS[selected_model]

        with st.sidebar.expander("📋 Model Specifications", expanded=False):
            st.markdown(f"**Description:** {model_specs['description']}")
            st.markdown(f"**Purpose:** {model_specs['purpose']}")
            st.markdown(f"**Architecture:** {model_specs['architecture']}")
            st.markdown(
                f"**Embedding Size:** {model_specs['embedding_size']} dimensions"
            )
            st.markdown(f"**Model Size:** {model_specs['model_size']}")
            st.markdown(f"**Training Data:** {model_specs['training_data']}")
            st.markdown(f"**Performance:** {model_specs['performance']}")

    # Device Configuration
    st.sidebar.subheader("💻 Hardware")
    device = st.sidebar.selectbox(
        "Device",
        options=["cpu", "cuda", "mps"],
        index=0,
        help="Device to run the embedding model on",
    )
    cfg = Config()
    cfg.DEVICE = device

    # Search Configuration
    st.sidebar.subheader("🔍 Search Settings")
    top_k = st.sidebar.slider(
        "Top K Results",
        min_value=1,
        max_value=20,
        value=st.session_state.get("top_k", 5),
        help="Number of similar images to return",
    )
    st.session_state.top_k = top_k

    show_metadata = st.sidebar.toggle(
        "Show Metadata",
        value=st.session_state.get("display_metadata", True),
        help="Display disease/pest labels and similarity scores",
    )
    st.session_state.display_metadata = show_metadata

    # Database Status
    st.sidebar.subheader("📊 Database Status")
    if st.sidebar.button("🔄 Check Database Status", use_container_width=True):
        check_database_status()

    if "db_status" in st.session_state:
        for collection, info in st.session_state.db_status.items():
            with st.sidebar.expander(f"📁 {collection}", expanded=False):
                st.markdown(f"**Points:** {info['points']}")
                st.markdown(f"**Vectors:** {info['vectors']}")
                st.markdown(f"**Status:** {info['status']}")


def create_navigation():
    """Create navigation sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")

    page_options = [
        "Data Ingestion",
        "Disease Report",
        "Pest Report",
        "Image Search",
        "Ask with Image",
    ]
    current_page = st.sidebar.selectbox(
        "Select Page",
        options=page_options,
        index=page_options.index(st.session_state.current_page),
    )

    if current_page != st.session_state.current_page:
        st.session_state.current_page = current_page
        st.rerun()

    # Add search parameters to sidebar for Image Search page
    if st.session_state.current_page == "Image Search":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Search Parameters")
        # Category
        search_category = st.sidebar.radio(
            "Search Category",
            options=["disease", "pest"],
            index=0 if st.session_state.get("category", "disease") == "disease" else 1,
            help="Choose whether to search for disease or pest",
            key="sidebar_search_category",
        )
        st.session_state.category = search_category
        # Top K
        top_k = st.sidebar.slider(
            "Top K Results",
            min_value=1,
            max_value=20,
            value=st.session_state.get("top_k", 5),
            help="Number of similar images to return",
            key="sidebar_top_k",
        )
        st.session_state.top_k = top_k
        # Distance metric
        distance_metric = st.sidebar.selectbox(
            "Distance Metric",
            options=["cosine", "euclid", "dot", "manhattan"],
            index=0,
            help="Distance metric for vector similarity",
            key="sidebar_distance_metric",
        )
        st.session_state.distance_metric = distance_metric

    return current_page
