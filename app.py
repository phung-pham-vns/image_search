import streamlit as st
import torch

from src.core.config import Config
from ui.sidebar import create_sidebar_settings, create_navigation, EMBEDDING_MODELS
from ui.data_ingestion import create_data_ingestion_page
from ui.reports import create_reports_page
from ui.search import create_search_page
from ui.ask_with_image import create_ask_with_image_page


def initialize_session_state():
    """Initialize session state with default values."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Image Search"

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "SigLIP2 Base"

    if "qdrant_uri" not in st.session_state:
        st.session_state.qdrant_uri = "http://localhost:6333"

    if "collection_name_prefix" not in st.session_state:
        st.session_state.collection_name_prefix = "durian"

    if "device" not in st.session_state:
        st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"

    if "top_k" not in st.session_state:
        st.session_state.top_k = 5

    if "display_metadata" not in st.session_state:
        st.session_state.display_metadata = True

    if "ingestion_status" not in st.session_state:
        st.session_state.ingestion_status = {}


def get_current_config() -> Config:
    """Get the current configuration from session state."""
    model_path = EMBEDDING_MODELS[st.session_state.selected_model]["model_path"]

    return Config(
        QDRANT_URI=st.session_state.qdrant_uri,
        COLLECTION_NAME_PREFIX=st.session_state.collection_name_prefix,
        MODEL_NAME_OR_PATH=model_path,
        DEVICE=st.session_state.device,
        TOP_K=st.session_state.top_k,
    )


def main():
    """Main function to run the Streamlit app."""
    # Page configuration
    st.set_page_config(
        page_title="Image Retrieval System",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()
    cfg = get_current_config()

    # Create sidebar and handle navigation
    create_sidebar_settings()
    create_navigation()

    # Main content area
    if st.session_state.current_page == "Image Search":
        create_search_page(cfg)
    elif st.session_state.current_page == "Ask with Image":
        create_ask_with_image_page(cfg)
    elif st.session_state.current_page == "Data Ingestion":
        create_data_ingestion_page(cfg)
    elif st.session_state.current_page == "Reports":
        create_reports_page(cfg)
    else:
        st.error(f"Unknown page: {st.session_state.current_page}")

    # Footer
    st.markdown("---")
    st.markdown(
        "🖼️ **Image Retrieval System** - Powered by Streamlit, Qdrant, and modern AI models."
    )


if __name__ == "__main__":
    main()
