import re
import streamlit as st
import streamlit_cropper
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
from src.core.config import Config
from streamlit_cropper import st_cropper
from ui.sidebar import EMBEDDING_MODELS
from src.core.search import ImageSearcher


def short_model_name(model_name_or_path: str) -> str:
    name = model_name_or_path.lower()
    name = re.sub(r"^.*[\\/]|openai/|google/|facebook/", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def initialize_searcher(cfg: Config) -> ImageSearcher:
    """Initializes or retrieves the ImageSearcher from session state."""
    # Use a key that includes the model name to ensure re-initialization on model change
    searcher_key = f"searcher_{st.session_state.selected_model}"

    if searcher_key not in st.session_state:
        with st.spinner(f"Loading model '{st.session_state.selected_model}'..."):
            st.session_state[searcher_key] = ImageSearcher(cfg)

    return st.session_state[searcher_key]


# def display_search_results(results: List[Dict[str, Any]], cfg: Config):
#     """Displays search results in a grid."""
#     if not results:
#         st.warning("No results found.")
#         return

#     cols = st.columns(cfg.TOP_K)
#     for i, result in enumerate(results):
#         if i >= len(cols):
#             break
#         with cols[i]:
#             try:
#                 img = Image.open(result["image_path"])
#                 st.image(
#                     img,
#                     use_container_width=True,
#                     caption=f"Score: {result['score']:.4f}",
#                 )
#                 if st.session_state.display_metadata:
#                     with st.expander("Metadata"):
#                         st.json(result.get("payload", {}))
#             except FileNotFoundError:
#                 st.error(f"Image not found at {result['image_path']}")
#             except Exception as e:
#                 st.error(f"Error displaying image: {e}")


def display_search_results(results: List[Dict[str, Any]], cfg: Config):
    """Displays search results in a grid."""
    if not results:
        st.warning("No results found.")
        return

    # cols = st.columns(cfg.TOP_K)
    cols = st.columns(2)
    for result in results:
        try:
            with cols[0]:
                img = Image.open(result["image_path"])
                st.image(
                    img,
                    use_container_width=True,
                    caption=f"Score: {result['score']:.4f}",
                )
            with cols[1]:
                if st.session_state.display_metadata:
                    with st.expander("Metadata"):
                        st.json(result.get("payload", {}))
        except FileNotFoundError:
            st.error(f"Image not found at {result['image_path']}")
        except Exception as e:
            st.error(f"Error displaying image: {e}")


def create_search_page(cfg: Config):
    """Creates the main UI for the image search page."""
    st.title("🖼️ Image Similarity Search")
    st.markdown("Upload an image to find visually similar items in the database.")

    try:
        searcher = initialize_searcher(cfg)
    except Exception as e:
        st.error(f"Failed to initialize search components: {e}")
        st.error("Please ensure the Qdrant server is running and accessible.")
        return

    col1, col2 = st.columns([0.4, 0.6])

    with col1:
        st.subheader("📤 Query Image")

        # Search category selection
        category = st.radio(
            "Select a category to search in:",
            options=["disease", "pest", "variety"],
            horizontal=True,
        )

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")

            st.subheader("✂️ Crop (Optional)")
            cropped_img = st_cropper(img, aspect_ratio=(1, 1))

            if st.button("🚀 Search", use_container_width=True, type="primary"):
                with st.spinner("Finding similar images..."):
                    try:
                        results = searcher.search(
                            query_image=cropped_img,
                            category=category,
                            top_k=cfg.TOP_K,
                        )
                        st.session_state.search_results = results
                    except Exception as e:
                        st.error(f"An error occurred during search: {e}")
                        st.session_state.search_results = []

    with col2:
        st.subheader("✨ Search Results")
        if "search_results" in st.session_state:
            display_search_results(st.session_state.search_results, cfg)
        else:
            st.info("Upload an image and click 'Search' to see results here.")
