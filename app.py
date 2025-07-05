import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import streamlit as st
import streamlit_cropper
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_cropper import st_cropper, _recommended_box

from src.config import Config
from src.vector_stores.qdrant import QdrantVectorStore
from src.embeddings.image_embedding import ImageEmbedding

# Page configuration
st.set_page_config(
    page_title="Durian Image Retrieval System",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Available embedding models
EMBEDDING_MODELS = {
    "SigLIP2 Base": "google/siglip2-base-patch16-224",
    "SigLIP2 Large": "google/siglip2-large-patch16-224",
    "CLIP ViT-B/32": "openai/clip-vit-base-patch32",
    "CLIP ViT-L/14": "openai/clip-vit-large-patch14",
    "DINOv2 ViT-B/14": "facebook/dinov2-base",
    "DINOv2 ViT-L/14": "facebook/dinov2-large",
}

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Disease Report"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "SigLIP2 Base"
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "store" not in st.session_state:
    st.session_state.store = None

# Initialize configuration
cfg = Config()


def initialize_components():
    """Initialize embedding model and vector store"""
    if (
        st.session_state.embedder is None
        or st.session_state.selected_model != st.session_state.get("last_model")
    ):
        try:
            with st.spinner(f"Loading {st.session_state.selected_model}..."):
                model_path = EMBEDDING_MODELS[st.session_state.selected_model]
                st.session_state.embedder = ImageEmbedding(
                    model_name_or_path=model_path, device=cfg.DEVICE
                )
                st.session_state.last_model = st.session_state.selected_model
                st.session_state.store = QdrantVectorStore(uri=cfg.QDRANT_URI)
        except Exception as e:
            st.error(f"Failed to initialize model: {e}")
            return False
    return True


def count_files_in_directories(base_path: str) -> Dict[str, int]:
    """Count files in each subdirectory"""
    counts = {}
    base_dir = Path(base_path)

    if not base_dir.exists():
        return counts

    for item in base_dir.iterdir():
        if item.is_dir():
            # Count all image files in the directory
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            file_count = sum(
                1
                for f in item.rglob("*")
                if f.is_file() and f.suffix.lower() in image_extensions
            )
            counts[item.name] = file_count

    return counts


def create_disease_report_page():
    """Page 1: Disease Report with file counts"""
    st.title("🥭 Durian Disease Report")
    st.caption("📊 Disease Statistics")

    # File counting section
    diseases_path = "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/original_dataset/diseases"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 Disease Categories File Count")

        if st.button("🔄 Refresh File Counts", use_container_width=True):
            st.session_state.disease_file_counts = count_files_in_directories(
                diseases_path
            )

        if "disease_file_counts" not in st.session_state:
            st.session_state.disease_file_counts = count_files_in_directories(
                diseases_path
            )

        if st.session_state.disease_file_counts:
            # Create DataFrame for better display
            df = pd.DataFrame(
                [
                    {"Category": name, "File Count": count}
                    for name, count in st.session_state.disease_file_counts.items()
                ]
            )

            # Sort by file count descending
            df = df.sort_values("File Count", ascending=False)

            # Display as a beautiful table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Category": st.column_config.TextColumn(
                        "Disease Category", width="medium"
                    ),
                    "File Count": st.column_config.NumberColumn(
                        "Number of Images", width="small"
                    ),
                },
            )

            # Create visualization
            fig = px.bar(
                df,
                x="File Count",
                y="Category",
                orientation="h",
                title="Distribution of Images Across Disease Categories",
                color="File Count",
                color_continuous_scale="viridis",
            )
            fig.update_layout(
                height=600,
                showlegend=False,
                xaxis_title="Number of Images",
                yaxis_title="Disease Category",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No disease categories found or directory doesn't exist.")

    with col2:
        st.subheader("📈 Summary Statistics")

        if st.session_state.disease_file_counts:
            total_files = sum(st.session_state.disease_file_counts.values())
            total_categories = len(st.session_state.disease_file_counts)
            avg_files = total_files / total_categories if total_categories > 0 else 0

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📊 Total Images", f"{total_files:,}")
            with col_b:
                st.metric("📁 Categories", f"{total_categories}")

            st.metric("📏 Average per Category", f"{avg_files:.1f}")

            # Top categories
            if st.session_state.disease_file_counts:
                top_categories = sorted(
                    st.session_state.disease_file_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]

                st.subheader("🏆 Top 5 Categories")
                for i, (category, count) in enumerate(top_categories, 1):
                    st.metric(f"{i}. {category}", f"{count} images")


def create_pest_report_page():
    """Page 2: Pest Report with file counts"""
    st.title("🥭 Durian Pest Report")
    st.caption("📊 Pest Statistics")

    # File counting section
    pests_path = "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/original_dataset/pests"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 Pest Categories File Count")

        if st.button("🔄 Refresh File Counts", use_container_width=True):
            st.session_state.pest_file_counts = count_files_in_directories(pests_path)

        if "pest_file_counts" not in st.session_state:
            st.session_state.pest_file_counts = count_files_in_directories(pests_path)

        if st.session_state.pest_file_counts:
            # Create DataFrame for better display
            df = pd.DataFrame(
                [
                    {"Category": name, "File Count": count}
                    for name, count in st.session_state.pest_file_counts.items()
                ]
            )

            # Sort by file count descending
            df = df.sort_values("File Count", ascending=False)

            # Display as a beautiful table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Category": st.column_config.TextColumn(
                        "Pest Category", width="medium"
                    ),
                    "File Count": st.column_config.NumberColumn(
                        "Number of Images", width="small"
                    ),
                },
            )

            # Create visualization
            fig = px.bar(
                df,
                x="File Count",
                y="Category",
                orientation="h",
                title="Distribution of Images Across Pest Categories",
                color="File Count",
                color_continuous_scale="viridis",
            )
            fig.update_layout(
                height=600,
                showlegend=False,
                xaxis_title="Number of Images",
                yaxis_title="Pest Category",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No pest categories found or directory doesn't exist.")

    with col2:
        st.subheader("📈 Summary Statistics")

        if st.session_state.pest_file_counts:
            total_files = sum(st.session_state.pest_file_counts.values())
            total_categories = len(st.session_state.pest_file_counts)
            avg_files = total_files / total_categories if total_categories > 0 else 0

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📊 Total Images", f"{total_files:,}")
            with col_b:
                st.metric("📁 Categories", f"{total_categories}")

            st.metric("📏 Average per Category", f"{avg_files:.1f}")

            # Top categories
            if st.session_state.pest_file_counts:
                top_categories = sorted(
                    st.session_state.pest_file_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]

                st.subheader("🏆 Top 5 Categories")
                for i, (category, count) in enumerate(top_categories, 1):
                    st.metric(f"{i}. {category}", f"{count} images")


def create_search_page():
    """Page 3: Image Search with enhanced UI"""
    st.title("🔍 Durian Image Similarity Search")
    st.caption("🔍 Image Search")

    # Model selection in sidebar
    with st.sidebar:
        st.subheader("🤖 Model Configuration")

        selected_model = st.selectbox(
            "Select Embedding Model",
            options=list(EMBEDDING_MODELS.keys()),
            index=list(EMBEDDING_MODELS.keys()).index(st.session_state.selected_model),
            help="Choose the embedding model for image similarity search",
        )

        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.embedder = None  # Reset embedder to reload with new model
            st.rerun()

        # Search parameters
        st.subheader("⚙️ Search Parameters")

        search_category = st.radio(
            "Search Category",
            options=["disease", "pest"],
            index=0 if st.session_state.get("category", "disease") == "disease" else 1,
            help="Choose whether to search for diseases or pests",
        )
        st.session_state.category = search_category

        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=st.session_state.get("top_k", 5),
            help="Number of similar images to return",
        )
        st.session_state.top_k = top_k

        show_metadata = st.toggle(
            "Show Metadata",
            value=st.session_state.get("display_metadata", True),
            help="Display disease/pest labels and similarity scores",
        )
        st.session_state.display_metadata = show_metadata

    # Initialize components
    if not initialize_components():
        st.error(
            "Failed to initialize the embedding model. Please check your configuration."
        )
        return

    # Main search interface
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📤 Upload Query Image")

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image to find similar images in the dataset",
        )

        if uploaded_file:
            # Save uploaded file temporarily
            with open("figures/temp.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Preprocess and resize for cropping
            uploaded_img = st.session_state.embedder.preprocess_image(uploaded_file)
            w, h = uploaded_img.size
            resized_img = uploaded_img.resize((370, int(370 / w * h)))

            st.subheader("✂️ Crop Image (Optional)")
            st.caption("Edit the box to change the Region of Interest (ROI)")

            # Override default recommended crop box
            def custom_crop_box(img: Image.Image, aspect_ratio: tuple) -> dict:
                width, height = img.size
                return {
                    "left": 0,
                    "top": 0,
                    "width": width - 2,
                    "height": height - 2,
                }

            streamlit_cropper._recommended_box = custom_crop_box

            cropped_img = st_cropper(
                resized_img,
                box_color="#FF0000",
                realtime_update=True,
                aspect_ratio=(16, 9),
            )

            # Search button
            if st.button(
                "🔍 Search Similar Images", use_container_width=True, type="primary"
            ):
                st.session_state.search_results = perform_search(cropped_img)
                st.rerun()

    with col2:
        st.subheader("📋 Search Results")

        if "search_results" in st.session_state and st.session_state.search_results:
            display_search_results(st.session_state.search_results)
        else:
            st.info(
                "👆 Upload an image and click 'Search Similar Images' to see results"
            )


def perform_search(query_image: Image.Image) -> List[Dict]:
    """Perform similarity search"""
    try:
        # Get embedding
        query_embedding = st.session_state.embedder.embed(query_image)

        if query_embedding is None:
            st.error("Failed to generate embedding for the query image")
            return []

        # Configure collection based on category
        if st.session_state.category == "disease":
            image_dir = Path(Config.DATASET_DIR) / "diseases" / "images"
            collection_name = "durian_diseases"
        else:
            image_dir = Path(Config.DATASET_DIR) / "pests" / "images"
            collection_name = "durian_pests"

        # Query vector store
        results = st.session_state.store.query(
            collection_name=collection_name,
            query_vector=query_embedding,
            top_k=st.session_state.top_k,
        )

        # Process results
        processed_results = []
        for hit in results:
            img_name = hit.payload["image_name"]
            label = hit.payload[st.session_state.category]
            score = hit.score

            # Load result image
            result_img_path = image_dir / img_name
            if result_img_path.exists():
                result_img = Image.open(result_img_path)
                processed_results.append(
                    {
                        "image": result_img,
                        "label": label,
                        "score": score,
                        "filename": img_name,
                    }
                )

        return processed_results

    except Exception as e:
        st.error(f"Search failed: {e}")
        return []


def display_search_results(results: List[Dict]):
    """Display search results in a simple grid"""
    if not results:
        st.warning("No results found")
        return

    # Create grid layout
    cols = st.columns(min(3, len(results)))

    for i, result in enumerate(results):
        col_idx = i % 3

        with cols[col_idx]:
            # Display image
            st.image(
                result["image"], use_container_width=True, caption=result["filename"]
            )

            if st.session_state.display_metadata:
                # Display metadata
                st.markdown(
                    f"**{st.session_state.category.capitalize()}:** {result['label']}"
                )

                # Create a progress bar for similarity score
                score_percentage = result["score"] * 100
                st.progress(
                    result["score"], text=f"Similarity: {score_percentage:.1f}%"
                )

                st.caption(f"Score: `{result['score']:.3f}`")


# Navigation
st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Navigation")

page_options = ["Disease Report", "Pest Report", "Image Search"]
current_page = st.sidebar.selectbox(
    "Select Page",
    options=page_options,
    index=page_options.index(st.session_state.current_page),
)

if current_page != st.session_state.current_page:
    st.session_state.current_page = current_page
    st.rerun()

# Display current page
if st.session_state.current_page == "Disease Report":
    create_disease_report_page()
elif st.session_state.current_page == "Pest Report":
    create_pest_report_page()
elif st.session_state.current_page == "Image Search":
    create_search_page()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🥭 Durian Image Retrieval System | Built with Streamlit & AI
    </div>
    """,
    unsafe_allow_html=True,
)
