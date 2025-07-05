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

# Custom CSS for beautiful UI
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .upload-section {
        background: rgba(102, 126, 234, 0.1);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    
    .model-selector {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .page-indicator {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
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
    st.session_state.current_page = "Dataset Overview"
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


def create_dataset_overview_page():
    """Page 1: Dataset Overview with file counts"""
    st.markdown(
        '<div class="main-header"><h1>🥭 Durian Dataset Overview</h1></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-indicator">📊 Page 1: Dataset Statistics</div>',
        unsafe_allow_html=True,
    )

    # File counting section
    diseases_path = "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/original_dataset/diseases"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 Disease Categories File Count")

        if st.button("🔄 Refresh File Counts", use_container_width=True):
            st.session_state.file_counts = count_files_in_directories(diseases_path)

        if "file_counts" not in st.session_state:
            st.session_state.file_counts = count_files_in_directories(diseases_path)

        if st.session_state.file_counts:
            # Create DataFrame for better display
            df = pd.DataFrame(
                [
                    {"Category": name, "File Count": count}
                    for name, count in st.session_state.file_counts.items()
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

        if st.session_state.file_counts:
            total_files = sum(st.session_state.file_counts.values())
            total_categories = len(st.session_state.file_counts)
            avg_files = total_files / total_categories if total_categories > 0 else 0

            st.markdown(
                f"""
            <div class="metric-card">
                <h3>📊 Total Images</h3>
                <h2>{total_files:,}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="metric-card">
                <h3>📁 Categories</h3>
                <h2>{total_categories}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="metric-card">
                <h3>📏 Average per Category</h3>
                <h2>{avg_files:.1f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Top categories
            if st.session_state.file_counts:
                top_categories = sorted(
                    st.session_state.file_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]

                st.subheader("🏆 Top 5 Categories")
                for i, (category, count) in enumerate(top_categories, 1):
                    st.markdown(f"**{i}.** {category}: **{count}** images")


def create_search_page():
    """Page 2: Image Search with enhanced UI"""
    st.markdown(
        '<div class="main-header"><h1>🔍 Durian Image Similarity Search</h1></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-indicator">🔍 Page 2: Image Search</div>',
        unsafe_allow_html=True,
    )

    # Model selection in sidebar
    with st.sidebar:
        st.subheader("🤖 Model Configuration")

        st.markdown('<div class="model-selector">', unsafe_allow_html=True)
        selected_model = st.selectbox(
            "Select Embedding Model",
            options=list(EMBEDDING_MODELS.keys()),
            index=list(EMBEDDING_MODELS.keys()).index(st.session_state.selected_model),
            help="Choose the embedding model for image similarity search",
        )
        st.markdown("</div>", unsafe_allow_html=True)

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

        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image to find similar images in the dataset",
        )
        st.markdown("</div>", unsafe_allow_html=True)

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
                box_color="#667eea",
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
    """Display search results in a beautiful grid"""
    if not results:
        st.warning("No results found")
        return

    # Create grid layout
    cols = st.columns(min(3, len(results)))

    for i, result in enumerate(results):
        col_idx = i % 3

        with cols[col_idx]:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)

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

            st.markdown("</div>", unsafe_allow_html=True)


# Navigation
st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Navigation")

page_options = ["Dataset Overview", "Image Search"]
current_page = st.sidebar.selectbox(
    "Select Page",
    options=page_options,
    index=page_options.index(st.session_state.current_page),
)

if current_page != st.session_state.current_page:
    st.session_state.current_page = current_page
    st.rerun()

# Display current page
if st.session_state.current_page == "Dataset Overview":
    create_dataset_overview_page()
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
