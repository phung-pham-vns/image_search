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
from src.ingest_data import DataIngester

# Page configuration
st.set_page_config(
    page_title="Durian Image Retrieval System",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Data Ingestion"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "SigLIP2 Base"
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "store" not in st.session_state:
    st.session_state.store = None
if "collection_name_prefix" not in st.session_state:
    st.session_state.collection_name_prefix = "durian"
if "qdrant_uri" not in st.session_state:
    st.session_state.qdrant_uri = "http://localhost:6333/"
if "ingestion_status" not in st.session_state:
    st.session_state.ingestion_status = {}

# Initialize configuration
cfg = Config()


# Sidebar settings
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


def check_database_status():
    """Check the status of Qdrant collections"""
    try:
        store = QdrantVectorStore(uri=st.session_state.qdrant_uri)
        collections = store.client.get_collections()

        db_status = {}
        for collection in collections.collections:
            collection_name = collection.name
            if collection_name.startswith(st.session_state.collection_name_prefix):
                info = store.client.get_collection(collection_name=collection_name)
                db_status[collection_name] = {
                    "points": info.points_count,
                    "vectors": info.vectors_count,
                    "status": info.status.value,
                }

        st.session_state.db_status = db_status
        st.sidebar.success("Database status updated!")

    except Exception as e:
        st.sidebar.error(f"Failed to check database status: {e}")


def initialize_components():
    """Initialize embedding model and vector store"""
    if (
        st.session_state.embedder is None
        or st.session_state.selected_model != st.session_state.get("last_model")
    ):
        try:
            with st.spinner(f"Loading {st.session_state.selected_model}..."):
                model_path = EMBEDDING_MODELS[st.session_state.selected_model][
                    "model_path"
                ]
                st.session_state.embedder = ImageEmbedding(
                    model_name_or_path=model_path, device=cfg.DEVICE
                )
                st.session_state.last_model = st.session_state.selected_model
                st.session_state.store = QdrantVectorStore(
                    uri=st.session_state.qdrant_uri
                )
        except Exception as e:
            st.error(f"Failed to initialize model: {e}")
            return False
    return True


def create_data_ingestion_page():
    """Page for data ingestion with comprehensive settings and monitoring"""
    st.title("📥 Data Ingestion")
    st.caption("Ingest image data into the vector database for similarity search")

    # Configuration section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("⚙️ Ingestion Configuration")

        # Dataset paths
        dataset_path = st.text_input(
            "Dataset Directory",
            value=cfg.DATASET_DIR,
            help="Path to the processed dataset directory",
        )

        # Categories to ingest
        categories = st.multiselect(
            "Categories to Ingest",
            options=["disease", "pest"],
            default=["disease", "pest"],
            help="Select which categories to ingest into the vector database",
        )

        # Batch size
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=64,
            value=32,
            help="Number of images to process in each batch",
        )

        # Distance metric
        distance_metric = st.selectbox(
            "Distance Metric",
            options=["cosine", "euclid", "dot", "manhattan"],
            index=0,
            help="Distance metric for vector similarity",
        )

        # Ingestion button
        if st.button("🚀 Start Ingestion", type="primary", use_container_width=True):
            if not categories:
                st.error("Please select at least one category to ingest.")
                return

            if not dataset_path or not Path(dataset_path).exists():
                st.error("Please provide a valid dataset directory path.")
                return

            # Update config with current settings
            cfg.DATASET_DIR = dataset_path
            cfg.QDRANT_URI = st.session_state.qdrant_uri
            cfg.EMBEDDING_NAME = EMBEDDING_MODELS[st.session_state.selected_model][
                "model_path"
            ]

            # Start ingestion
            perform_data_ingestion(categories, batch_size, distance_metric)

    with col2:
        st.subheader("📊 Current Status")

        # Show database URL
        st.markdown("**🔗 Database URL:**")
        st.code(st.session_state.qdrant_uri, language="text")

        if st.button("🌐 Open Qdrant Dashboard", use_container_width=True):
            st.markdown(
                f"[Open Qdrant Dashboard]({st.session_state.qdrant_uri}dashboard#/collections/)"
            )

        # Show ingestion status
        if st.session_state.ingestion_status:
            st.subheader("📈 Ingestion Progress")
            for category, status in st.session_state.ingestion_status.items():
                with st.expander(f"📁 {category}", expanded=True):
                    if status.get("status") == "completed":
                        st.success("✅ Completed")
                        st.metric("Images Processed", status.get("processed", 0))
                        st.metric("Vectors Stored", status.get("vectors", 0))
                    elif status.get("status") == "failed":
                        st.error("❌ Failed")
                        st.text(status.get("error", "Unknown error"))
                    else:
                        st.info("⏳ In Progress...")
                        if "progress" in status:
                            st.progress(status["progress"])

    # Results section
    if st.session_state.ingestion_status:
        st.subheader("📋 Ingestion Results")

        # Create results table
        results_data = []
        for category, status in st.session_state.ingestion_status.items():
            if status.get("status") == "completed":
                results_data.append(
                    {
                        "Category": category,
                        "Images Processed": status.get("processed", 0),
                        "Vectors Stored": status.get("vectors", 0),
                        "Collection Name": f"{st.session_state.collection_name_prefix}_{category}",
                        "Status": "✅ Completed",
                    }
                )

        if results_data:
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Show collection URLs
            st.subheader("🔗 Collection URLs")
            for _, row in df.iterrows():
                collection_url = (
                    f"{st.session_state.qdrant_uri}collections/{row['Collection Name']}"
                )
                st.markdown(
                    f"**{row['Category']}:** [{collection_url}]({collection_url})"
                )


def perform_data_ingestion(categories, batch_size, distance_metric):
    """Perform the actual data ingestion"""
    try:
        # Create a new config with updated values
        updated_cfg = Config()
        updated_cfg.QDRANT_URI = st.session_state.qdrant_uri
        updated_cfg.COLLECTION_NAME_PREFIX = st.session_state.collection_name_prefix
        updated_cfg.EMBEDDING_NAME = EMBEDDING_MODELS[st.session_state.selected_model][
            "model_path"
        ]
        updated_cfg.DEVICE = cfg.DEVICE
        updated_cfg.DATASET_DIR = cfg.DATASET_DIR
        updated_cfg.PROCESSED_DATASET_DIR = cfg.PROCESSED_DATASET_DIR

        # Initialize ingester with updated config
        ingester = DataIngester(updated_cfg)

        # Update ingestion status
        for category in categories:
            st.session_state.ingestion_status[category] = {
                "status": "in_progress",
                "progress": 0,
            }

        # Perform ingestion for each category
        for category in categories:
            try:
                with st.spinner(f"Ingesting {category}..."):
                    # Create collection
                    collection_name = (
                        f"{st.session_state.collection_name_prefix}_{category}"
                    )
                    embedding_size = EMBEDDING_MODELS[st.session_state.selected_model][
                        "embedding_size"
                    ]

                    # Check if store is initialized before creating collection
                    if ingester.store is None:
                        st.error(
                            "Store is not initialized. Please check your configuration."
                        )
                        return

                    ingester.store.create_collection(
                        collection_name=collection_name,
                        embedding_size=embedding_size,
                        distance=distance_metric,
                    )

                    # Ingest data
                    ingester.ingest_category(category, batch_size=batch_size)

                    # Update status
                    collection_info = ingester.store.client.get_collection(
                        collection_name=collection_name
                    )
                    st.session_state.ingestion_status[category] = {
                        "status": "completed",
                        "processed": collection_info.points_count,
                        "vectors": collection_info.vectors_count,
                    }

            except Exception as e:
                st.session_state.ingestion_status[category] = {
                    "status": "failed",
                    "error": str(e),
                }

        st.success("Data ingestion completed!")
        st.rerun()

    except Exception as e:
        st.error(f"Data ingestion failed: {e}")


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
    disease_path = "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/original_dataset/disease"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 Disease Categories File Count")

        if st.button("🔄 Refresh File Counts", use_container_width=True):
            st.session_state.disease_file_counts = count_files_in_directories(
                disease_path
            )

        if "disease_file_counts" not in st.session_state:
            st.session_state.disease_file_counts = count_files_in_directories(
                disease_path
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
    pest_path = "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/original_dataset/pest"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 Pest Categories File Count")

        if st.button("🔄 Refresh File Counts", use_container_width=True):
            st.session_state.pest_file_counts = count_files_in_directories(pest_path)

        if "pest_file_counts" not in st.session_state:
            st.session_state.pest_file_counts = count_files_in_directories(pest_path)

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

    # Search parameters
    st.subheader("⚙️ Search Parameters")

    search_category = st.radio(
        "Search Category",
        options=["disease", "pest"],
        index=0 if st.session_state.get("category", "disease") == "disease" else 1,
        help="Choose whether to search for disease or pest",
    )
    st.session_state.category = search_category

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
            image_dir = Path(Config.DATASET_DIR) / "disease" / "images"
            collection_name = f"{st.session_state.collection_name_prefix}_disease"
        else:
            image_dir = Path(Config.DATASET_DIR) / "pest" / "images"
            collection_name = f"{st.session_state.collection_name_prefix}_pest"

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


# Create sidebar settings
create_sidebar_settings()

# Navigation
st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Navigation")

page_options = ["Data Ingestion", "Disease Report", "Pest Report", "Image Search"]
current_page = st.sidebar.selectbox(
    "Select Page",
    options=page_options,
    index=page_options.index(st.session_state.current_page),
)

if current_page != st.session_state.current_page:
    st.session_state.current_page = current_page
    st.rerun()

# Display current page
if st.session_state.current_page == "Data Ingestion":
    create_data_ingestion_page()
elif st.session_state.current_page == "Disease Report":
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
