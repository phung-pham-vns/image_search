import re
import streamlit as st
import pandas as pd
from pathlib import Path
from src.config import Config
from src.ingest_data import DataIngester
from ui.sidebar import EMBEDDING_MODELS


def short_model_name(model_name_or_path: str) -> str:
    name = model_name_or_path.lower()
    name = re.sub(r"^.*[\\/]|openai/|google/|facebook/", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def perform_data_ingestion(
    cfg: Config,
    categories: list[str],
    batch_size: int,
    distance_metric: str,
):
    """Perform the actual data ingestion"""
    try:
        # Create a new config with updated values
        cfg.QDRANT_URI = st.session_state.qdrant_uri
        cfg.COLLECTION_NAME_PREFIX = st.session_state.collection_name_prefix
        cfg.EMBEDDING_NAME = EMBEDDING_MODELS[st.session_state.selected_model][
            "model_path"
        ]
        cfg.MODEL_NAME_OR_PATH = EMBEDDING_MODELS[st.session_state.selected_model][
            "model_path"
        ]
        cfg.DEVICE = "cpu"  # Default device
        cfg.DATASET_DIR = "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/processed_dataset"
        cfg.PROCESSED_DATASET_DIR = "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/processed_dataset"

        # Initialize ingester with updated config
        ingester = DataIngester(cfg)

        # Update ingestion status
        for category in categories:
            st.session_state.ingestion_status[category] = {
                "status": "in_progress",
                "progress": 0,
            }

        model_part = short_model_name(cfg.MODEL_NAME_OR_PATH)
        # Perform ingestion for each category
        for category in categories:
            try:
                with st.spinner(f"Ingesting {category}..."):
                    # Create collection
                    collection_name = f"{st.session_state.collection_name_prefix}_{category}_{model_part}"
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


def create_data_ingestion_page(cfg: Config):
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
            value="/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/processed_dataset",
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

            # Start ingestion
            perform_data_ingestion(cfg, categories, batch_size, distance_metric)

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
        model_part = short_model_name(
            EMBEDDING_MODELS[st.session_state.selected_model]["model_path"]
        )
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
                        "Collection Name": f"{st.session_state.collection_name_prefix}_{category}_{model_part}",
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
