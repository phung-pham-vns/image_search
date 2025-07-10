import re
import pandas as pd
import streamlit as st
from src.core.config import Config
from src.services.ingestion import DataIngester
from ui.sidebar import EMBEDDING_MODELS


def short_model_name(model_name_or_path: str) -> str:
    name = model_name_or_path.lower()
    name = re.sub(r"^.*[\\/]|openai/|google/|facebook/", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def run_ingestion(cfg: Config, categories: list[str]):
    """
    Runs the data ingestion process and updates the session state with the status.
    """
    if not categories:
        st.error("Please select at least one category to ingest.")
        return

    st.session_state.ingestion_status = {
        cat: {"status": "in_progress", "details": ""} for cat in categories
    }
    st.rerun()  # Rerun to show the initial "in_progress" status


def create_data_ingestion_page(cfg: Config):
    """Creates the UI for the data ingestion page."""
    st.title("📥 Data Ingestion")
    st.markdown(
        "Select data categories and start the process to populate the vector database. "
        "This will create and fill collections based on the selected embedding model."
    )

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("⚙️ Ingestion Configuration")

        # Get available directories from the processed dataset path
        try:
            available_categories = [
                d.name for d in cfg.PROCESSED_DATASET_DIR.iterdir() if d.is_dir()
            ]
        except FileNotFoundError:
            st.error(f"Dataset directory not found at: {cfg.PROCESSED_DATASET_DIR}")
            available_categories = []

        categories_to_ingest = st.multiselect(
            "Select categories to ingest:",
            options=available_categories,
            help="Choose the data categories to process and upload to the database.",
        )

        st.info(
            f"Ingestion will use the **{st.session_state.selected_model}** model. "
            f"Data will be stored in collections prefixed with "
            f"**'{cfg.COLLECTION_NAME_PREFIX}'**."
        )

        if st.button(
            "🚀 Start Ingestion",
            use_container_width=True,
            type="primary",
            disabled=not available_categories,
        ):
            run_ingestion(cfg, categories_to_ingest)

    with col2:
        st.subheader("📊 Ingestion Status")
        status_container = st.container(height=300)

        if (
            "ingestion_status" not in st.session_state
            or not st.session_state.ingestion_status
        ):
            status_container.info("Awaiting ingestion to start.")
        else:
            for category, info in st.session_state.ingestion_status.items():
                status = info.get("status", "unknown")
                details = info.get("details", "")

                if status == "completed":
                    status_container.success(
                        f"✅ **{category.capitalize()}:** Completed\n\n{details}"
                    )
                elif status == "failed":
                    status_container.error(
                        f"❌ **{category.capitalize()}:** Failed\n\n{details}"
                    )
                elif status == "in_progress":
                    status_container.info(
                        f"⏳ **{category.capitalize()}:** In Progress..."
                    )

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
                        "Status": "✅ Completed",
                        "Details": status.get("details", ""),
                    }
                )

        if results_data:
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
