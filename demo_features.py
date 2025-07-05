#!/usr/bin/env python3
"""
Demo script showcasing the new features of the Durian Image Retrieval System.

This script demonstrates:
1. Data ingestion capabilities
2. Sidebar settings configuration
3. Database status monitoring
4. Model specifications
"""

import streamlit as st
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from app import (
    EMBEDDING_MODELS,
    create_sidebar_settings,
    check_database_status,
    create_data_ingestion_page,
)


def demo_embedding_models():
    """Demonstrate the embedding model specifications"""
    st.header("🤖 Embedding Model Specifications")
    st.write(
        "The system includes 6 pre-configured embedding models with detailed specifications:"
    )

    for model_name, specs in EMBEDDING_MODELS.items():
        with st.expander(f"📋 {model_name}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Description:** {specs['description']}")
                st.markdown(f"**Purpose:** {specs['purpose']}")
                st.markdown(f"**Architecture:** {specs['architecture']}")

            with col2:
                st.markdown(f"**Embedding Size:** {specs['embedding_size']} dimensions")
                st.markdown(f"**Model Size:** {specs['model_size']}")
                st.markdown(f"**Training Data:** {specs['training_data']}")
                st.markdown(f"**Performance:** {specs['performance']}")


def demo_sidebar_settings():
    """Demonstrate the sidebar settings functionality"""
    st.header("⚙️ Sidebar Settings Demo")
    st.write("The sidebar provides comprehensive configuration options:")

    # Simulate sidebar settings
    with st.sidebar:
        st.title("⚙️ Settings Demo")

        # Vector Store Configuration
        st.subheader("🗄️ Vector Store")
        st.text_input("Qdrant URI", value="http://localhost:6333/", disabled=True)
        st.text_input("Collection Name Prefix", value="durian", disabled=True)

        # Embedding Model Selection
        st.subheader("🤖 Embedding Model")
        selected_model = st.selectbox(
            "Choose Model", options=list(EMBEDDING_MODELS.keys()), index=0
        )

        # Model Specifications
        if selected_model in EMBEDDING_MODELS:
            model_specs = EMBEDDING_MODELS[selected_model]
            with st.expander("📋 Model Specifications", expanded=False):
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
        st.subheader("💻 Hardware")
        st.selectbox("Device", options=["cpu", "cuda", "mps"], index=0, disabled=True)

        # Search Configuration
        st.subheader("🔍 Search Settings")
        st.slider("Top K Results", min_value=1, max_value=20, value=5, disabled=True)
        st.toggle("Show Metadata", value=True, disabled=True)

        # Database Status
        st.subheader("📊 Database Status")
        if st.button("🔄 Check Database Status", use_container_width=True):
            st.success("Database status updated!")
            st.info("📁 durian_diseases: 1,234 points")
            st.info("📁 durian_pests: 567 points")


def demo_data_ingestion():
    """Demonstrate the data ingestion functionality"""
    st.header("📥 Data Ingestion Demo")
    st.write(
        "The data ingestion page provides comprehensive data management capabilities:"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("⚙️ Ingestion Configuration")

        # Dataset paths
        st.text_input(
            "Dataset Directory", value="/path/to/processed/dataset", disabled=True
        )

        # Categories to ingest
        st.multiselect(
            "Categories to Ingest",
            options=["diseases", "pests"],
            default=["diseases", "pests"],
            disabled=True,
        )

        # Batch size
        st.slider("Batch Size", min_value=1, max_value=64, value=32, disabled=True)

        # Distance metric
        st.selectbox(
            "Distance Metric",
            options=["cosine", "euclid", "dot", "manhattan"],
            index=0,
            disabled=True,
        )

        # Ingestion button
        if st.button(
            "🚀 Start Ingestion (Demo)", type="primary", use_container_width=True
        ):
            st.success("Demo: Ingestion completed successfully!")
            st.info("📊 Processed 1,234 images")
            st.info("🗄️ Created 2 collections")

    with col2:
        st.subheader("📊 Current Status")

        # Show database URL
        st.markdown("**🔗 Database URL:**")
        st.code("http://localhost:6333/dashboard#", language="text")

        # Show ingestion status
        st.subheader("📈 Ingestion Progress")
        with st.expander("📁 diseases", expanded=True):
            st.success("✅ Completed")
            st.metric("Images Processed", 1234)
            st.metric("Vectors Stored", 1234)

        with st.expander("📁 pests", expanded=True):
            st.success("✅ Completed")
            st.metric("Images Processed", 567)
            st.metric("Vectors Stored", 567)


def demo_database_access():
    """Demonstrate database access features"""
    st.header("🔗 Database Access Demo")
    st.write("After ingestion, you can access the database through various methods:")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌐 Qdrant Dashboard")
        st.markdown(
            "**Main Dashboard:** [http://localhost:6333/](http://localhost:6333/)"
        )
        st.markdown("**Collections:**")
        st.markdown(
            "- [durian_diseases](http://localhost:6333/dashboard#/collections/durian_diseases)"
        )
        st.markdown(
            "- [durian_pests](http://localhost:6333/dashboard#/collections/durian_pests)"
        )

        st.info(
            "💡 Click the links above to access the Qdrant dashboard and collections"
        )

    with col2:
        st.subheader("📊 Collection Information")

        # Simulate collection info
        st.markdown("**durian_diseases Collection:**")
        st.markdown("- **Points:** 1,234")
        st.markdown("- **Vectors:** 1,234")
        st.markdown("- **Status:** Green")
        st.markdown("- **Distance:** Cosine")

        st.markdown("**durian_pests Collection:**")
        st.markdown("- **Points:** 567")
        st.markdown("- **Vectors:** 567")
        st.markdown("- **Status:** Green")
        st.markdown("- **Distance:** Cosine")


def main():
    """Main demo function"""
    st.set_page_config(
        page_title="Durian Image Retrieval - Feature Demo",
        page_icon="🥭",
        layout="wide",
    )

    st.title("🥭 Durian Image Retrieval System - Feature Demo")
    st.caption("Demonstrating the new data ingestion and sidebar features")

    # Navigation
    demo_sections = {
        "Embedding Models": demo_embedding_models,
        "Sidebar Settings": demo_sidebar_settings,
        "Data Ingestion": demo_data_ingestion,
        "Database Access": demo_database_access,
    }

    selected_section = st.sidebar.selectbox(
        "Select Demo Section", options=list(demo_sections.keys())
    )

    # Display selected section
    demo_sections[selected_section]()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            🥭 Durian Image Retrieval System | Feature Demo
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
