import streamlit as st
from src.config import Config
from ui.sidebar import create_sidebar_settings, create_navigation
from ui.data_ingestion import create_data_ingestion_page
from ui.reports import create_disease_report_page, create_pest_report_page
from ui.search import create_search_page
from ui.ask_with_image import create_ask_with_image_page

# Page configuration
st.set_page_config(
    page_title="Durian Image Retrieval System",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "qdrant_uri" not in st.session_state:
    st.session_state.qdrant_uri = "http://localhost:6333/"
if "collection_name_prefix" not in st.session_state:
    st.session_state.collection_name_prefix = "durian"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "SigLIP2 Base"
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Data Ingestion"
if "ingestion_status" not in st.session_state:
    st.session_state.ingestion_status = {}

# Create sidebar
create_sidebar_settings()
current_page = create_navigation()

# Main content area
if current_page == "Data Ingestion":
    create_data_ingestion_page()
elif current_page == "Disease Report":
    create_disease_report_page()
elif current_page == "Pest Report":
    create_pest_report_page()
elif current_page == "Image Search":
    create_search_page()
elif current_page == "Ask with Image":
    create_ask_with_image_page()
else:
    st.error(f"Unknown page: {current_page}")

# Footer
st.markdown("---")
st.markdown(
    "🥭 **Durian Image Retrieval System** - Built with Streamlit, Qdrant, and AI Models"
)
