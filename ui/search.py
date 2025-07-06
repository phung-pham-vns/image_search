import re
import streamlit as st
import streamlit_cropper
from PIL import Image
from pathlib import Path
from typing import List, Dict
from src.config import Config
from src.generate import generate_answer_from_retrieval
from streamlit_cropper import st_cropper
from ui.sidebar import EMBEDDING_MODELS


def short_model_name(model_name_or_path: str) -> str:
    name = model_name_or_path.lower()
    name = re.sub(r"^.*[\\/]|openai/|google/|facebook/", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def initialize_components(cfg: Config):
    """Initialize embedding model and vector store"""
    if (
        st.session_state.embedder is None
        or st.session_state.selected_model != st.session_state.get("last_model")
    ):
        try:
            with st.spinner(f"Loading {st.session_state.selected_model}..."):
                from src.embeddings.image_embedding import ImageEmbedding
                from src.vector_stores.qdrant import QdrantVectorStore

                model_path = EMBEDDING_MODELS[st.session_state.selected_model][
                    "model_path"
                ]
                cfg.MODEL_NAME_OR_PATH = model_path
                st.session_state.embedder = ImageEmbedding(
                    model_name_or_path=model_path, device="cpu"
                )
                st.session_state.last_model = st.session_state.selected_model
                st.session_state.store = QdrantVectorStore(
                    uri=st.session_state.qdrant_uri
                )
        except Exception as e:
            st.error(f"Failed to initialize model: {e}")
            return False
    return True


def perform_search(cfg: Config, query_image: Image.Image) -> List[Dict]:
    """Perform similarity search"""
    try:
        # Get embedding
        query_embedding = st.session_state.embedder.embed(query_image)

        if query_embedding is None:
            st.error("Failed to generate embedding for the query image")
            return []

        # Configure collection based on category
        model_path = EMBEDDING_MODELS[st.session_state.selected_model]["model_path"]
        model_part = short_model_name(model_path)
        if st.session_state.category == "disease":
            image_dir = Path(cfg.DATASET_DIR) / "disease" / "images"
            collection_name = (
                f"{st.session_state.collection_name_prefix}_disease_{model_part}"
            )
        else:
            image_dir = Path(cfg.DATASET_DIR) / "pest" / "images"
            collection_name = (
                f"{st.session_state.collection_name_prefix}_pest_{model_part}"
            )

        # Query vector store
        results = st.session_state.store.query(
            collection_name=collection_name,
            query_vector=query_embedding,
            top_k=st.session_state.top_k,
        )

        # Process results
        processed_results = []
        for hit in results:
            img_name = hit.payload.get("image_name")
            label = hit.payload.get(st.session_state.category)
            score = hit.score
            payload = hit.payload

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
                        "payload": payload,
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

            payload = result.get("payload", {})
            # Show local name, scientific name, english name, and similarity
            local_name = (
                payload.get("disease")
                or payload.get("pest")
                or payload.get("local_name")
            )
            scientific_name = payload.get("scientific_name")
            english_name = payload.get("english_translation") or payload.get(
                "english_name"
            )
            similarity = result["score"]

            st.markdown(f"**Local Name:** {local_name if local_name else '-'}")
            st.markdown(
                f"**Scientific Name:** {scientific_name if scientific_name else '-'}"
            )
            st.markdown(f"**English Name:** {english_name if english_name else '-'}")
            st.markdown(f"**Similarity:** `{similarity:.3f}`")

            # Expander for all metadata
            with st.expander("Show all metadata"):
                for k, v in payload.items():
                    st.markdown(f"**{k}:** {v}")


def create_search_page(cfg: Config):
    """Page 3: Image Search with enhanced UI"""
    st.title("🔍 Durian Image Similarity Search")
    st.caption("🔍 Image Search")

    # Use sidebar values
    search_category = st.session_state.get("category", "disease")
    top_k = st.session_state.get("top_k", 5)
    distance_metric = st.session_state.get("distance_metric", "cosine")

    # Initialize components
    if not initialize_components(cfg):
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
                # Ensure only the image is passed to perform_search, not a tuple or dict
                if isinstance(cropped_img, Image.Image):
                    st.session_state.search_results = perform_search(cfg, cropped_img)
                elif isinstance(cropped_img, tuple) and isinstance(
                    cropped_img[0], Image.Image
                ):
                    st.session_state.search_results = perform_search(
                        cfg, cropped_img[0]
                    )
                else:
                    st.error("Invalid image format for search.")
                    st.session_state.search_results = None
                st.session_state.llm_answer = None  # Reset any previous answer
                st.rerun()

    with col2:
        st.subheader("📋 Search Results")

        if "search_results" in st.session_state and st.session_state.search_results:
            display_search_results(st.session_state.search_results)

            # --- New: Question block for LLM answer ---
            st.markdown("---")
            st.subheader("💬 Ask a Question about the Results")
            user_question = st.text_input(
                "Ask a question about the retrieved images (e.g., 'What is the most common disease?')",
                key="llm_question_input",
            )
            if user_question:
                # Prepare metadata for LLM
                retrieved_metadata = []
                for result in st.session_state.search_results:
                    # You may want to include more fields as needed
                    retrieved_metadata.append(
                        {
                            "label": result.get("label"),
                            "filename": result.get("filename"),
                            # Add more fields if available
                        }
                    )
                # Call LLM answer generator
                with st.spinner("Generating answer..."):
                    answer = generate_answer_from_retrieval(
                        user_question,
                        retrieved_metadata,
                        # Optionally pass vectors if you want
                    )
                    st.session_state.llm_answer = answer
            # Display answer if available
            if st.session_state.get("llm_answer"):
                st.markdown(f"""**Answer:**\n\n{st.session_state.llm_answer}""")
        else:
            st.info(
                "👆 Upload an image and click 'Search Similar Images' to see results"
            )
