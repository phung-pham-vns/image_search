import streamlit as st
from PIL import Image
from pathlib import Path
from src.config import Config
from src.generate import generate_answer_from_retrieval


def initialize_components():
    """Initialize embedding model and vector store"""
    if (
        st.session_state.embedder is None
        or st.session_state.selected_model != st.session_state.get("last_model")
    ):
        try:
            with st.spinner(f"Loading {st.session_state.selected_model}..."):
                from ui.sidebar import EMBEDDING_MODELS
                from src.embeddings.image_embedding import ImageEmbedding
                from src.vector_stores.qdrant import QdrantVectorStore

                model_path = EMBEDDING_MODELS[st.session_state.selected_model][
                    "model_path"
                ]
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


def create_ask_with_image_page():
    """Page: Ask with Image (upload image, ask a question, get an answer, multi-turn)"""
    st.title("🖼️ Ask with Image")
    st.caption(
        "Upload an image and have a multi-turn conversation about it. You can set the top result as the focus for follow-up questions."
    )

    # Initialize session state for chat and focus
    if "ask_image_chat_history" not in st.session_state:
        st.session_state.ask_image_chat_history = (
            []
        )  # List of dicts: {question, answer}
    if "ask_image_focus_metadata" not in st.session_state:
        st.session_state.ask_image_focus_metadata = None
    if "ask_image_last_results" not in st.session_state:
        st.session_state.ask_image_last_results = []

    # Upload image
    uploaded_file = st.file_uploader(
        "Upload an image to ask about",
        type=["jpg", "jpeg", "png", "bmp"],
        key="ask_image_uploader",
    )

    if uploaded_file:
        # Save uploaded file temporarily
        with open("figures/temp_ask.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Show uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Retrieve similar images and metadata (only once per upload)
        if not st.session_state.ask_image_last_results:
            if not initialize_components():
                st.error(
                    "Failed to initialize the embedding model. Please check your configuration."
                )
                return
            embedder = st.session_state.embedder
            query_img = embedder.preprocess_image(uploaded_file)
            query_embedding = embedder.embed(query_img)
            if query_embedding is None:
                st.error("Failed to generate embedding for the uploaded image.")
                return
            category = st.session_state.get("category", "disease")
            top_k = st.session_state.get("top_k", 5)
            if category == "disease":
                collection_name = f"{st.session_state.collection_name_prefix}_disease"
            else:
                collection_name = f"{st.session_state.collection_name_prefix}_pest"
            results = st.session_state.store.query(
                collection_name=collection_name,
                query_vector=query_embedding,
                top_k=top_k,
            )
            st.session_state.ask_image_last_results = results
        else:
            results = st.session_state.ask_image_last_results

        # Show top result and allow setting as focus
        if results:
            top_result = results[0]
            top_label = (
                top_result.payload.get("disease")
                or top_result.payload.get("pest")
                or top_result.payload.get("local_name")
            )
            st.markdown(f"**Top Result:** {top_label if top_label else '-'}")
            if st.button("Set as Focus (for follow-up questions)", key="set_focus_btn"):
                st.session_state.ask_image_focus_metadata = top_result.payload
                st.success("Focus set! Follow-up questions will use this as context.")

        # Show chat history
        st.markdown("---")
        st.subheader("💬 Conversation")
        for turn in st.session_state.ask_image_chat_history:
            st.markdown(f"**You:** {turn['question']}")
            st.markdown(f"**Answer:** {turn['answer']}")
            st.markdown("---")

        # Ask a question
        user_question = st.text_input(
            "Ask a question about this image (multi-turn supported)",
            key="ask_image_question_input",
        )

        if user_question:
            # Prepare context for LLM
            conversation = st.session_state.ask_image_chat_history
            focus_metadata = st.session_state.ask_image_focus_metadata
            # Compose context: focus metadata (if set) + retrieved metadata
            if focus_metadata:
                context_metadata = [focus_metadata]
            else:
                context_metadata = [hit.payload for hit in results]
            # Compose conversation history as a string
            chat_history_str = "\n".join(
                [f"Q: {turn['question']}\nA: {turn['answer']}" for turn in conversation]
            )
            # Compose prompt for LLM
            prompt = (
                f"Conversation so far:\n{chat_history_str}\n\n"
                f"User's new question: {user_question}"
            )
            # Generate answer
            with st.spinner("Generating answer..."):
                answer = generate_answer_from_retrieval(
                    prompt,
                    context_metadata,
                )
            # Add to chat history
            st.session_state.ask_image_chat_history.append(
                {
                    "question": user_question,
                    "answer": answer,
                }
            )
            st.rerun()

        # Optionally show retrieved images/metadata
        with st.expander("Show retrieved similar images and metadata"):
            for i, hit in enumerate(results):
                st.markdown(f"**Result {i+1}:**")
                img_name = hit.payload.get("image_name")
                category = st.session_state.get("category", "disease")
                if category == "disease":
                    image_dir = Path(Config.DATASET_DIR) / "disease" / "images"
                else:
                    image_dir = Path(Config.DATASET_DIR) / "pest" / "images"
                img_path = image_dir / img_name if img_name else None
                if img_path and img_path.exists():
                    st.image(str(img_path), width=200)
                for k, v in hit.payload.items():
                    st.markdown(f"**{k}:** {v}")

    else:
        # Reset state if no image is uploaded
        st.session_state.ask_image_chat_history = []
        st.session_state.ask_image_focus_metadata = None
        st.session_state.ask_image_last_results = []
