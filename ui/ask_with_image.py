import streamlit as st
from PIL import Image

from src.core.config import Config
from src.core.search import ImageSearcher
from processing.generate import generate_answer_from_retrieval


def initialize_searcher(cfg: Config) -> ImageSearcher:
    """Initializes or retrieves the ImageSearcher from session state."""
    searcher_key = f"searcher_{st.session_state.selected_model}"

    if searcher_key not in st.session_state:
        with st.spinner(f"Loading model '{st.session_state.selected_model}'..."):
            st.session_state[searcher_key] = ImageSearcher(cfg)

    return st.session_state[searcher_key]


def create_ask_with_image_page(cfg: Config):
    """Page for asking questions about an uploaded image."""
    st.title("🖼️ Ask with Image")
    st.markdown(
        "Upload an image, and an AI assistant will answer your questions about it based on similar images from the database."
    )

    try:
        searcher = initialize_searcher(cfg)
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return

    # Initialize chat historyg
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([0.4, 0.6])

    with col1:
        st.subheader("📤 Query Image")
        category = st.radio(
            "Search Category:", options=["disease", "pest", "variety"], horizontal=True
        )
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Your uploaded image", use_container_width=True)

    with col2:
        st.subheader("💬 Chat with AI Assistant")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get user input
        if prompt := st.chat_input("Ask a question about the image..."):
            if not uploaded_file:
                st.warning("Please upload an image first.")
                return

            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.spinner("Thinking..."):
                try:
                    # 1. Find similar images
                    query_image = Image.open(uploaded_file).convert("RGB")
                    retrieved_docs = searcher.search(
                        query_image=query_image, category=category, top_k=cfg.TOP_K
                    )

                    # 2. Generate answer using retrieved context
                    retrieved_metadata = [
                        doc.get("payload", {}) for doc in retrieved_docs
                    ]
                    answer = generate_answer_from_retrieval(
                        question=prompt,
                        retrieved_metadata=retrieved_metadata,
                    )

                    # 3. Add AI response to chat history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                    # Rerun to display the new message
                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    if not uploaded_file:
        st.info("Upload an image to start the conversation.")
