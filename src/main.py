import os
import tempfile
from pathlib import Path
import streamlit as st

from utils.file_processor import process_file
from utils.chunker import Chunker
from services.vector_db import VectorStore
from services.rag import RAGSystem

def get_model_files(models_dir: str = "./models"):
    """
    Returns a list of available model files in the static model directory.
    Displays an error if the directory does not exist or is empty.
    """
    models_path = Path(models_dir).expanduser().resolve()
    if not models_path.exists() or not models_path.is_dir():
        st.error(f"Models directory {models_path} not found.")
        return []
    model_files = [f for f in models_path.iterdir() if f.is_file()]
    if not model_files:
        st.error(f"No model files found in {models_path}.")
    return model_files

def process_documents(uploaded_files, llm_model: str) -> RAGSystem:
    """
    Processes each uploaded document by:
      - Saving the document to a temporary file.
      - Processing the file (e.g. chunking its content).
      - Adding the chunks to a vector store.
    Returns an initialized RAG (Retrieval-Augmented Generation) system.
    """
    # Initialize services
    vector_db = VectorStore()
    chunker = Chunker()
    rag = RAGSystem(vector_db, llm_model)

    # Set up progress reporting
    progress_bar = st.progress(0)
    status_text = st.empty()
    num_files = len(uploaded_files)
    processed_files = 0

    for file in uploaded_files:
        file_name = file.name
        status_text.text(f"Processing file: {file_name}...")
        suffix = Path(file_name).suffix.lower()

        # Only process supported file types
        if suffix in {'.docx', '.pdf', '.csv', '.xlsx', '.xls', '.xlsm'}:
            try:
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                # Process and chunk the document
                pages = process_file(tmp_path)
                chunks = chunker.chunk_document(file_name, pages)
                st.info(f"{len(chunks)} chunks created for file: {file_name}")
                vector_db.add_chunks(chunks)
            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")
        else:
            st.warning(f"Unsupported file type for file: {file_name}")

        processed_files += 1
        # Update the progress bar for each processed file
        progress_percent = int((processed_files / num_files) * 100)
        progress_bar.progress(progress_percent)

    status_text.text("Completed processing all documents!")
    st.success("Documents processed successfully!")
    return rag

def main():
    st.title("Dr. X Research Assistant")
    st.write("Upload your documents and select an LLM model to begin.")

    st.sidebar.header("Configuration")
    model_files = get_model_files("./models")
    if model_files:
        model_options = {model_file.name: str(model_file) for model_file in model_files}
        selected_model_name = st.sidebar.selectbox("Select LLM Model:", list(model_options.keys()))
        selected_model = model_options[selected_model_name]
    else:
        selected_model = None

    # File uploader for document uploads
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents",
        type=["docx", "pdf", "csv", "xlsx", "xls", "xlsm"],
        accept_multiple_files=True
    )

    if st.sidebar.button("Process Documents"):
        if not selected_model:
            st.error("No model selected. Please ensure models exist in the './models' directory.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            # Use a spinner to indicate overall processing
            with st.spinner("Processing documents..."):
                rag = process_documents(uploaded_files, selected_model)
                st.session_state.rag = rag

    # Q&A session area
    if "rag" in st.session_state:
        st.markdown("## Q&A Session")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                try:
                    answer = st.session_state.rag.generate_answer(question)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
