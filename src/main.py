import os
import tempfile
import logging
from pathlib import Path
from typing import List

import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Project-specific imports (with fallback)
try:
    from src.utils.file_processor import process_file
    from src.utils.chunker import Chunker
    from src.utils.utils import AVAILABLE_MODELS
    from src.services.vector_db import VectorStore
    from src.services.rag import RAGSystem
    from src.services.translator import TextTranslator
except ImportError:
    st.error("Cannot import from src/, trying fallback...")
    try:
        from utils.file_processor import process_file
        from utils.chunker import Chunker
        from utils.utils import AVAILABLE_MODELS
        from services.vector_db import VectorStore
        from services.rag import RAGSystem
        from services.translator import TextTranslator
    except ImportError as e:
        st.error(f"Missing required modules: {e}")
        st.stop()

# Ensure models directory exists
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def download_model(name: str, url: str, dest: Path) -> bool:
    """Download a model file with a progress bar."""
    st.info(f"Downloading model '{name}'...")
    progress_bar = st.progress(0.0)
    status = st.empty()
    
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB

            status.text(f"Size: {total_size/(1024*1024):.2f} MB" if total_size else "Size: unknown")
            
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = downloaded / total_size if total_size else 0.0
                        progress_bar.progress(min(progress, 1.0))
                        status.text(
                            f"{downloaded/(1024*1024):.2f}/{total_size/(1024*1024):.2f} MB"
                            if total_size else f"{downloaded/(1024*1024):.2f} MB"
                        )

            progress_bar.progress(1.0)
            status.success("Download complete")
            return True

    except Exception as e:
        st.error(f"Download failed: {e}")
        progress_bar.empty()
        status.empty()
        if dest.exists():
            dest.unlink()
        return False

def process_documents(
    uploaded_files: List[UploadedFile],
    llm,
    model_path: str
):
    """Process documents through extraction, translation, chunking, and indexing."""
    # Configure logging
    debug_mode = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not model_path or not Path(model_path).is_file():
        st.error(f"Invalid LLM path: {model_path}")
        return None
    if not uploaded_files:
        st.warning("No files uploaded.")
        return None

    # Initialize services
    try:
        vector_db = VectorStore()
        chunker = Chunker()
        translator = TextTranslator(llm=llm, debug=debug_mode)
        rag = RAGSystem(vector_store=vector_db, llm=llm)
    except Exception as e:
        logger.exception("Service initialization error")
        st.error(f"Failed to initialize services: {e}")
        return None

    total_files = len(uploaded_files)
    progress_bar = st.progress(0.0)
    status_display = st.empty()
    processed_files = []
    failed_files = []

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        file_name = uploaded_file.name
        status_display.text(f"Processing [{idx}/{total_files}]: {file_name}")
        progress_bar.progress(idx/total_files)

        file_ext = Path(file_name).suffix.lower()
        if file_ext not in {".docx", ".pdf", ".csv", ".xlsx", ".xls", ".xlsm"}:
            st.warning(f"Skipped unsupported file: {file_name}")
            continue

        try:
            # Use context manager for temp file handling
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            # Process file contents
            pages = process_file(tmp_path)
            if not pages:
                st.warning(f"No content extracted from {file_name}")
                continue

            processed_docs = translator.process_documents(pages)
            if not processed_docs:
                st.warning(f"No processed documents from {file_name}")
                continue

            chunks = chunker.chunk_document(file_name, processed_docs)
            if chunks:
                vector_db.add_chunks(chunks)
                processed_files.append(file_name)
            else:
                st.warning(f"No chunks generated from {file_name}")

        except Exception as e:
            logger.exception(f"Error processing {file_name}")
            failed_files.append(file_name)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning(f"Error deleting temp file: {e}")

    # Update final status
    progress_bar.progress(1.0)
    status_display.empty()

    if processed_files:
        st.success(f"Successfully processed: {', '.join(processed_files)}")
    if failed_files:
        st.error(f"Failed to process: {', '.join(failed_files)}")
        return None

    st.balloons()
    return rag

def main():
    st.set_page_config(
        page_title="Dr. X Research Assistant",
        layout="wide",
        page_icon="🔬"
    )
    st.title("🔬 Dr. X Research Assistant")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "1. Select Language Model",
            options=list(AVAILABLE_MODELS.keys()))
        model_info = AVAILABLE_MODELS[model_choice]
        model_path = MODELS_DIR / model_info["filename"]

        # Model download handling
        if model_path.exists():
            st.success(f"Model available: {model_choice}")
            st.session_state.model_path = str(model_path)
        else:
            st.warning("Selected model not found")
            if st.button(f"Download {model_choice}"):
                if download_model(model_choice, model_info["url"], model_path):
                    st.session_state.model_path = str(model_path)
                    st.rerun()

        # Document upload
        st.markdown("---")
        uploaded_docs = st.file_uploader(
            "2. Upload Research Documents",
            type=["docx", "pdf", "csv", "xlsx", "xls", "xlsm"],
            accept_multiple_files=True
        )

        # Processing control
        st.markdown("---")
        process_disabled = not ("model_path" in st.session_state and uploaded_docs)
        help_text = "Requires model download and document upload" if process_disabled else ""
        
        if st.button(
            "3. Process Documents",
            disabled=process_disabled,
            help=help_text
        ):
            try:
                from llama_cpp import Llama
                llm = Llama(
                    model_path=st.session_state.model_path,
                    n_ctx=4096,
                    n_threads=os.cpu_count() or 4
                )
                rag_system = process_documents(
                    uploaded_docs,
                    llm,
                    st.session_state.model_path
                )
                
                if rag_system:
                    st.session_state.rag = rag_system
                    st.success("✅ System ready for queries")
                else:
                    st.error("❌ Document processing failed")
                    
            except Exception as e:
                st.error(f"Initialization error: {e}")

    # Main interaction area
    st.markdown("---")
    if "rag" in st.session_state:
        st.header("Research Query Interface")
        user_query = st.text_input("Enter your research question:")
        
        if st.button("Get Answer") and user_query:
            try:
                with st.spinner("Analyzing documents..."):
                    answer = st.session_state.rag.generate_answer(user_query)
                st.markdown("**Research Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Query failed: {e}")
    else:
        st.info("Please configure and process documents to enable queries.")

if __name__ == "__main__":
    main()