# main.py
import os
import time # For potential delays or checks
import tempfile
import logging
from pathlib import Path
from typing import List, Any, Dict, Optional

import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# --- Project Imports ---
try:
    from src.utils.file_processor import process_file
    from src.utils.chunker import Chunker
    from src.utils.utils import AVAILABLE_MODELS
    from src.services.vector_db import VectorStore
    from src.services.rag import RAGSystem
    from src.services.translator import TextTranslator
    from src.services.summarizer import Summarizer
    from src.core.core import DocumentChunk
except ImportError:
    st.error(
        "Fatal Error: Could not import necessary modules from 'src'. "
        "Please ensure the project structure and dependencies are correct. "
        "Application cannot start."
    )
    st.stop()

# --- Configure Logging ---
DEBUG_MODE = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.info(f"Debug mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")

# --- Constants ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_FILE_EXTENSIONS = {".docx", ".pdf", ".csv", ".xlsx", ".xls", ".xlsm", ".txt"}

# --- Helper Functions ---

def download_model(name: str, url: str, dest: Path) -> bool:
    """Downloads a model file with progress updates (enhanced feedback)."""
    if not url:
        st.error(f"No download URL provided for model '{name}'.")
        return False
    st.info(f"Initiating download for model '{name}'...")
    progress_bar = st.progress(0.0, text="Waiting to start...")
    status_text = st.empty()
    status_text.text("Connecting...")
    start_time = time.time()
    try:
        headers = {'Accept-Encoding': 'identity'}
        with requests.get(url, stream=True, headers=headers, timeout=600) as response: # Increased timeout
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0
            chunk_size = 1024 * 1024  # 1 MB
            size_info = f"{total_size / (1024 * 1024):.2f} MB" if total_size else "Unknown size"
            status_text.text(f"Connected. File size: {size_info}. Starting download...")

            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = min(downloaded_size / total_size, 1.0)
                            elapsed_time = time.time() - start_time
                            speed = (downloaded_size / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0
                            progress_bar.progress(progress, text=f"{downloaded_size / (1024 * 1024):.1f} / {size_info} MB ({speed:.2f} MB/s)")
                        else:
                            progress_bar.progress(0.5, text=f"Downloading... {downloaded_size / (1024 * 1024):.1f} MB")

            elapsed_time = time.time() - start_time
            progress_bar.progress(1.0, text=f"Download complete! ({elapsed_time:.1f}s)")
            status_text.success(f"✅ Download successful: {name} ({size_info})")
            logger.info(f"Successfully downloaded model '{name}' to {dest} in {elapsed_time:.1f}s")
            # Short pause for user to see success message
            time.sleep(1)
            return True

    except requests.exceptions.RequestException as e:
        st.error(f"Download failed: {e}")
        logger.error(f"Download failed for '{name}' from {url}: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        logger.exception(f"Unexpected error downloading {name}")
    finally:
        # Clear progress indicators on failure or completion
        progress_bar.empty()
        status_text.empty()
        # Clean up partial file if it exists and download didn't complete successfully
        # Check if file exists AND if response status suggests failure or file size mismatch
        # This is complex, simpler to just delete if it exists and function returns False
        # Note: This might delete a valid file if the check is somehow wrong. Be cautious.
        # A more robust check might compare final file size to expected size if known.

    # If we reach here, an error occurred
    if dest.exists():
        try:
            dest.unlink()
            logger.info(f"Deleted potentially incomplete file: {dest}")
        except OSError as unlink_err:
            logger.warning(f"Could not delete file {dest} after failed download: {unlink_err}")
    return False


def initialize_services(model_path: str) -> Optional[Dict[str, Any]]:
    """Initializes LLM and other required services."""
    if not model_path or not Path(model_path).is_file():
        st.error(f"LLM path is invalid or file not found: {model_path}")
        return None
    try:
        from llama_cpp import Llama # Import here to ensure it's available
        logger.info(f"Initializing Llama model from: {model_path}")
        llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=max(1, os.cpu_count() // 2), # Use half cores, sensible default
            n_batch=512,
            verbose=DEBUG_MODE,
            # n_gpu_layers=-1 # Set layers to offload to GPU if compiled with GPU support
        )
        logger.info("Llama model initialized successfully.")

        # Initialize other dependent services
        vector_db = VectorStore()
        summarizer = Summarizer(llm=llm)
        rag = RAGSystem(vector_store=vector_db, llm=llm)
        translator = TextTranslator(llm=llm, debug=DEBUG_MODE)
        chunker = Chunker()

        logger.info("All services initialized successfully.")
        return {
            'llm': llm,
            'vector_db': vector_db,
            'summarizer': summarizer,
            'rag': rag,
            'translator': translator,
            'chunker': chunker,
        }

    except ImportError:
        st.error("`llama-cpp-python` not installed. Please install it (`pip install llama-cpp-python`).")
        logger.critical("llama-cpp-python package not found.")
        return None
    except Exception as e:
        # Catch specific exceptions from Llama if possible, e.g., model load errors
        st.error(f"Error initializing services: {e}", icon="🔥")
        logger.exception("Failed to initialize services.")
        return None


def process_uploaded_documents(
    uploaded_files: List[UploadedFile],
    services: Dict[str, Any]
) -> bool:
    """Processes uploaded files: extracts, translates, chunks, and adds to VectorStore."""
    if not uploaded_files:
        st.warning("No files were uploaded to process.")
        return False
    required_services = ['vector_db', 'chunker', 'translator']
    if not all(k in services for k in required_services):
         st.error(f"Cannot process documents: Missing required services ({', '.join(s for s in required_services if s not in services)})")
         logger.error("process_uploaded_documents called without required services.")
         return False

    vector_db: VectorStore = services['vector_db']
    chunker: Chunker = services['chunker']
    translator: TextTranslator = services['translator']

    total_files = len(uploaded_files)
    # Use columns for better layout of progress info
    progress_col, status_col = st.columns([3, 1])
    with progress_col:
        progress_bar = st.progress(0.0, text=f"Starting processing for {total_files} file(s)...")
    with status_col:
        status_display = st.empty() # Placeholder for brief status like "Processing X..."
        status_display.text("Starting...")

    processed_files_count = 0
    failed_files_info: Dict[str, str] = {}
    all_chunks_for_session: List[DocumentChunk] = []
    start_process_time = time.time()

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        file_name = uploaded_file.name
        status_display.text(f"File {idx}/{total_files}")
        logger.info(f"Processing file {idx}/{total_files}: {file_name}")

        # Update progress bar text more dynamically
        progress_text = f"Processing: {file_name} ({idx}/{total_files})"
        progress_bar.progress( (idx-1) / total_files, text=progress_text)


        file_ext = Path(file_name).suffix.lower()
        if file_ext not in SUPPORTED_FILE_EXTENSIONS:
            logger.warning(f"Skipping unsupported file type: {file_name} ({file_ext})")
            failed_files_info[file_name] = "Unsupported file type"
            continue

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            logger.debug(f"Saved uploaded file '{file_name}' to temporary path: {tmp_path}")

            # 1. Extract Content
            extracted_content = process_file(tmp_path)
            if not extracted_content:
                logger.warning(f"No content could be extracted from {file_name}")
                failed_files_info[file_name] = "No content extracted"
                continue
            logger.info(f"Extracted content from {file_name}.")

            # 2. Translate Content
            translated_docs = translator.process_documents(extracted_content)
            if not translated_docs:
                 logger.warning(f"Translation step yielded no documents for {file_name}")
                 processed_content = extracted_content # Use original if translation fails/is empty
            else:
                 logger.info(f"Translated content for {file_name}.")
                 processed_content = translated_docs

            # 3. Chunk Document
            chunks = chunker.chunk_document(file_name, processed_content)
            if not chunks:
                logger.warning(f"No chunks were generated for {file_name}")
                failed_files_info[file_name] = "No chunks generated"
                continue
            logger.info(f"Generated {len(chunks)} chunks for {file_name}.")

            all_chunks_for_session.extend(chunks)
            processed_files_count += 1

        except Exception as e:
            logger.exception(f"Error processing file: {file_name}")
            failed_files_info[file_name] = f"Error: {str(e)[:100]}"
        finally:
            if tmp_path and Path(tmp_path).exists():
                try: os.unlink(tmp_path)
                except OSError as e: logger.warning(f"Could not delete temp file {tmp_path}: {e}")
            # Update progress after attempting the file
            progress_bar.progress( idx / total_files, text=progress_text)


    # --- Post-Loop: Add to Vector DB ---
    status_display.text("Finalizing...")
    db_add_success = False
    if all_chunks_for_session:
        try:
            progress_bar.progress(1.0, text=f"Adding {len(all_chunks_for_session)} chunks to knowledge base...")
            vector_db.add_chunks(all_chunks_for_session)
            logger.info(f"Successfully added {len(all_chunks_for_session)} chunks from {processed_files_count} files to VectorStore.")
            # Avoid success message here if failures occurred
            db_add_success = True
        except Exception as e:
            st.error(f"Failed to add processed chunks to Knowledge Base: {e}", icon="💾")
            logger.exception("Error adding chunks to VectorStore.")
            # Keep progress bar at 1.0 but update text
            progress_bar.progress(1.0, text="Error updating knowledge base!")
    elif processed_files_count > 0:
         # Files processed but no chunks generated overall
         logger.warning("Processing finished, but no valid chunks were generated to add to the knowledge base.")
         progress_bar.progress(1.0, text="Processing complete. No new data added.")
    else:
         # No files processed successfully at all
         progress_bar.progress(1.0, text="Processing complete. No files processed.")


    total_process_time = time.time() - start_process_time
    logger.info(f"Document processing finished in {total_process_time:.2f} seconds.")
    # Clear brief status display
    status_display.empty()


    # --- Final Status Report ---
    if failed_files_info:
        st.warning(f"Completed with issues. {len(failed_files_info)} file(s) could not be fully processed:")
        with st.expander("Show Processing Issues"):
             for fname, err in failed_files_info.items():
                 st.caption(f"- {fname}: {err}")

    overall_success = (processed_files_count > 0) and db_add_success and not failed_files_info
    if overall_success:
        st.success(f"✅ Successfully processed {processed_files_count} file(s) in {total_process_time:.1f}s.")
        st.balloons()
    elif processed_files_count > 0 and db_add_success:
         st.success(f"Processed {processed_files_count} file(s) with some issues (see above). Knowledge base updated.")
    elif processed_files_count > 0 and not db_add_success:
         st.error("Processed files but failed to update the knowledge base.")
    else: # No files processed successfully at all
        st.error("Processing complete, but no files were successfully processed.")

    return (processed_files_count > 0) and db_add_success


# --- Main Streamlit Application ---
def main():
    st.set_page_config(
        page_title="Dr. X Research Assistant",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    st.title("🔬 Dr. X Research Assistant")
    st.caption(f"AI-powered analysis for research documents. Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize session state variables
    if "services_initialized" not in st.session_state: st.session_state.services_initialized = False
    if "current_summary" not in st.session_state: st.session_state.current_summary = None
    if "summarized_source" not in st.session_state: st.session_state.summarized_source = None
    if "selected_model_name" not in st.session_state: st.session_state.selected_model_name = None
    if "initialized_model_name" not in st.session_state: st.session_state.initialized_model_name = None

    # === Sidebar ===
    with st.sidebar:
        st.header("⚙️ Configuration")

        # --- 1. Model Selection ---
        st.subheader("1. Select Language Model")
        if not AVAILABLE_MODELS:
             st.error("Model configuration `AVAILABLE_MODELS` is empty!")
             st.stop()

        # Use on_change to reset initialization status if model selection changes
        def model_selection_change():
            logger.debug(f"Model selection changed to: {st.session_state.model_select_widget}")
            # If the selected model is different from the initialized one, reset status
            if st.session_state.services_initialized and \
               st.session_state.model_select_widget != st.session_state.initialized_model_name:
                logger.info(f"Model selection changed from '{st.session_state.initialized_model_name}' to '{st.session_state.model_select_widget}'. Resetting initialization status.")
                st.session_state.services_initialized = False
                st.session_state.initialized_model_name = None
                # Clear potentially model-specific services? Optional.
                # for key in ['llm', 'vector_db', 'summarizer', 'rag', 'translator', 'chunker']:
                #     if key in st.session_state:
                #         del st.session_state[key]


        selected_model_name = st.selectbox(
            "Select Model:",
            options=list(AVAILABLE_MODELS.keys()),
            key="model_select_widget", # Use a distinct key for the widget
            index=None, # Default to no selection
            placeholder="Choose a model...",
            on_change=model_selection_change # Callback when selection changes
        )
        st.session_state.selected_model_name = selected_model_name # Store the choice

        # --- Display Model Status & Initialization Button ---
        model_status_placeholder = st.empty()
        init_button_placeholder = st.empty()

        if selected_model_name:
            model_info = AVAILABLE_MODELS[selected_model_name]
            model_filename = model_info.get("filename")
            model_url = model_info.get("url")
            local_model_path = MODELS_DIR / model_filename if model_filename else None

            model_exists_locally = local_model_path and local_model_path.exists()

            if st.session_state.services_initialized and st.session_state.initialized_model_name == selected_model_name:
                 model_status_placeholder.success(f"✅ Services initialized using '{selected_model_name}'.")
            elif model_exists_locally:
                model_status_placeholder.info(f"ℹ️ Model '{selected_model_name}' found locally. Ready to initialize.")
                if init_button_placeholder.button("🚀 Initialize Services", key="init_existing", type="primary"):
                    with st.spinner(f"Initializing services with {selected_model_name}... Please wait."):
                        services = initialize_services(str(local_model_path))
                        if services:
                            st.session_state.update(services)
                            st.session_state.services_initialized = True
                            st.session_state.initialized_model_name = selected_model_name # Track initialized model
                            model_status_placeholder.success(f"✅ Services initialized successfully!")
                            logger.info(f"Services initialized with existing model: {selected_model_name}")
                            # Short pause and rerun to update main UI
                            time.sleep(1)
                            st.rerun()
                        else:
                            # Error displayed by initialize_services
                            st.session_state.services_initialized = False
                            st.session_state.initialized_model_name = None
            elif model_url:
                 model_status_placeholder.warning(f"⚠️ Model '{selected_model_name}' not found locally. Download required ({model_info.get('size', 'size unknown')}).")
                 if init_button_placeholder.button(f"⬇️ Download & Initialize", key="download_init", type="primary"):
                     download_success = download_model(selected_model_name, model_url, local_model_path)
                     if download_success:
                         # Proceed to initialization
                         with st.spinner(f"Initializing services with {selected_model_name}..."):
                             services = initialize_services(str(local_model_path))
                             if services:
                                 st.session_state.update(services)
                                 st.session_state.services_initialized = True
                                 st.session_state.initialized_model_name = selected_model_name
                                 model_status_placeholder.success(f"✅ Model downloaded & services initialized!")
                                 logger.info(f"Model downloaded and services initialized: {selected_model_name}")
                                 time.sleep(1)
                                 st.rerun()
                             else:
                                 st.session_state.services_initialized = False
                                 st.session_state.initialized_model_name = None
                                 # Keep warning status, error shown by initialize_services
                                 model_status_placeholder.warning(f"⚠️ Model '{selected_model_name}' downloaded, but failed to initialize.")
                     else:
                         # Download failed, error shown by download_model
                         st.session_state.services_initialized = False
                         st.session_state.initialized_model_name = None
                         model_status_placeholder.error(f"🚫 Failed to download model '{selected_model_name}'. Cannot initialize.")

            elif not model_url:
                 model_status_placeholder.error(f"🚫 Model '{selected_model_name}' not found locally and no download URL is configured.")

        else:
            model_status_placeholder.info("Select a language model from the dropdown.")


        st.markdown("---")

        # --- 2. Document Upload ---
        st.subheader("2. Upload Documents")
        # Disable upload if services aren't ready
        upload_disabled = not st.session_state.services_initialized
        upload_help = "Initialize services first before uploading documents." if upload_disabled else "Upload one or more documents for analysis."
        uploaded_docs = st.file_uploader(
            "Select documents:",
            type=list(SUPPORTED_FILE_EXTENSIONS),
            accept_multiple_files=True,
            key="file_uploader",
            disabled=upload_disabled,
            help=upload_help
        )

        st.markdown("---")

        # --- 3. Process Button ---
        st.subheader("3. Process Documents")
        # Enable button only if services are initialized AND files are uploaded
        can_process = st.session_state.services_initialized and uploaded_docs
        process_help = ""
        if not st.session_state.services_initialized: process_help += "Services not initialized. "
        if not uploaded_docs: process_help += "No documents uploaded."

        if st.button("⚙️ Process Uploaded Documents", disabled=not can_process, help=process_help.strip(), type="primary"):
             if can_process:
                logger.info("Processing uploaded documents...")
                # Use a dedicated area for processing feedback
                processing_feedback_area = st.container()
                with processing_feedback_area:
                    # Prepare the dict of services needed
                    services_for_processing = {
                        'vector_db': st.session_state.vector_db,
                        'chunker': st.session_state.chunker,
                        'translator': st.session_state.translator,
                    }
                    # Run processing function
                    success = process_uploaded_documents(uploaded_docs, services_for_processing)
                    if success:
                        logger.info("Document processing completed successfully.")
                        # Rerun to refresh source lists etc.
                        st.rerun()
                    else:
                        logger.error("Document processing failed or had issues.")
             else:
                 # This case should ideally not happen due to disabled state, but log just in case
                 logger.warning("Process button clicked but prerequisites not met.")

    st.markdown("---") # Separator

    # === Main Content Area (Tabs) ===
    if st.session_state.services_initialized:
        tab_qa, tab_summarize = st.tabs(["❓ **Q&A Interface**", "📝 **Summarization**"])

        # --- Q&A Tab ---
        with tab_qa:
            st.header("❓ Ask Questions About Your Documents")
            with st.form(key="qa_form"):
                user_query = st.text_input(
                    "Enter your question:",
                    key="qa_query_input",
                    placeholder="E.g., What are the main findings regarding topic X?"
                )
                submit_qa = st.form_submit_button("🧠 Get Answer", type="primary")

            if submit_qa and user_query:
                if 'rag' in st.session_state:
                    try:
                        with st.spinner("⏳ Thinking..."):
                            answer = st.session_state.rag.generate_answer(user_query)
                        st.markdown("#### Answer:")
                        st.info(answer) # Using info box for visibility
                    except Exception as e:
                        st.error(f"Q&A Error: {e}", icon="🔥")
                        logger.exception(f"Error answering query: {user_query}")
                else:
                    st.error("RAG system component not found. Initialization might have failed.")
            elif submit_qa and not user_query:
                 st.warning("Please enter a question before submitting.")

        # --- Summarization Tab ---
        with tab_summarize:
            st.header("📝 Generate Document Summaries")

            if 'vector_db' in st.session_state and 'summarizer' in st.session_state:
                vector_db: VectorStore = st.session_state.vector_db
                summarizer: Summarizer = st.session_state.summarizer

                try:
                    # Cache this potentially? For now, call directly.
                    available_sources = vector_db.list_all_sources()
                except Exception as e:
                    st.error(f"Error listing documents: {e}", icon="💾")
                    logger.exception("Error calling list_all_sources.")
                    available_sources = []

                if available_sources:
                    # Use a form for summary configuration + action
                    with st.form("summary_form"):
                        st.subheader("1. Configure Summary")
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            selected_source = st.selectbox(
                                "Document to Summarize:", options=available_sources, key="summary_source_select"
                            )
                        with col2:
                             summarize_style = st.selectbox(
                                 "Style:", options=['concise', 'detailed', 'bullet points', 'paragraph'], index=0, key="summary_style_select"
                            )

                        user_max_length = st.slider(
                            "Target Max Length (chars):", min_value=100, max_value=4000, value=1000, step=100, key="summary_length_slider"
                        )
                        generate_summary_button = st.form_submit_button("📄 Generate Summary", type="primary")

                    if generate_summary_button and selected_source:
                         with st.spinner(f"⏳ Generating {summarize_style} summary for '{selected_source}'..."):
                            try:
                                full_text = vector_db.get_chunks_by_source(selected_source, return_text=True)
                                if not full_text:
                                    st.warning(f"No text found for '{selected_source}'.")
                                    st.session_state.current_summary = None
                                    st.session_state.summarized_source = None
                                else:
                                    logger.info(f"Starting summarization for '{selected_source}', length: {len(full_text)} chars.")
                                    summarizer.max_length = user_max_length # Update instance setting
                                    generated_summary = summarizer.summarize(full_text, style=summarize_style)
                                    st.session_state.current_summary = generated_summary
                                    st.session_state.summarized_source = selected_source
                                    logger.info(f"Summary generated for '{selected_source}', result length: {len(generated_summary)} chars.")
                                    # Rerun needed to display the summary outside the form scope cleanly
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Summarization failed: {e}", icon="🔥")
                                logger.exception(f"Error summarizing source: {selected_source}")
                                st.session_state.current_summary = None
                                st.session_state.summarized_source = None
                                # Don't rerun on error, message stays visible

                else:
                    st.info("No documents found in the knowledge base. Use the sidebar to upload and process documents.")

                # --- Display Summary and Evaluation (Outside the form) ---
                if st.session_state.current_summary and st.session_state.summarized_source:
                     st.markdown("---")
                     st.subheader(f"📄 Summary Result for: `{st.session_state.summarized_source}`")
                     st.markdown(st.session_state.current_summary)

                     # --- Evaluation Section ---
                     st.markdown("---")
                     st.subheader("📊 Evaluate Summary Quality (Optional)")
                     with st.form("evaluation_form"):
                         reference_summary = st.text_area(
                             "Paste reference summary here:", height=150, key="ref_summary_input",
                             placeholder="Provide a 'gold standard' summary for comparison."
                         )
                         evaluate_button = st.form_submit_button("📈 Calculate ROUGE Score")

                     if evaluate_button and reference_summary:
                         try:
                             with st.spinner("Calculating ROUGE scores..."):
                                 scores = summarizer.evaluate(
                                     generated_summaries=[st.session_state.current_summary],
                                     reference_summaries=[reference_summary]
                                 )
                             st.markdown("##### ROUGE Score Results:")
                             if scores and 'average' in scores:
                                 st.json(scores['average'])
                             else:
                                 st.warning("Evaluation did not return expected 'average' scores.")
                         except Exception as e:
                             st.error(f"ROUGE calculation failed: {e}", icon="🔥")
                             logger.exception("Error during ROUGE score calculation.")
                     elif evaluate_button and not reference_summary:
                         st.warning("Please provide a reference summary to calculate scores.")

            else:
                st.warning("Core services (VectorDB, Summarizer) not available. Check initialization.")

    else:
        # Initial state - guide user
        st.info("👋 **Welcome!** Please configure the application using the sidebar:", icon="👈")
        st.markdown("""
            **Get Started:**
            1.  **Select a language model** in the sidebar.
            2.  Click **'Initialize Services'** (or 'Download & Initialize').
            3.  **Upload** your documents.
            4.  Click **'Process Uploaded Documents'**.

            *Q&A and Summarization tabs will become active after processing.*
        """)


if __name__ == "__main__":
    main()