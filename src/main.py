import os
import tempfile
from pathlib import Path
import requests
import streamlit as st

try:
    from src.utils.file_processor import process_file
    from src.utils.chunker import Chunker
    from src.utils.utils import AVAILABLE_MODELS
    from src.services.vector_db import VectorStore
    from src.services.rag import RAGSystem
except ImportError:
    st.error("Failed to import necessary modules. Make sure your project structure (src/utils, src/services) is correct and accessible.")
    # Attempt relative imports as a fallback (might work if run differently)
    try:
        from utils.file_processor import process_file
        from utils.chunker import Chunker
        from utils.utils import AVAILABLE_MODELS
        from services.vector_db import VectorStore
        from services.rag import RAGSystem
    except ImportError:
         st.stop() # Stop execution if imports fail

# --- Configuration ---
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True) # Ensure models directory exists

def download_model(model_name: str, url: str, dest_path: Path):
    """Downloads a model file with progress indication."""
    st.info(f"Downloading {model_name}...")
    st.write(f"Source: {url}")
    st.write(f"Destination: {dest_path}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Get total size from headers, default to 0 if not found
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # Process in 1MB chunks
        downloaded_size = 0

        # --- Progress Display Elements ---
        # 1. Create a progress bar element
        progress_bar = st.progress(0.0)
        # 2. Create a text element placeholder using st.empty()
        status_text = st.empty()
        # ---------------------------------

        # Set initial status text
        if total_size > 0:
             status_text.text(f"Starting download... (Total size: {total_size / (1024*1024):.2f} MB)")
        else:
             status_text.text("Starting download... (Total size unknown)")


        with open(dest_path, 'wb') as f:
            # Iterate over the download stream chunk by chunk
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # Calculate progress percentage
                    if total_size > 0:
                        # Ensure progress doesn't exceed 1.0 due to potential inaccuracies
                        progress = min(float(downloaded_size) / total_size, 1.0)
                    else:
                        # If size is unknown, we can't show accurate percentage.
                        # Keep the bar at 0 or indicate uncertainty.
                        progress = 0.0

                    # --- Update Progress Display ---
                    # 1. Update the progress bar value
                    progress_bar.progress(progress)
                    # 2. Update the text placeholder with current download size vs total
                    if total_size > 0:
                         status_text.text(f"Downloaded {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB")
                    else:
                         status_text.text(f"Downloaded {downloaded_size / (1024*1024):.2f} MB / Unknown MB")
                    # -----------------------------

        # --- Final Update on Success ---
        progress_bar.progress(1.0) # Ensure bar visually completes
        status_text.success(f"✅ Model '{model_name}' downloaded successfully!")
        # -------------------------------
        return True

    except requests.exceptions.RequestException as e:
        st.error(f"Download failed: {e}")
        # Clear progress elements on failure if they exist
        if 'progress_bar' in locals(): progress_bar.empty()
        if 'status_text' in locals(): status_text.empty()
        if dest_path.exists():
            dest_path.unlink() # Remove partial file on error
        return False
    except IOError as e:
        st.error(f"Failed to write model file: {e}")
        if 'progress_bar' in locals(): progress_bar.empty()
        if 'status_text' in locals(): status_text.empty()
        if dest_path.exists():
            dest_path.unlink()
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        if 'progress_bar' in locals(): progress_bar.empty()
        if 'status_text' in locals(): status_text.empty()
        if dest_path.exists():
            dest_path.unlink()
        return False

def process_documents(uploaded_files, llm_model_path: str) -> RAGSystem | None:
    """
    Processes uploaded documents, chunks them, adds to vector store,
    and initializes the RAG system. Returns RAG system or None on failure.
    """
    if not llm_model_path or not Path(llm_model_path).is_file():
        st.error(f"Invalid LLM model path provided: {llm_model_path}")
        return None
    if not uploaded_files:
        st.warning("No files uploaded to process.")
        return None

    st.info("Initializing services...")
    try:
        vector_db = VectorStore() # Consider adding error handling here if init can fail
        chunker = Chunker()       # Consider adding error handling here if init can fail
        rag = RAGSystem(vector_db, llm_model_path) # Pass the actual path
    except Exception as e:
        st.error(f"Failed to initialize RAG components: {e}")
        return None

    st.info("Starting document processing...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    num_files = len(uploaded_files)
    processed_files = 0
    successful_files = 0
    files_with_errors = []

    for file in uploaded_files:
        file_name = file.name
        status_text.text(f"Processing file: {file_name}...")
        processed_files += 1
        progress_percent = int((processed_files / num_files) * 100)

        try:
            suffix = Path(file_name).suffix.lower()
            supported_types = {'.docx', '.pdf', '.csv', '.xlsx', '.xls', '.xlsm'}

            if suffix not in supported_types:
                st.warning(f"Skipping unsupported file type: {file_name} ({suffix})")
                progress_bar.progress(progress_percent) # Update progress even if skipped
                continue

            # Save the uploaded file to a temporary file safely
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                try:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                except IOError as e:
                    st.error(f"Error writing temporary file for {file_name}: {e}")
                    files_with_errors.append(file_name)
                    progress_bar.progress(progress_percent)
                    continue # Skip to next file

            # Process and chunk the document (ensure tmp_path is valid)
            try:
                pages = process_file(tmp_path) # Assuming process_file handles its own errors/returns []
                if not pages:
                    st.warning(f"No content extracted from {file_name}.")
                    # Optionally add to errors list or just continue
                else:
                    chunks = chunker.chunk_document(file_name, pages)
                    st.info(f"Created {len(chunks)} chunks for {file_name}.")
                    if chunks:
                        vector_db.add_chunks(chunks) # Assuming add_chunks handles potential errors
                    else:
                         st.warning(f"No chunks generated for {file_name}, possibly empty or unprocessable.")
                successful_files += 1
            except Exception as e:
                st.error(f"Error processing content of {file_name}: {e}")
                files_with_errors.append(file_name)
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(tmp_path)
                except OSError as e:
                    st.warning(f"Could not delete temporary file {tmp_path}: {e}")

        except Exception as e:
            # Catch broader errors during file handling
            st.error(f"Unexpected error handling file {file_name}: {e}")
            files_with_errors.append(file_name)
        finally:
             # Update progress bar regardless of success/failure for this file
            progress_bar.progress(progress_percent)

    # Final status update
    status_text.empty() # Clear the individual file status
    progress_bar.progress(100) # Ensure it reaches 100%

    if successful_files > 0:
        st.success(f"Successfully processed {successful_files} out of {num_files} documents.")
    if files_with_errors:
        st.error(f"Failed to process {len(files_with_errors)} documents: {', '.join(files_with_errors)}")
        return None # Indicate partial failure maybe? Or return rag anyway? Depends on desired behavior.
    if successful_files == 0 and not files_with_errors:
         st.warning("No documents were successfully processed (perhaps none were supported or content extraction failed).")
         return None

    st.balloons() # Fun indicator of completion
    return rag

# --- Main Application ---

def main():
    st.set_page_config(page_title="Dr. X Research Assistant", layout="wide")
    st.title("🔬 Dr. X Research Assistant")
    st.markdown("Upload research papers, data files, or other documents, select an LLM, and ask questions about the content.")

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")

        # 1. Model Selection and Download
        st.subheader("1. Select LLM Model")
        selected_model_name = st.selectbox(
            "Choose a model:",
            options=list(AVAILABLE_MODELS.keys()),
            help="Select the Large Language Model to use for analysis."
        )

        model_info = AVAILABLE_MODELS[selected_model_name]
        model_filename = model_info["filename"]
        model_url = model_info["url"]
        model_path = MODELS_DIR / model_filename

        # Check if model exists locally
        model_available_locally = model_path.is_file()

        if model_available_locally:
            st.success(f"✅ Model '{selected_model_name}' is available locally.")
            st.session_state.selected_model_path = str(model_path) # Store the confirmed path
        else:
            st.warning(f"Model '{selected_model_name}' not found locally.")
            st.markdown(f"**Required file:** `{model_filename}` in `{MODELS_DIR}`")
            if st.button(f"Download {selected_model_name} ({model_path.name})"):
                with st.spinner(f"Downloading {model_filename}... This may take a while."):
                    download_success = download_model(selected_model_name, model_url, model_path)
                    if download_success:
                        st.session_state.selected_model_path = str(model_path)
                        st.rerun() # Rerun to update UI reflecting downloaded model
                    else:
                         # Error handled in download_model, clear session state if needed
                         if 'selected_model_path' in st.session_state:
                             del st.session_state.selected_model_path

        st.markdown("---")

        # 2. Document Upload
        st.subheader("2. Upload Documents")
        uploaded_files = st.file_uploader(
            "Select one or more documents:",
            type=["docx", "pdf", "csv", "xlsx", "xls", "xlsm"],
            accept_multiple_files=True,
            help="Upload documents (Word, PDF, Excel, CSV) for the RAG system."
        )

        st.markdown("---")

        # 3. Processing Trigger
        st.subheader("3. Process Data")
        # Disable button if no model is confirmed available or no files uploaded
        process_button_disabled = 'selected_model_path' not in st.session_state or not uploaded_files
        button_tooltip = ""
        if 'selected_model_path' not in st.session_state:
            button_tooltip += "Select and ensure model is available locally. "
        if not uploaded_files:
             button_tooltip += "Upload at least one document."

        if st.button("Process Uploaded Documents", disabled=process_button_disabled, help=button_tooltip or None):
            if 'selected_model_path' in st.session_state and uploaded_files:
                model_to_use = st.session_state.selected_model_path
                with st.spinner("⚙️ Processing documents... Indexing content..."):
                    rag_system = process_documents(uploaded_files, model_to_use)
                    if rag_system:
                        st.session_state.rag = rag_system # Store the initialized RAG system
                        st.success("✅ Processing complete! Ready for Q&A.")
                    else:
                        st.error("❌ Document processing failed. Please check errors above.")
                        # Clear potentially partially initialized rag state
                        if 'rag' in st.session_state:
                            del st.session_state.rag
            else:
                 # This case should be prevented by the disabled button, but good practice
                 st.warning("Please ensure a model is selected/downloaded and documents are uploaded.")


    # --- Main Area for Q&A ---
    st.markdown("---") # Divider

    if "rag" in st.session_state and st.session_state.rag is not None:
        st.header("❓ Ask Questions")
        st.info("The RAG system is ready. Ask questions based on the content of your uploaded documents.")

        question = st.text_input("Enter your question:", key="qa_input")

        if st.button("Get Answer", key="get_answer_button"):
            if question:
                with st.spinner("🧠 Thinking..."):
                    try:
                        answer = st.session_state.rag.generate_answer(question)
                        st.markdown("#### Answer:")
                        st.markdown(answer) # Use markdown for potential formatting in the answer
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
                        st.exception(e) # Show detailed traceback for debugging if needed
            else:
                st.warning("Please enter a question.")
    else:
         st.info("Upload documents and click 'Process Uploaded Documents' in the sidebar to enable the Q&A section.")


if __name__ == "__main__":
    # Basic check for dependencies mentioned in the original code
    # You might want more robust checks depending on your environment
    try:
        import llama_cpp
        import sentence_transformers
        # Add other crucial imports your RAG system depends on
    except ImportError as e:
         st.error(f"Missing essential library: {e.name}. Please install it (e.g., `pip install {e.name}`). Your RAG system might not work.")
         # Consider stopping execution if llama_cpp or sentence-transformers are absolutely essential upfront
         # st.stop()

    main()