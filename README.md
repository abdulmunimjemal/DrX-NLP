# NLP Project - Dr. X's Research Assistant

## Contents

- [1. Introduction](#1-introduction)
- [2. Project Objective](#2-project-objective)
- [Screenshots](#screenshots)
- [3. Methodology & Workflow](#3-methodology--workflow)
- [4. Technology Stack](#4-technology-stack)
- [5. Project Structure](#5-project-structure)
- [6. Setup and Installation](#6-setup-and-installation)
- [7. Usage](#7-usage)
- [8. Performance Metrics](#8-performance-metrics)
- [9. Key Features & Creativity](#9-key-features--creativity)
- [10. LLM and Embedding Model Information](#10-llm-and-embedding-model-information)


## 1. Introduction

This project addresses the local NLP challenge, focusing on processing the scattered research publications of the vanished Dr. X. The goal is to create an NLP system capable of extracting information from various document formats, making sense of the research through analysis, summarization, and translation, and enabling interactive querying via a Retrieval-Augmented Generation (RAG) system. The project aims to uncover insights into Dr. X's work and potentially shed light on their disappearance using only text-based NLP techniques and local models.

## 2. Project Objective

The primary objective is to build a robust pipeline that can:

1.  **Ingest & Extract:** Read and extract text content from diverse file formats (.docx, .pdf, .csv, .xlsx, .xls, .xlsm), including text within tables.
2.  **Process & Chunk:** Prepare the extracted text by dividing it into smaller, manageable chunks suitable for embedding, while retaining source metadata (file name, page number, chunk number).
3.  **Embed & Index:** Generate vector embeddings for each text chunk using a local embedding model and store them efficiently in a local vector database.
4.  **Query & Answer:** Implement a RAG system allowing users to ask questions about the documents. The system retrieves relevant chunks using a novel re-ranking algorithm and uses a local Large Language Model (LLM) to generate context-aware answers, maintaining conversation history.
5.  **Summarize:** Implement a hierarchical summarization process capable of handling large documents by iteratively chunking and summarizing to produce a final coherent summary within specified length constraints.
6.  **Translate:** Detect the language of the documents and translate non-English/non-Arabic content into English, preprocessing for accuracy and leveraging an LLM for fluency improvements.
7.  **Measure Performance:** Track processing efficiency, specifically "tokens per second," during key NLP tasks (embedding, RAG, translation, summarization).

### Screenshots

**Home Page:**
![image](https://github.com/user-attachments/assets/c0e3da97-ab3d-4742-bd29-9eaec781add1)

**RAG Q&A:**
![image](https://github.com/user-attachments/assets/917f844c-5dc9-4929-ab1c-c5d99ec450cc)

**Summarization:**
![image](https://github.com/user-attachments/assets/04a7403b-f8ae-48b3-b21f-47eb51340caf)

**ROUGE Evaluation:**
![image](https://github.com/user-attachments/assets/cd11bd3f-b599-4612-9a3e-431af6ebf60f)


## 3. Methodology & Workflow

The system follows a multi-stage NLP pipeline orchestrated via a Streamlit web interface:

1.  **Initialization & Configuration (Streamlit UI - `main.py`):**
    * The user selects a local LLM (e.g., Llama 2, Mistral, Gemma) from available options.
    * If the chosen model file (`.gguf`) is not present in the `./models` directory, the application offers to download it.
    * The user uploads Dr. X's publications (various formats).

2.  **Document Processing (`main.py` orchestrates `utils/` and `services/`):**
    * **File Handling (`utils/file_processor.py`):** Each uploaded file is processed based on its extension. Text is extracted from documents, including handling tables by extracting their textual content without preserving visual layout. Returns structured content.
    * **Translation & Refinement (`services/translator.py`):**
        * Text undergoes **preprocessing** to enhance language detection accuracy.
        * The language of the extracted text is detected (`langdetect`).
        * Text identified as neither English nor Arabic is translated to English using a Python translation package (`deep_translator`).
        * The selected local **LLM is used to post-process** the machine translation (`translator.improve_fluency`), aiming for better coherency and fluency while preserving meaning.
    * **Chunking (`utils/chunker.py`):** The processed (and potentially translated) text is divided into smaller chunks using tokenizer logic (e.g., `cl100k_base` via `tiktoken`). Each chunk retains metadata: `source` (filename), `page` number, and a sequential `chunk_number`.
    * **Embedding & Indexing (`services/vector_db.py`):**
        * A `NomicEmbedder` class wraps the `sentence-transformers` library to load the `nomic-ai/nomic-embed-text-v1` model (running locally).
        * This embedder generates vector representations for each text chunk.
        * A `VectorStore` class uses `ChromaDB` (a local vector database) to store the chunks, their embeddings, and associated metadata. A unique ID is generated for each entry.

3.  **Hierarchical Summarization (`services/summarizer.py` - *assumed location*):**
    * To handle potentially large documents or concatenated text that exceeds context limits, a **recursive summarization** strategy is employed:
        1.  The input text is broken down into smaller chunks that fit within the summarization model's context window.
        2.  Each chunk is summarized individually using the selected local LLM.
        3.  The individual summaries are concatenated together.
        4.  If the concatenated summary still exceeds a defined maximum length limit, the process repeats: the concatenated summary itself is chunked and summarized again.
        5.  This continues until the final summary is coherent and respects the length constraints.

4.  **RAG Q&A System (`services/rag.py` & `main.py`):**
    * The user enters a question into the Streamlit interface.
    * **Retrieval (`vector_db.query` and re-ranking logic in `rag.py`):**
        * The question is used to query the ChromaDB vector store.
        * An initial pool of the top N (e.g., 100) most similar document chunks based on vector similarity is retrieved.
        * **Key Innovation: Aging-Based Re-ranking:** To ensure diversity and prevent single documents from dominating the context, a re-ranking algorithm inspired by "Priority Scheduling with Aging" is applied:
            1.  Select the chunk with the highest similarity score from the initial pool. Add it to the final context list.
            2.  Identify the source document of the selected chunk.
            3.  Penalize all other chunks *from the same source document* within the remaining pool by decreasing their similarity score by a fixed amount (e.g., 0.05).
            4.  Re-sort the pool based on the (potentially updated) similarity scores.
            5.  Repeat steps 1-4 until the desired number of top-k chunks for the final context is reached.
        * The final `k` re-ranked chunks are selected to form the context.
    * **Generation (`rag.generate_answer`):**
        * The re-ranked, retrieved chunks are concatenated to form the context.
        * Recent conversation history (last few Q&A pairs) is included in the prompt to maintain context.
        * A prompt is constructed containing the context, history, and the user's question.
        * The selected local LLM (loaded via `llama-cpp-python`) generates an answer based on the prompt.
        * *Performance Tracking:* The `@track_performance` decorator measures the tokens per second during generation.
    * **Response:** The generated answer is displayed to the user in the Streamlit UI.

## 4. Technology Stack

* **Core Language:** Python 3.12
* **Web Framework / UI:** Streamlit
* **LLMs:** Local models run via `llama-cpp-python`. Supported models (downloadable via UI):
    * Llama 2 (`llama-2-7b-chat.Q4_K_M.gguf`)
    * Mistral (`mistral-7b-instruct-v0.1.Q4_K_M.gguf`, `mistral-7b-v0.1.Q4_K_M.gguf`)
    * Gemma (`gemma-2b-it-q4_k_m.gguf`)
    * You can add your own model by modifying the `AVAILABLE_MODELS` dictionary in `utils/utils.py`
* **Embedding Model:** `nomic-ai/nomic-embed-text-v1` (via `sentence-transformers`, running locally)
* **Vector Database:** ChromaDB (local/in-memory)
* **Text Processing & NLP:**
    * `tiktoken`: Tokenization/counting
    * `langdetect`: Language detection
    * `deep_translator` (or similar): Machine translation Python package
    * Standard libraries for file processing (`python-docx`, `pymupdf`, `camelot-py`, `openpyxl`, ...)
* **Containerization:** Docker (Dockerfile provided)
* **Task Runner:** Make (Makefile provided)
* **Dependency Management:** `pip` and `requirements.txt`

## 5. Project Structure

```
.
├── data                     # Sample input documents
│   ├── ... (various .docx, .pdf, .xlsx files)
├── Dockerfile               # Docker configuration
├── Makefile                 # Make tasks (clean, install, run)
├── models                   # Directory for storing downloaded LLM models
│   ├── gemma-2b-it-q4_k_m.gguf  # Example model file (initially empty)
│   └── ... (other model files)
├── pyproject.toml           # Project metadata and dependencies
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── src                      # Source code directory
    ├── core                 # Core utilities (performance tracking, etc.)
    │   ├── core.py
    │   └── __init__.py
    ├── __init__.py
    ├── main.py              # Main Streamlit application entry point
    ├── services             # Business logic modules
    │   ├── __init__.py
    │   ├── rag.py           # RAG system (incl. re-ranking logic)
    │   ├── summarizer.py    # Hierarchical summarization logic (assumed)
    │   ├── translator.py    # Translation logic (preprocessing, translation, LLM refinement)
    │   └── vector_db.py     # Vector database interaction
    └── utils                # Utility functions
        ├── chunker.py       # Text chunking logic
        ├── file_processor.py # File reading and text extraction
        ├── __init__.py
        └── utils.py         # General utilities (e.g., model definitions)
```

## 6. Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/abdulmunimjemal/DrX-NLP.git
    cd DrX-NLP
    ```

2.  **Using Make (Recommended):**
    * This command automatically creates a Python virtual environment (`.venv/`) if it doesn't exist, activates it, and installs all required dependencies from `requirements.txt`.
    ```bash
    make install
    ```

3.  **Manual Setup (Alternative):**
    * Create a Virtual Environment:
        ```bash
        python -m venv .venv
        source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
        ```
    * Install Dependencies:
        ```bash
        pip install -r requirements.txt
        ```

4.  **LLM Models:**
    * Models are *not* included in the repository.
    * They will be downloaded automatically when selected for the first time in the Streamlit UI and stored in the `./models/` directory. Ensure this directory exists and is writable.
    * Alternatively, manually download the desired `.gguf` models (from sources like Hugging Face) and place them in the `./models/` directory, ensuring the filenames match those defined in `src/utils/utils.py` (`AVAILABLE_MODELS`).

5.  **Docker (Optional):**
    * Build the Docker image: `docker build -t dr-x-assistant .`
    * Run the container: `docker run -p 8501:8501 dr-x-assistant` (The app will be accessible at `http://localhost:8501`)

## 7. Usage

1.  **Using Make (Recommended):**
    * This command handles setting the `PYTHONPATH`, activating the virtual environment (and runs `make install` if the environment isn't set up), and starts the Streamlit application.
    ```bash
    make run
    ```

2.  **Manual Start (Alternative):**
    * Ensure your virtual environment is activated (`source .venv/bin/activate` or `.venv\Scripts\activate`).
    * Set the Python path (optional but good practice, especially if running from the root directory):
        ```bash
        export PYTHONPATH=$(pwd) # Linux/macOS
        # set PYTHONPATH=%cd% # Windows Command Prompt
        # $env:PYTHONPATH = (Get-Location).Path # Windows PowerShell
        ```
    * Run the Streamlit Application:
        ```bash
        streamlit run src/main.py
        ```
    * This will open the application in your web browser (usually at `http://localhost:8501`).

3.  **Application Workflow:**
    * In the sidebar, select the desired local LLM. Download if necessary.
    * Upload the research documents using the file uploader.
    * Click the "Process Documents" button. Wait for extraction, translation, chunking, embedding, and indexing to complete.
    * Once processed, use the "Research Query Interface" to ask questions. The RAG system will retrieve relevant information using the re-ranking algorithm and generate an answer.
    * Use the summarization features as implemented in the UI (details may vary).

## 8. Performance Metrics

The system includes mechanisms (`@track_performance` decorator) to measure "tokens per second" for LLM-intensive operations (RAG generation, summarization, translation refinement). Logging provides detailed insights into processing steps.

## 9. Key Features & Creativity

* **Multi-Format Handling:** Extracts text from diverse formats including tables.
* **Advanced RAG Retrieval:** Implements a novel **"Priority Scheduling with Aging"-inspired re-ranking algorithm** for context retrieval. This ensures diversity in retrieved chunks, preventing single large documents from dominating the context provided to the LLM.
* **Hierarchical Summarization:** Employs a recursive chunking and summarization strategy to effectively summarize very large documents while adhering to length constraints.
* **LLM-Enhanced Translation:** Uses a **preprocessing step for better language detection** and leverages the local LLM to post-process machine translations, improving fluency and coherency beyond standard package output.
* **Local-First Approach:** Operates entirely with local LLMs, embedding models, and vector storage, ensuring data privacy and offline capability.
* **Simplified Setup/Execution:** Utilizes `make` for easy environment setup (`make install`) and application execution (`make run`).


## 10. LLM and Embedding Model Information

* **LLMs Used:** GGUF-formatted models (Llama 2, Mistral, Gemma variants supported) via `llama-cpp-python`. User selects at runtime.
* **Obtaining LLMs:** Downloaded on demand via UI or manually placed in `./models`. Sourced from Hugging Face.
* **Embedding Model:** `nomic-ai/nomic-embed-text-v1`. Loaded locally via `sentence-transformers`. Chosen for performance and offline capability.
