from typing import List, Any
from chromadb import Documents, EmbeddingFunction, ChromaClient
from sentence_transformers import SentenceTransformer
from src.core.core import DocumentChunk 
from src.core.core import count_tokens 

class NomicEmbedder(EmbeddingFunction):
    """
    A wrapper around a SentenceTransformer model to generate embeddings.
    """
    def __init__(self):
        try:
            self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1')
            # Optionally set the model to evaluation mode if it supports it:
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformer: {e}")

    def __call__(self, texts: Documents) -> List[List[float]]:
        """
        Encode a list of texts into embeddings.
        
        :param texts: A list of document texts.
        :return: List of embeddings (each embedding is a list of floats).
        """
        try:
            # encode returns a numpy array; converting it to a list of lists.
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Error in encoding texts: {e}")


class VectorStore:
    """
    A simple vector store that uses ChromaClient for managing a document collection.
    """
    def __init__(self):
        try:
            self.client = ChromaClient()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaClient: {e}")

        self.embedder = NomicEmbedder()
        try:
            self.collection = self.client.create_collection(
                name="docs",
                embedding_function=self.embedder
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create a collection in ChromaClient: {e}")

    def add_chunks(self, chunks: List[DocumentChunk]):
        """
        Add document chunks to the vector store.
        
        :param chunks: A list of DocumentChunk objects.
        """
        if not chunks:
            print("Warning: No chunks to add.")
            return
        
        documents = [chunk.text for chunk in chunks]
        metadatas = [{
            "source": chunk.source,
            "page": chunk.page,
            "chunk_number": chunk.chunk_number
        } for chunk in chunks]
        ids = [str(i) for i in range(len(documents))]
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            raise RuntimeError(f"Error adding chunks to collection: {e}")

    def query(self, text: str, k: int = 5) -> List[DocumentChunk]:
        """
        Query the vector store for document chunks similar to the input text.
        
        :param text: The query text.
        :param k: Number of results to retrieve.
        :return: A list of DocumentChunk objects representing the best matches.
        """
        try:
            results = self.collection.query(
                query_texts=[text],
                n_results=k
            )
        except Exception as e:
            raise RuntimeError(f"Query failed: {e}")
        
        # Depending on ChromaClient's schema, results may be a dict with key "documents" or a list.
        # Here we assume results[0] is an iterable of objects with attributes: document and metadata.
        if not results or not results[0]:
            return []

        chunks = []
        for res in results[0]:
            try:
                chunk = DocumentChunk(
                    source=res.metadata["source"],
                    page=res.metadata["page"],
                    chunk_number=res.metadata["chunk_number"],
                    text=res.document,
                    tokens=count_tokens(res.document)
                )
                chunks.append(chunk)
            except KeyError as e:
                print(f"Missing metadata key in result: {e}")
            except Exception as e:
                print(f"Error processing a result item: {e}")
        
        return chunks


# --- Test Harness ---

if __name__ == "__main__":
    # For testing, we'll simulate a minimal environment.
    # If you don't have an actual ChromaClient or want to test without side effects,
    # you can define dummy classes below.
    try:
        # Test NomicEmbedder with a dummy text list
        embedder = NomicEmbedder()
        sample_texts = ["This is a test.", "Another test sentence."]
        embeddings = embedder(sample_texts)
        print("Embeddings:", embeddings)
    except Exception as e:
        print("NomicEmbedder test failed:", e)

    # Prepare some dummy DocumentChunk objects for testing add_chunks and query
    # You might want to replace DocumentChunk with your actual implementation.
    try:
        # Let's create two dummy DocumentChunk objects:
        dummy_chunks = [
            DocumentChunk(source="dummy.txt", page=1, chunk_number=0, text="Test document chunk one.", tokens=count_tokens("Test document chunk one.")),
            DocumentChunk(source="dummy.txt", page=1, chunk_number=1, text="Test document chunk two.", tokens=count_tokens("Test document chunk two."))
        ]
        
        # Create a VectorStore instance
        store = VectorStore()
        # Add the dummy chunks
        store.add_chunks(dummy_chunks)
        print("Chunks added successfully.")
        
        # Perform a query using a test string. Depending on your actual collection and client,
        # this will return results similar to our dummy data.
        query_results = store.query("Test document", k=2)
        print("Query Results:")
        for chunk in query_results:
            print(f"Source: {chunk.source}, Page: {chunk.page}, Chunk: {chunk.chunk_number}, Tokens: {chunk.tokens}")
            print("Text:", chunk.text)
    except Exception as e:
        print("VectorStore test encountered an error:", e)
