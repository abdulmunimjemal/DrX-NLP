from src.core.core import DocumentChunk, count_tokens
from typing import List, Dict, Any, Optional
import uuid

# Import ChromaDB components and SentenceTransformer for embeddings.
from chromadb import Documents, EmbeddingFunction, Client  # Updated usage based on ChromaDB docs.
from sentence_transformers import SentenceTransformer
import tiktoken  # Tokenizer, as used in your count_tokens function.


# --- NomicEmbedder: a wrapper for SentenceTransformer embedding ---
class NomicEmbedder(EmbeddingFunction):
    def __init__(self):
        try:
            # According to SentenceTransformer documentation, this loads the model.
            self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
            self.model.eval()  # Put the model in evaluation mode.
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformer: {e}")

    def __call__(self, texts: Documents) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts)
            # Convert numpy array to a list of lists.
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Error encoding texts: {e}")

# --- VectorStore: Manages the document chunks and querying ---
class VectorStore:
    def __init__(self):
        try:
            self.client = Client()  # Instantiate the ChromaDB client.
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Client: {e}")
        
        self.embedder = NomicEmbedder()
        try:
            # Create (or reuse, as appropriate) a collection named "docs"
            self.collection = self.client.create_collection(
                name="docs",
                embedding_function=self.embedder
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create a collection in Client: {e}")

    def add_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            print("Warning: No chunks to add.")
            return
        
        # Prepare document texts, metadata, and unique IDs.
        documents = [chunk.text for chunk in chunks]
        metadatas = [{
            "source": chunk.source,
            "page": chunk.page,
            "chunk_number": chunk.chunk_number
        } for chunk in chunks]
        # Generate a unique ID that combines metadata and UUID.
        ids = [
            f"{chunk.source}-{chunk.page}-{chunk.chunk_number}-{uuid.uuid4()}"
            for chunk in chunks
        ]
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            raise RuntimeError(f"Error adding chunks to collection: {e}")

    def query(
        self,
        text: str,
        max_chunks: int = 5,
        per_source: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[DocumentChunk]:
        """
        Query the vector store for document chunks similar to the input text.

        This function supports:
          - Filtering by metadata (e.g., a specific "source").
          - Grouping results by "source", then selecting up to `per_source` items per source.
          - Re-ranking: Most similar chunk has no penalty, subsequent chunks get an increasing penalty.
        
        :param text: The query text.
        :param per_source: Maximum number of responses per document source.
        :param metadata_filter: Optional dict to filter results by metadata.
        :param rerank: Whether to apply custom re-ranking logic.
        :return: List of DocumentChunk objects.
        """
        try:
            # Retrieve many candidates to allow for re-ranking and grouping.
            n_results = 100
            if max_chunks > n_results:
                n_results = max_chunks

            query_params = {
                "query_texts": [text],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }
            if metadata_filter:
                query_params["where"] = metadata_filter

            results = self.collection.query(**query_params)
            print("\nRaw Query Results:", results, "\n")
        except Exception as e:
            raise RuntimeError(f"Query failed: {e}")

        if not results or not results.get("documents"):
            return []

        # Process results assuming single query – take first list.
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        # `distances` is assumed to be a list of numbers (lower distance means higher similarity).
        dists = results.get("distances", [[None] * len(docs)])[0]

        # Build items with computed similarity (using an example conversion).
        items = []
        for i, (doc_text, metadata) in enumerate(zip(docs, metas)):
            distance = dists[i] if dists and dists[i] is not None else None
            similarity = 1.0 if distance is None else 1 / (1 + distance)
            item = {
                "doc_text": doc_text,
                "metadata": metadata,
                "distance": distance,
                "similarity": similarity
            }
            items.append(item)

        # Group results by the "source" field in metadata.
        grouped = {}
        for item in items:
            source = item["metadata"].get("source", "unknown")
            grouped.setdefault(source, []).append(item)

        # Apply re-ranking per source: allow only up to `per_source` items and add a penalty.
        reranked_items = []
        penalty = 0.05  # Penalty offset per rank position within the same source.
        for source, group_items in grouped.items():
            # Sort items for each source by similarity (higher is better).
            group_items.sort(key=lambda x: x["similarity"], reverse=True)
            for rank, item in enumerate(group_items[:per_source]):
                adjusted_similarity = item["similarity"] - (rank * penalty)
                item["adjusted_similarity"] = adjusted_similarity
                reranked_items.append(item)

        # Globally sort all items based on adjusted similarity (if re-ranking is enabled).
        if rerank:
            reranked_items.sort(key=lambda x: x["adjusted_similarity"], reverse=True)
        else:
            reranked_items.sort(key=lambda x: x["similarity"], reverse=True)

        # Convert items back into DocumentChunk objects.
        chunks = []
        for item in reranked_items:
            try:
                chunk = DocumentChunk(
                    source=item["metadata"]["source"],
                    page=item["metadata"]["page"],
                    chunk_number=item["metadata"]["chunk_number"],
                    text=item["doc_text"],
                    tokens=count_tokens(item["doc_text"])
                )
                chunks.append(chunk)
            except KeyError as e:
                print(f"Missing metadata key in result: {e}")
            except Exception as e:
                print(f"Error processing a result item: {e}")

        print(f"Final re-ranked query returned {len(chunks)} chunks.")
        for chunk in chunks:
            print(f"Chunk source: {chunk.source}, page: {chunk.page}, "
                  f"chunk_number: {chunk.chunk_number}, tokens: {chunk.tokens}")
        return chunks[:max_chunks] if max_chunks > 0 else chunks
