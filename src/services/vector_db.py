from typing import List, Dict, Any, Optional, Set, Union
import uuid
from src.core.core import DocumentChunk, count_tokens
from typing import List, Dict, Any, Optional
import uuid
from chromadb import Documents, EmbeddingFunction, Client  # Updated usage based on ChromaDB docs.
from sentence_transformers import SentenceTransformer

# --- NomicEmbedder: a wrapper for SentenceTransformer embedding ---
class NomicEmbedder(EmbeddingFunction):
    def __init__(self, model_name: str = 'nomic-ai/nomic-embed-text-v1'):
        """
        Initializes the NomicEmbedder with a specified SentenceTransformer model.

        Args:
            model_name (str): The name of the Sentence Transformer model to load.
                              Defaults to 'nomic-ai/nomic-embed-text-v1'.
        """
        try:
            # Load the specified Sentence Transformer model.
            # trust_remote_code=True is necessary for some models like nomic-embed-text.
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            self.model.eval()  # Put the model in evaluation mode.
            print(f"NomicEmbedder initialized with model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformer model '{model_name}': {e}")

    def __call__(self, texts: Documents) -> List[List[float]]:
        """
        Generates embeddings for a list of text documents.

        Args:
            texts (Documents): A list of strings (documents) to embed.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        if not isinstance(texts, list):
            raise TypeError("Input 'texts' must be a list of strings.")
            
        # Ensure all elements are strings, handling potential None values
        processed_texts = [str(text) if text is not None else "" for text in texts]

        try:
            embeddings = self.model.encode(processed_texts, convert_to_tensor=False)
            # Convert numpy array (if returned) to a list of lists.
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Error encoding texts: {e}")


# --- VectorStore: Manages the document chunks and querying ---
class VectorStore:
    def __init__(self, collection_name: str = "docs"):
        """
        Initializes the VectorStore.

        Args:
            collection_name (str): The name of the collection to use in ChromaDB.
                                   Defaults to "docs".
        """
        try:
            # Use ephemeral client by default (in-memory).
            # For persistence, specify a path: Client(Settings(persist_directory="path/to/db"))
            self.client = Client() # Consider adding persistence: Settings(persist_directory="./chroma_db")
            print("ChromaDB client initialized.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB Client: {e}")

        self.embedder = NomicEmbedder() # Uses the default 'nomic-ai/nomic-embed-text-v1'
        self.collection_name = collection_name

        try:
            # *** FIX: Use get_or_create_collection ***
            # This will create the collection if it doesn't exist, or load it if it does.
            self.collection: Collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedder # Pass the embedder instance
            )
            print(f"ChromaDB collection '{self.collection_name}' loaded/created successfully.")
        except Exception as e:
            # Catching potential specific ChromaDB errors might be better if needed
            raise RuntimeError(f"Failed to get or create collection '{self.collection_name}': {e}")

    def add_chunks(self, chunks: List[DocumentChunk]):
        """
        Adds a list of DocumentChunk objects to the collection.
        Checks if sources already exist and prints a message, but still adds the new chunks.

        Args:
            chunks (List[DocumentChunk]): The list of document chunks to add.
        """
        if not chunks:
            print("Warning: No chunks provided to add.")
            return

        # *** Refinement: Check for existing sources and notify ***
        unique_sources_in_input = {chunk.source for chunk in chunks if chunk.source}
        for source_name in unique_sources_in_input:
            try:
                existing = self.collection.get(where={"source": source_name}, limit=1, include=[])
                if existing and existing.get('ids') and len(existing['ids']) > 0:
                    print(f"Info: Source '{source_name}' already has entries in the collection. Adding new chunks.")
            except Exception as e:
                print(f"Warning: Could not check for existing source '{source_name}'. Error: {e}")


        # Prepare document texts, metadata, and unique IDs.
        documents = [chunk.text for chunk in chunks]
        metadatas = [{
            "source": chunk.source,
            "page": chunk.page, # Ensure page is serializable (int/str)
            "chunk_number": chunk.chunk_number # Ensure chunk_number is serializable
        } for chunk in chunks]
        
        # Generate a unique ID that combines metadata and UUID.
        # Using UUID ensures uniqueness even if content is identical.
        # If you want updates based on content match, ID needs to be deterministic
        # (e.g., hash of content or source+page+chunk_number).
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
            print(f"Successfully added {len(chunks)} chunks to collection '{self.collection_name}'.")
        except Exception as e:
            # More specific error catching (e.g., DuplicateIDError if IDs weren't unique)
            # might be useful depending on ID strategy.
            raise RuntimeError(f"Error adding chunks to collection '{self.collection_name}': {e}")

    def list_all_sources(self) -> List[str]:
        """
        Retrieves a list of all unique source names stored in the collection's metadata.

        Returns:
            List[str]: A sorted list of unique source names.
        """
        try:
            # Get all items, only including metadata
            results = self.collection.get(include=['metadatas'])
            
            if not results or not results.get('metadatas'):
                return []

            # Extract the 'source' field from each metadata dictionary
            sources: Set[str] = set()
            for meta in results['metadatas']:
                if meta and 'source' in meta:
                    sources.add(meta['source'])
                else:
                     print("Warning: Found metadata entry without a 'source' field.")


            return sorted(list(sources))
        except Exception as e:
            print(f"Error retrieving sources from collection '{self.collection_name}': {e}")
            return [] # Return empty list on error

    def get_chunks_by_source(
        self,
        source_name: str,
        return_text: bool = False
    ) -> Union[List[DocumentChunk], str]:
        """
        Retrieves all document chunks associated with a specific source name.

        Args:
            source_name (str): The name of the source to filter by.
            return_text (bool): If True, concatenates the text of all found chunks
                                into a single string separated by newlines.
                                If False (default), returns a list of DocumentChunk objects.

        Returns:
            Union[List[DocumentChunk], str]: Either a list of DocumentChunk objects or
                                             a single string containing the combined text,
                                             depending on the `return_text` parameter.
                                             Returns an empty list or empty string if no
                                             chunks are found for the source or an error occurs.
        """
        try:
            # Query the collection for items matching the source name
            results = self.collection.get(
                where={"source": source_name},
                include=['documents', 'metadatas'] # We need text and metadata
            )

            if not results or not results.get('ids') or len(results['ids']) == 0:
                 print(f"No chunks found for source: {source_name}")
                 return "" if return_text else []

            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])

            if not documents or not metadatas or len(documents) != len(metadatas):
                print(f"Warning: Mismatch between documents and metadatas retrieved for source: {source_name}")
                return "" if return_text else []


            if return_text:
                # Combine all document texts into a single string
                return "\n\n".join(documents) # Use double newline as separator
            else:
                # Reconstruct DocumentChunk objects
                chunks = []
                for doc_text, meta in zip(documents, metadatas):
                    try:
                         # Ensure required keys exist in metadata
                        if 'source' not in meta or 'page' not in meta or 'chunk_number' not in meta:
                            print(f"Warning: Skipping chunk due to missing metadata keys: {meta}")
                            continue
                            
                        chunk = DocumentChunk(
                            source=meta['source'],
                            page=meta['page'],
                            chunk_number=meta['chunk_number'],
                            text=doc_text,
                            tokens=count_tokens(doc_text) # Recalculate tokens
                        )
                        chunks.append(chunk)
                    except KeyError as e:
                        print(f"Warning: Missing expected key in metadata {meta} for source {source_name}: {e}")
                    except Exception as e:
                        print(f"Error reconstructing DocumentChunk for source {source_name}: {e}")
                return chunks

        except Exception as e:
            print(f"Error retrieving chunks for source '{source_name}': {e}")
            return "" if return_text else [] # Return empty on error

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
          - Re-ranking: Most similar chunk has no penalty, subsequent chunks get an increasing penalty based on distance.

        Args:
            text (str): The query text.
            max_chunks (int): The absolute maximum number of chunks to return. Defaults to 5.
            per_source (int): Maximum number of responses per document source after initial retrieval and grouping. Defaults to 5.
            metadata_filter (Optional[Dict[str, Any]]): Optional dict to filter results by metadata using ChromaDB's 'where' clause.
            rerank (bool): Whether to apply custom re-ranking logic based on source diversity and distance penalty. Defaults to True.

        Returns:
            List[DocumentChunk]: List of relevant DocumentChunk objects.
        """
        try:
            # Retrieve a larger number of candidates initially to allow for effective re-ranking and grouping.
            # Let's fetch enough to potentially cover several sources `per_source` limit.
            n_results_to_fetch = max(100, max_chunks * 5, per_source * 10) # Heuristic for fetching enough candidates

            query_params = {
                "query_texts": [text],
                "n_results": n_results_to_fetch,
                "include": ["documents", "metadatas", "distances"] # Ensure distances are included
            }
            if metadata_filter:
                # Combine potential external filter with internal logic if needed,
                # for now, just pass it through.
                query_params["where"] = metadata_filter

            results = self.collection.query(**query_params)
            # print("\nRaw Query Results:", results, "\n") # Optional: for debugging

        except Exception as e:
            raise RuntimeError(f"ChromaDB query failed: {e}")

        if not results or not results.get("documents") or not results["documents"][0]:
            print("Query returned no results.")
            return []

        # Process results assuming a single query text – take the first list element.
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        # Distances are crucial for similarity calculation and re-ranking
        dists = results.get("distances", [[]])[0] # Default to empty list if not present

        if len(docs) != len(metas) or len(docs) != len(dists):
             print(f"Warning: Mismatch in lengths of query results: "
                   f"{len(docs)} docs, {len(metas)} metas, {len(dists)} dists. Truncating.")
             min_len = min(len(docs), len(metas), len(dists))
             docs = docs[:min_len]
             metas = metas[:min_len]
             dists = dists[:min_len]


        # Build items with calculated similarity (lower distance is better)
        items = []
        for i, (doc_text, metadata, distance) in enumerate(zip(docs, metas, dists)):
             # Handle potential None distance (though less likely with query include)
             if distance is None:
                 similarity = 0.0 # Or some other default?
                 print(f"Warning: Found None distance for item index {i}")
             else:
                 # Simple inverse distance similarity, capped at 1.0 (avoid division by zero)
                 # Add a small epsilon to prevent division by zero if distance is exactly 0.
                 similarity = 1.0 / (1.0 + distance + 1e-9)

             item = {
                 "doc_text": doc_text,
                 "metadata": metadata,
                 "distance": distance,
                 "similarity": similarity # Raw similarity based on distance
             }
             items.append(item)

        # Group results by the "source" field in metadata.
        grouped: Dict[str, List[Dict]] = {}
        for item in items:
            source = item["metadata"].get("source", "unknown_source") # Handle missing source
            grouped.setdefault(source, []).append(item)

        # Apply re-ranking logic if enabled
        processed_items = []
        if rerank:
            penalty = 0.05  # Penalty offset per rank position *within the same source*
            for source, group_items in grouped.items():
                # Sort items within each source group by original similarity (descending)
                group_items.sort(key=lambda x: x["similarity"], reverse=True)
                # Apply penalty and limit by per_source
                for rank, item in enumerate(group_items[:per_source]):
                    # Penalty increases for lower-ranked items within the same source
                    adjusted_similarity = item["similarity"] - (rank * penalty)
                    item["adjusted_similarity"] = max(0, adjusted_similarity) # Ensure similarity doesn't go below 0
                    processed_items.append(item)
            # Globally sort all selected items based on adjusted similarity
            processed_items.sort(key=lambda x: x["adjusted_similarity"], reverse=True)
        else:
            # If not re-ranking, just use all retrieved items and sort by original similarity
            processed_items = items
            processed_items.sort(key=lambda x: x["similarity"], reverse=True)


        # Convert the top items back into DocumentChunk objects, respecting max_chunks limit
        final_chunks = []
        # Limit the number of items *before* creating DocumentChunk objects
        items_to_convert = processed_items[:max_chunks] if max_chunks > 0 else processed_items

        for item in items_to_convert:
            try:
                # Ensure metadata has the required fields
                if not all(k in item["metadata"] for k in ["source", "page", "chunk_number"]):
                     print(f"Warning: Skipping item due to missing metadata keys: {item['metadata']}")
                     continue

                chunk = DocumentChunk(
                    source=item["metadata"]["source"],
                    page=item["metadata"]["page"],
                    chunk_number=item["metadata"]["chunk_number"],
                    text=item["doc_text"],
                    tokens=count_tokens(item["doc_text"]) # Recalculate tokens
                )
                final_chunks.append(chunk)
                
                # Optional: Print info about the selected chunk and its (adjusted) similarity
                # sim_value = item.get("adjusted_similarity", item["similarity"])
                # print(f"  - Source: {chunk.source}, Page: {chunk.page}, Chunk: {chunk.chunk_number}, Sim: {sim_value:.4f}")

            except KeyError as e:
                print(f"Error: Missing expected metadata key in result item {item['metadata']}: {e}")
            except Exception as e:
                print(f"Error processing a result item into DocumentChunk: {e}")

        print(f"\nQuery processing complete. Returning {len(final_chunks)} chunks (max_chunks={max_chunks}).")
        return final_chunks