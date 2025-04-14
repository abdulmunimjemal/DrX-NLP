from llama_cpp import Llama
from src.core.core import DocumentChunk, track_performance
from typing import Any, List
import os

class RAGSystem:
    """
    Retrieval-Augmented Generation (RAG) system that retrieves document chunks from a
    vector store and generates answers using a Llama-based language model.
    """
    def __init__(self, vector_store: Any, model_path: str):
        """
        Initializes the RAG system with the given vector store and Llama model.

        Args:
            vector_store: An instance of a vector store that has a query method.
            model_path (str): The file path to the Llama model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Llama model not found at {model_path}. Download a GGUF model first!")
        
        self.vector_store = vector_store
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4
        )
        self.history: List[str] = []
    
    @track_performance
    def generate_answer(self, question: str) -> str:
        """
        Retrieves relevant document chunks and generates an answer for the given question.

        Args:
            question (str): The input question.

        Returns:
            str: The generated answer from the language model.
        """
        # Retrieve relevant document chunks from the vector store.
        try:
            chunks = self.vector_store.query(question)
        except Exception as e:
            raise RuntimeError(f"Error querying vector store: {e}")
        
        # Build context from chunks; if none are found, include a default message.
        if chunks:
            context = "\n\n".join(chunk.text for chunk in chunks)
        else:
            context = "No relevant context found."
        
        # Use the last three conversation entries as history if available.
        history_context = "\n".join(self.history[-3:]) if self.history else ""

        # Construct the prompt using context, conversation history, and the question.
        prompt = (
            f"Context:\n{context}\n\n"
            f"Previous conversation:\n{history_context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        
        # Generate a response using the Llama model.
        try:
            response = self.llm(
                prompt=prompt,
                max_tokens=512,
                temperature=0.2,
                stop=["\n\n"]
            )
        except Exception as e:
            raise RuntimeError(f"Error generating response from LLM: {e}")
        
        # Safely extract the answer from the model's response.
        try:
            answer_text = response.get('choices', [{}])[0].get('text', '').strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response format from LLM: {e}")
        
        # Update the conversation history.
        self.history.append(f"Q: {question}")
        self.history.append(f"A: {answer_text}")
        
        return answer_text

# --- Example Usage ---
if __name__ == "__main__":
    # Dummy vector store with a minimal query implementation for testing.
    class DummyVectorStore:
        def query(self, text: str):
            # Returns a list of dummy DocumentChunk-like objects for test purposes.
            class DummyChunk:
                def __init__(self, text):
                    self.text = text
            return [DummyChunk("This is a sample context for testing purposes.")]
    
    dummy_vector_store = DummyVectorStore()
    # Replace with the actual path to your Llama model file.
    model_path = "path/to/your/llama/model.bin"
    
    # Initialize the RAG system.
    rag_system = RAGSystem(vector_store=dummy_vector_store, model_path=model_path)
    
    # Example question.
    question = "What is the purpose of this system?"
    
    # Generate and print the answer.
    try:
        answer = rag_system.generate_answer(question)
        print("Generated Answer:")
        print(answer)
    except RuntimeError as error:
        print(f"An error occurred: {error}")
