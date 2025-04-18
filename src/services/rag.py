from src.core.core import track_performance, logger
from typing import Any, List
import os

class RAGSystem:
    """
    Retrieval-Augmented Generation (RAG) system that retrieves document chunks from a
    vector store and generates answers using a Llama-based language model.
    """
    def __init__(self, vector_store: Any, llm):
        """
        Initializes the RAG system with the given vector store and Llama model.

        Args:
            vector_store: An instance of a vector store that has a query method.
            model_path (str): The file path to the Llama model.
        """
        
        self.vector_store = vector_store
        self.llm = llm
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
            logger.info(f"Error querying vector store: {e}")
            raise RuntimeError(f"Error querying vector store: {e}")
        
        # Build context from chunks; if none are found, include a default message.
        if chunks:
            chunks = chunks[:4]
            context = "\n\n".join(chunk.text for chunk in chunks)
        else:
            context = "No relevant context found."
        
        # Use the last three conversation entries as history if available.
        history_context = "\n".join(self.history[-3:]) if self.history else ""

        # Construct the prompt using context, conversation history, and the question.
        prompt = (
            f"Answer the question based on the provided context and previous conversation.\n\n"
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
