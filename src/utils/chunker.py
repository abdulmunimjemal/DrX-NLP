from typing import List, Tuple
from src.core.core import DocumentChunk, logger
import tiktoken

class Chunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize the Chunker.
        
        :param chunk_size: Maximum number of tokens per chunk.
        :param overlap: Number of overlapping tokens between consecutive chunks.
        :raises ValueError: If chunk_size is not greater than overlap.
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk_document(self, filename: str, pages: List[Tuple[int, str]]) -> List[DocumentChunk]:
        """
        Splits document pages into overlapping token chunks.

        :param filename: Source filename for metadata.
        :param pages: List of tuples in the form (page_number, text).
        :return: List of DocumentChunk objects.
        """
        chunks = []
        chunk_counter = 0
        
        for page in pages:
            text = page.get('text', '')
            page_num = page.get('page_number', 0)
            if not page['text']:
                logger.info(f"Skipping empty page {page_num} in {filename}")
                continue  # Skip empty text
                
            # Encode the entire page's text
            tokens = self.encoder.encode(text)
            start = 0
            
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                # Get token slice and decode back to text
                chunk_tokens = tokens[start:end]
                chunk_text = self.encoder.decode(chunk_tokens)
                
                chunks.append(DocumentChunk(
                    source=filename,
                    page=page_num,
                    chunk_number=chunk_counter,
                    text=chunk_text,
                    tokens=len(chunk_tokens)
                ))
                
                chunk_counter += 1
                
                # If we've reached the end, break out of the loop
                if end == len(tokens):
                    break
                
                # Update the start pointer and always move at least one token forward
                start = max(start + self.chunk_size - self.overlap, start + 1)
        
        return chunks