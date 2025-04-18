import time
import logging
import pandas as pd
import tiktoken
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunk(BaseModel):
    source: str
    page: int
    chunk_number: int
    text: str
    tokens: int

class PerformanceMetrics:
    def __init__(self):
        self.metrics = {}
        
    def track(self, operation: str, time_taken: float, tokens: int):
        tps = tokens / time_taken if time_taken > 0 else 0
        self.metrics.setdefault(operation, []).append(tps)
        logger.info(f"{operation}: {tps:.2f} tokens/sec")

metrics = PerformanceMetrics()

def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def track_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        if isinstance(result, str):
            tokens = count_tokens(result)
        elif isinstance(result, list):
            tokens = sum(count_tokens(chunk.text) for chunk in result)
        else:
            tokens = 0
        metrics.track(func.__name__, time.time() - start_time, tokens)
        return result
    return wrapper