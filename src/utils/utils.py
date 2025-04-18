AVAILABLE_MODELS = {
    "Mistral-7B-Instruct-v0.1 (Q4_K_M)": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    },
     "Mistral-7B (Q4_K_M)": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf",
        "filename": "mistral-7b-v0.1.Q4_K_M.gguf"
    },
    "Llama-2-7B-Chat (Q4_K_M)": {
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        "filename": "llama-2-7b-chat.Q4_K_M.gguf"
    },
    "Phi-2 (Q4_K_M)": {
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "filename": "phi-2.Q4_K_M.gguf"
    },
    "Gemma-2B-IT (Q4_K_M)": {
        "url": "https://huggingface.co/TheBloke/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q4_K_M.gguf",
        "filename": "gemma-2b-it.Q4_K_M.gguf"
    },
    "Mixtral-8x7B-Instruct-v0.1 (Q4_K_M)": {
        "url": "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        "filename": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
    },
    "CodeLlama-7B-Instruct (Q4_K_M)": {
        "url": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf",
        "filename": "codellama-7b-instruct.Q4_K_M.gguf"
    },
    "CodeLlama-13B-Instruct (Q4_K_M)": {
        "url": "https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_K_M.gguf",
        "filename": "codellama-13b-instruct.Q4_K_M.gguf"
    },
}

MIN_TEXT_LENGTH_WORDS = 5 

def preprocess_text(text):
    """Cleans text for better language detection."""
    import re
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # no url
    text = re.sub(r'\S+@\S+', '', text) # no email
    text = re.sub(r'<.*?>', '', text) # no HTML tags
    text = re.sub(r'\d+', '', text) # no numbers
    text = re.sub(r'[^\w\s.?!]', '', text) # no special characters
    text = re.sub(r'\s+', ' ', text).strip() # no extra spaces
    text = re.sub(r'^[^a-zA-Z0-9]+', '', text) # no leading special characters
    # text = text.lower()
    return text