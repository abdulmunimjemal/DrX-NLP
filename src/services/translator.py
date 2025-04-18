import re
import logging
from typing import List, Dict, Any
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from llama_cpp import Llama  # Ensure llama-cpp-python is installed
from src.utils.utils import preprocess_text

class TextTranslator:
    def __init__(self, llm: Llama, max_length: int = 1000, debug: bool = False):
        """
        Initializes the TextTranslator.

        :param llm: An instance of a local LLM (e.g., llama_cpp.Llama).
        :param max_length: Maximum character length before chunking is applied.
        :param debug: Enables verbose logging if set to True.
        """
        self.llm = llm
        self.max_length = max_length
        self.debug = debug
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the given text.

        :param text: The text to detect the language of.
        :return: The detected language code (e.g., 'en', 'ar').
        """
        try:
            text = preprocess_text(text)
            if len(text.split()) < 10:
                logging.warning("Text too short for reliable language detection.")
                return "en"
            language = detect(text)
            return language
        except LangDetectException as e:
            logging.warning(f"Language detection failed: {e}")
            return "en"

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks not exceeding max_length, attempting to split at sentence boundaries.

        :param text: The text to chunk.
        :return: A list of text chunks.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > self.max_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        logging.debug(f"Chunked text into {len(chunks)} parts.")
        return chunks

    def translate_text(self, text: str, source_lang: str) -> str:
        """
        Translates text to English if the source language is neither English nor Arabic.

        :param text: The text to translate.
        :param source_lang: The detected source language code.
        :return: The translated text.
        """
        if source_lang in ['en', 'ar']:
            logging.debug("No translation needed.")
            return text
        try:
            translated = GoogleTranslator(source=source_lang, target='en').translate(text)
            logging.debug("Translation successful.")
            return translated
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            return text

    def improve_fluency(self, original: str, translated: str) -> str:
        """
        Improves the fluency of the translated text using the local LLM.

        :param original: The original text.
        :param translated: The translated text.
        :return: The fluency-improved text.
        """
        prompt = (
            "You are a translation post-editor. Given the ORIGINAL text and its RAW MACHINE TRANSLATION, "
            "produce a refined, fluent, and natural English translation that preserves the meaning and formatting.\n\n"
            f"ORIGINAL:\n\"\"\"\n{original}\n\"\"\"\n\n"
            f"RAW TRANSLATION:\n\"\"\"\n{translated}\n\"\"\"\n\n"
            "REFINED TRANSLATION:"
        )
        try:
            response = self.llm(prompt=prompt, max_tokens=1024, temperature=0.2)
            refined = response.get("choices", [{}])[0].get("text", "").strip()
            if refined:
                logging.debug("Fluency improvement successful.")
                return refined
            else:
                logging.warning("LLM returned empty response; using raw translation.")
                return translated
        except Exception as e:
            logging.error(f"Fluency improvement failed: {e}")
            return translated

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a list of documents, translating and improving fluency as needed.

        :param documents: A list of dictionaries, each containing a 'text' key.
        :return: A list of dictionaries with the 'text' key updated.
        """
        processed_docs = []
        for idx, doc in enumerate(documents):
            if 'text' not in doc:
                logging.warning(f"Document at index {idx} missing 'text' key; skipping.")
                continue
            text = doc['text']
            logging.info(f"Processing document {idx + 1}/{len(documents)}.")
            lang = self.detect_language(text)
            if len(text) > self.max_length:
                chunks = self.chunk_text(text)
                translated_chunks = []
                for chunk in chunks:
                    translated = self.translate_text(chunk, lang)
                    if lang not in ['en', 'ar']:
                        translated = self.improve_fluency(chunk, translated)
                    translated_chunks.append(translated)
                final_text = "\n".join(translated_chunks)
            else:
                translated = self.translate_text(text, lang)
                if lang not in ['en', 'ar']:
                    translated = self.improve_fluency(text, translated)
                final_text = translated
            updated_doc = doc.copy()
            updated_doc['text'] = final_text
            processed_docs.append(updated_doc)
        return processed_docs
