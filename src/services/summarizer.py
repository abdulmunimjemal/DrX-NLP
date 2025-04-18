import math
from src.core.core import track_performance, logger
from typing import List, Dict, Any, Optional # Added Optional
from llama_cpp import Llama
# Import necessary components from rouge-score
from rouge_score import rouge_scorer, scoring


class Summarizer:
    """
    Summarizes large documents using llama-cpp-python, handles context limits
    via iterative chunking, and includes ROUGE evaluation capabilities.
    """
    def __init__(self,
                 llm: Llama,
                 max_length: int = 1000,
                 chunk_size: int = 1500,
                 chunk_overlap: int = 100,
                 rouge_metrics: Optional[List[str]] = None): # Added rouge_metrics
        """
        Args:
            llm: An initialized llama_cpp.Llama instance.
            max_length: Target maximum character length for the FINAL summary.
            chunk_size: Character size for breaking down text.
            chunk_overlap: Character overlap between chunks.
            rouge_metrics: List of ROUGE metrics to compute (e.g., ['rouge1', 'rouge2', 'rougeL']).
                           Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'].
        """
        if not isinstance(llm, Llama):
             raise TypeError("llm must be an instance of llama_cpp.Llama")

        self.llm = llm
        self.max_length = max_length

        effective_chunk_size = chunk_size
        if hasattr(llm, 'context_params') and llm.context_params.n_ctx > 0:
             estimated_max_chars = (llm.context_params.n_ctx - 500) * 3
             if chunk_size > estimated_max_chars:
                 logger.warning(f"Initial chunk_size {chunk_size} might be too large for n_ctx {llm.context_params.n_ctx}. Reducing to {estimated_max_chars}.")
                 effective_chunk_size = estimated_max_chars

        self.chunk_size = effective_chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_max_tokens = 512
        self.llm_temperature = 0.2
        self.stop_sequences = ["\n\n", "Summary:", "Summarize:", "Chunk Summary:", "User:", "Assistant:", "###"]

        # Initialize ROUGE scorer
        self.rouge_metrics = rouge_metrics or ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        self.scorer = rouge_scorer.RougeScorer(self.rouge_metrics, use_stemmer=True)

    def _split_text(self, text: str) -> List[str]:
        """Splits text into chunks based on character count with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            next_start = start + self.chunk_size - self.chunk_overlap
            if next_start <= start:
                 start += self.chunk_size
            else:
                 start = next_start
            if start >= len(text):
                 break
        return chunks

    def _summarize_chunk(self, chunk: str, style: str, is_final_rewrite: bool = False) -> str:
        """Generates a summary for a single text chunk using the Llama LLM."""
        if not chunk: return ""
        if is_final_rewrite:
            prompt = (
                f"Rewrite the following text into a single, coherent, and {style} final summary. "
                f"Ensure key information is retained and it flows well. "
                f"Aim for less than {self.max_length} characters:\n\n"
                f"Text:\n{chunk}\n\nCoherent Summary:"
            )
            target_tokens = min(self.llm_max_tokens * 2, max(256, math.ceil(self.max_length / 3)))
        else:
            prompt = (
                f"Provide a {style} summary of the following text chunk, focusing on key ideas:\n\n"
                f"Text Chunk:\n{chunk}\n\nSummary of Chunk:"
            )
            target_tokens = self.llm_max_tokens
        try:
            response = self.llm(
                prompt=prompt, max_tokens=target_tokens,
                temperature=self.llm_temperature, stop=self.stop_sequences
            )
            summary = response.get('choices', [{}])[0].get('text', '').strip()
            print(f"Summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Error during Llama API call: {e}")
            try: # Debug info
                 prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8", errors="ignore")))
                 logger.info(f"Failed chunk summary. Prompt tokens: {prompt_tokens}, n_ctx: {self.llm.context_params.n_ctx}")
            except: pass
            return ""

    @track_performance
    def summarize(self, text: str, style: str = 'concise') -> str:
        """
        Generates a summary for input text using iterative refinement.
        """
        if not text: return ""
        if len(text) <= self.max_length and len(text) <= self.chunk_size:
            logger.info("Input text short enough for single-pass summarization.")
            return self._summarize_chunk(text, style, is_final_rewrite=True)

        logger.info(f"Starting iterative summarization for text length {len(text)}.")
        chunks = self._split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks.")
        chunk_summaries = [self._summarize_chunk(chunk, style) for chunk in chunks]
        chunk_summaries = [s for s in chunk_summaries if s]
        if not chunk_summaries:
            logger.error("Could not generate summaries for any initial chunks.")
            return ""
        merged_summary = "\n\n".join(chunk_summaries)
        logger.info(f"Initial merged summary length: {len(merged_summary)} characters.")

        iteration = 1; max_iterations = 10
        while len(merged_summary) > self.max_length and iteration <= max_iterations:
            logger.info(f"Iteration {iteration}: Merged length {len(merged_summary)} > max_length {self.max_length}. Refining...")
            chunks = self._split_text(merged_summary)
            logger.info(f"Split intermediate summary into {len(chunks)} chunks for refinement.")
            chunk_summaries = [self._summarize_chunk(chunk, style) for chunk in chunks]
            chunk_summaries = [s for s in chunk_summaries if s]
            if not chunk_summaries:
                logger.error(f"No summaries generated during refinement iteration {iteration}. Returning previous.")
                return merged_summary
            merged_summary = "\n\n".join(chunk_summaries)
            logger.info(f"Iteration {iteration}: New merged summary length: {len(merged_summary)} chars.")
            iteration += 1
        if iteration > max_iterations: logger.warning("Reached max iterations.")

        logger.info("Performing final coherence rewrite pass.")
        final_summary = self._summarize_chunk(merged_summary, style, is_final_rewrite=True)

        if len(final_summary) > self.max_length:
            logger.warning(f"Final summary length ({len(final_summary)}) exceeds target ({self.max_length}). Truncating intelligently.")
            cut_off_point = self.max_length
            last_period_index = final_summary.rfind('.', max(0, cut_off_point - 50), cut_off_point)
            if last_period_index != -1: final_summary = final_summary[:last_period_index + 1]
            else: final_summary = final_summary[:self.max_length]

        logger.info(f"Final summary length: {len(final_summary)} characters.")
        return final_summary

    # --- NEW EVALUATION METHOD ---
    @track_performance
    def evaluate(self, generated_summaries: List[str], reference_summaries: List[str]) -> Dict[str, Any]:
        """
        Evaluates generated summaries against reference summaries using ROUGE.

        Args:
            generated_summaries: List of summaries produced by the model.
            reference_summaries: List of corresponding ground-truth reference summaries.

        Returns:
            A dictionary containing aggregated ROUGE scores (precision, recall, fmeasure)
            and detailed scores for each generated/reference pair.
            Example: {'average': {'rouge1': Score(...), ...}, 'detailed': [{'rouge1': Score(...), ...}, ...]}
                     Access f-measure like: results['average']['rouge1'].fmeasure
        """
        if len(generated_summaries) != len(reference_summaries):
            raise ValueError("Number of generated summaries must match the number of reference summaries.")
        if not generated_summaries:
            logger.warning("Received empty lists for evaluation.")
            return {'average': {}, 'detailed': []}

        aggregator = scoring.BootstrapAggregator()
        detailed_scores = []

        for gen_sum, ref_sum in zip(generated_summaries, reference_summaries):
            if not gen_sum or not ref_sum:
                 logger.warning("Skipping evaluation for an item due to empty generated or reference summary.")
                 # Add placeholder for detailed scores to maintain list alignment if needed
                 detailed_scores.append({metric: None for metric in self.rouge_metrics})
                 continue
            scores = self.scorer.score(ref_sum, gen_sum)
            aggregator.add_scores(scores)
            detailed_scores.append(scores) # Store the raw Score objects

        # Calculate aggregated results (provides confidence intervals, but we'll use mid-point)
        aggregated_result = aggregator.aggregate()

        # Format output for clarity
        output = {
            # Average scores across all pairs
            'average': {metric: aggregated_result[metric].mid for metric in self.rouge_metrics},
            # Detailed scores for each pair
            'detailed': [{metric: scores[metric] for metric in self.rouge_metrics} if scores else None
                          for scores in detailed_scores] # Extract from raw Score objects
        }

        return output