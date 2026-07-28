"""
Document summarization service using a safe batched (map-reduce style) strategy
for lengthy documents so we never put unbounded text into a single prompt.
"""
from typing import List, Dict, Any

from app.rag.prompts import SUMMARIZATION_PROMPT_TEMPLATE
from app.rag.gemini_client import get_gemini_client
from app.core.logging import get_logger

logger = get_logger(__name__)

# Roughly 4 characters per token; keep well under typical context limits.
MAX_CONTEXT_CHARS_PER_BATCH = 12000
MAX_BATCHES_FOR_DIRECT_SUMMARY = 1


class SummarizationService:
    def __init__(self):
        self.gemini = get_gemini_client()

    def _batch_chunks(self, chunk_texts: List[str]) -> List[str]:
        """Groups chunk texts into batches that stay under the per-prompt character budget."""
        batches: List[str] = []
        current = ""
        for text in chunk_texts:
            if len(current) + len(text) + 2 > MAX_CONTEXT_CHARS_PER_BATCH:
                if current:
                    batches.append(current)
                current = text
            else:
                current = f"{current}\n\n{text}" if current else text
        if current:
            batches.append(current)
        return batches

    def summarize_document(self, file_name: str, chunk_texts: List[str]) -> Dict[str, Any]:
        batches = self._batch_chunks(chunk_texts)

        if len(batches) <= MAX_BATCHES_FOR_DIRECT_SUMMARY:
            prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(file_name=file_name, context=batches[0] if batches else "")
            summary = self.gemini.generate(prompt)
            return {"summary": summary, "batches_used": len(batches)}

        # Map step: summarize each batch independently to condense the content.
        partial_summaries = []
        for i, batch in enumerate(batches):
            partial_prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(
                file_name=f"{file_name} (part {i + 1}/{len(batches)})",
                context=batch,
            )
            partial_summaries.append(self.gemini.generate(partial_prompt))

        # Reduce step: summarize the combined partial summaries into the final structured summary.
        combined = "\n\n".join(partial_summaries)
        final_prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(file_name=file_name, context=combined)
        final_summary = self.gemini.generate(final_prompt)

        return {"summary": final_summary, "batches_used": len(batches)}
