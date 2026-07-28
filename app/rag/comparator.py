"""
Multi-document comparison service.
"""
from typing import List, Dict, Any

from app.rag.prompts import COMPARISON_PROMPT_TEMPLATE
from app.rag.gemini_client import get_gemini_client
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_CHARS_PER_DOCUMENT = 8000


class ComparisonService:
    def __init__(self):
        self.gemini = get_gemini_client()

    def compare_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        documents: list of {"file_name": str, "chunks": List[{"page_number": int, "text": str}]}
        """
        file_names = ", ".join(d["file_name"] for d in documents)

        context_sections = []
        for doc in documents:
            excerpt = ""
            for chunk in doc["chunks"]:
                addition = f"\n[{doc['file_name']} - Page {chunk['page_number']}]\n{chunk['text']}\n"
                if len(excerpt) + len(addition) > MAX_CHARS_PER_DOCUMENT:
                    break
                excerpt += addition
            context_sections.append(f"=== Document: {doc['file_name']} ===\n{excerpt}")

        context_str = "\n\n".join(context_sections)

        prompt = COMPARISON_PROMPT_TEMPLATE.format(file_names=file_names, context=context_str)
        return self.gemini.generate(prompt)
