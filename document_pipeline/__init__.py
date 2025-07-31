from .parser import extract_text_from_pdf
from .cleaner import remove_common_headers_footers, normalize_whitespace
from .chunker import recursive_split
from .embedder import embed_chunks
from .chunk_schema import DocumentChunk
from .pipeline_runner import run_pipeline
from .vectorstore import upsert_chunks
from .embedding_cache import embed_with_cache
from .retriever import retrieve_relevant_chunks   # ✅ ADD THIS

__all__ = [
    "extract_text_from_pdf",
    "remove_common_headers_footers",
    "normalize_whitespace",
    "recursive_split",
    "embed_chunks",
    "DocumentChunk",
    "run_pipeline",
    "upsert_chunks",
    "embed_with_cache",
    "retrieve_relevant_chunks"  # ✅ ADD THIS TOO
]
