from openai import OpenAI
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from document_pipeline.chunk_schema import DocumentChunk
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _embed_single_chunk(chunk: DocumentChunk):
    # Use text-embedding-3-small for consistency with existing vectors
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk.text
    )
    return DocumentChunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        token_count=chunk.token_count,
        char_range=chunk.char_range,
        embedding=response.data[0].embedding
    )

def embed_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_embed_single_chunk, chunks))
    return results