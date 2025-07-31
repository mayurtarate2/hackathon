# test_vectorstore.py
import os
from dotenv import load_dotenv

load_dotenv()  # ✅ This ensures .env variables like OPENAI_API_KEY are loaded
# Import the necessary functions from your document pipeline

from document_pipeline.embedder import embed_chunks
from document_pipeline.vectorstore import query_similar_chunks

def run_test_query():
    query = "What are the exclusions in the Arogya Sanjeevani Policy?"
    print(f"🔎 Embedding query: {query}")
    
    mock_chunk = {
        "chunk_id": "query_001",
        "text": query,
        "token_count": len(query.split()),
        "char_range": (0, len(query)),
        "embedding": None
    }
    query_vector = embed_chunks([mock_chunk])[0]["embedding"]

    print("📡 Querying Pinecone for top-5 relevant chunks...")
    matches = query_similar_chunks(query_vector, top_k=5)

    print(f"\n🎯 Top {len(matches)} Matches:\n" + "-"*50)
    for i, match in enumerate(matches, 1):
        print(f"[{i}] Score: {match.score:.4f}")
        print(match.metadata.get("text", "")[:300])  # Truncate long chunk
        print("-" * 50)

if __name__ == "__main__":
    run_test_query()
