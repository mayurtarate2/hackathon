# document_pipeline/vectorstore.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
from document_pipeline.chunk_schema import DocumentChunk
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional, Any
import logging
import time
import json

logger = logging.getLogger(__name__)
load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "hackathon-doc-index")

class EnhancedVectorStore:
    """Enhanced vector store with improved search capabilities and query expansion."""
    
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX", "hackathon-doc-index")
        
        # Initialize index
        self._initialize_index()
        self.index = self.pc.Index(self.index_name)
        
        # Query expansion terms for different domains
        self.domain_synonyms = {
            'insurance': ['coverage', 'policy', 'premium', 'claim', 'benefit', 'protection'],
            'medical': ['health', 'treatment', 'diagnosis', 'condition', 'symptom', 'care'],
            'legal': ['contract', 'agreement', 'terms', 'conditions', 'liability', 'obligation'],
            'financial': ['payment', 'cost', 'fee', 'expense', 'charge', 'amount']
        }
    
    def _initialize_index(self):
        """Initialize or create Pinecone index with enhanced configuration."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            try:
                from pinecone import ServerlessSpec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Successfully created index: {self.index_name}")
                time.sleep(10)  # Wait for index initialization
            except Exception as e:
                logger.error(f"Failed to create index: {e}")
                raise
        
        try:
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to connect to index: {e}")
            raise
    
    def upsert_chunks_enhanced(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Enhanced upsert with better metadata and error tracking.
        """
        if not chunks:
            logger.warning("No chunks to upsert")
            return {"success": False, "message": "No chunks provided"}
        
        batch_size = 50  # Reduced for stability
        vectors = []
        failed_chunks = []
        
        for chunk in chunks:
            try:
                if not chunk.embedding:
                    failed_chunks.append(f"Chunk {chunk.chunk_id}: No embedding")
                    continue
                
                if len(chunk.embedding) != 1536:
                    failed_chunks.append(f"Chunk {chunk.chunk_id}: Invalid dimension {len(chunk.embedding)}")
                    continue
                
                # Enhanced metadata
                metadata = {
                    "text": chunk.text[:2000],  # Increased metadata size
                    "token_count": chunk.token_count,
                    "char_start": chunk.char_range[0],
                    "char_end": chunk.char_range[1],
                    "doc_id": chunk.doc_id,
                    "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
                    "pipeline_version": chunk.pipeline_version or "v1.0"
                }
                
                # Add enhanced metadata if available
                if hasattr(chunk, 'section_type'):
                    metadata["section_type"] = getattr(chunk, 'section_type', 'body')
                if hasattr(chunk, 'semantic_score'):
                    metadata["semantic_score"] = getattr(chunk, 'semantic_score', 0.0)
                if hasattr(chunk, 'keywords'):
                    keywords = getattr(chunk, 'keywords', [])
                    metadata["keywords"] = json.dumps(keywords[:20])  # Limit keywords
                
                vectors.append({
                    "id": chunk.chunk_id,
                    "values": chunk.embedding,
                    "metadata": metadata
                })
                
            except Exception as e:
                failed_chunks.append(f"Chunk {chunk.chunk_id}: {str(e)}")
                continue
        
        if not vectors:
            return {
                "success": False, 
                "message": "No valid vectors to upsert",
                "failed_chunks": failed_chunks
            }
        
        # Batch upsert with retry logic
        batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]
        successful_batches = 0
        failed_batches = []
        
        for i, batch in enumerate(batches):
            try:
                self._upsert_with_retry(batch, max_retries=3)
                successful_batches += 1
                logger.info(f"Successfully upserted batch {i+1}/{len(batches)}")
            except Exception as e:
                failed_batches.append(f"Batch {i+1}: {str(e)}")
                logger.error(f"Failed to upsert batch {i+1}: {e}")
        
        return {
            "success": successful_batches > 0,
            "total_vectors": len(vectors),
            "successful_batches": successful_batches,
            "total_batches": len(batches),
            "failed_chunks": failed_chunks,
            "failed_batches": failed_batches
        }
    
    def _upsert_with_retry(self, batch: List[Dict], max_retries: int = 3):
        """Upsert with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                self.index.upsert(vectors=batch)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = 2 ** attempt
                logger.warning(f"Upsert attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
    
    def query_enhanced(self, 
                      query_embedding: List[float], 
                      top_k: int = 10,
                      query_text: str = "",
                      filters: Optional[Dict] = None,
                      alpha: float = 0.7) -> List[Dict[str, Any]]:
        """
        Enhanced query with hybrid search, query expansion, and result reranking.
        """
        try:
            # Expand query if needed
            expanded_queries = self._expand_query(query_text, query_embedding)
            
            all_results = []
            
            # Multi-query search
            for i, (expanded_embedding, weight) in enumerate(expanded_queries):
                try:
                    # Pinecone query with enhanced parameters
                    results = self.index.query(
                        vector=expanded_embedding,
                        top_k=min(top_k * 2, 50),  # Get more results for reranking
                        include_metadata=True,
                        filter=filters
                    )
                    
                    # Add weight to results
                    for match in results.matches:
                        match_dict = {
                            'id': match.id,
                            'score': match.score * weight,
                            'metadata': match.metadata,
                            'query_index': i
                        }
                        all_results.append(match_dict)
                        
                except Exception as e:
                    logger.warning(f"Query {i} failed: {e}")
                    continue
            
            if not all_results:
                return []
            
            # Deduplicate and rerank results
            deduplicated = self._deduplicate_results(all_results)
            reranked = self._rerank_results(deduplicated, query_text, alpha)
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Enhanced query failed: {e}")
            # Fallback to simple query
            return self._simple_query(query_embedding, top_k, filters)
    
    def _expand_query(self, query_text: str, query_embedding: List[float]) -> List[Tuple[List[float], float]]:
        """
        Expand query with domain-specific synonyms and variations.
        """
        expanded_queries = [(query_embedding, 1.0)]  # Original query with full weight
        
        if not query_text:
            return expanded_queries
        
        query_lower = query_text.lower()
        
        # Detect domain and add relevant synonyms
        detected_domains = []
        for domain, synonyms in self.domain_synonyms.items():
            if any(syn in query_lower for syn in synonyms):
                detected_domains.append(domain)
        
        # For now, return original query (can be enhanced with embedding modifications)
        # In a full implementation, you would generate embeddings for expanded queries
        
        return expanded_queries
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results and combine scores."""
        seen_ids = {}
        
        for result in results:
            result_id = result['id']
            if result_id in seen_ids:
                # Combine scores (take maximum)
                seen_ids[result_id]['score'] = max(seen_ids[result_id]['score'], result['score'])
            else:
                seen_ids[result_id] = result
        
        return list(seen_ids.values())
    
    def _rerank_results(self, results: List[Dict], query_text: str, alpha: float) -> List[Dict]:
        """
        Rerank results using multiple signals.
        """
        if not results:
            return results
        
        # Calculate additional ranking signals
        for result in results:
            metadata = result.get('metadata', {})
            
            # Semantic score boost
            semantic_score = float(metadata.get('semantic_score', 0.0))
            
            # Section type boost
            section_type = metadata.get('section_type', 'body')
            section_boost = {
                'important': 1.2,
                'header': 1.1,
                'list': 1.05,
                'table': 1.03,
                'body': 1.0
            }.get(section_type, 1.0)
            
            # Token count penalty for very short or very long chunks
            token_count = int(metadata.get('token_count', 0))
            length_penalty = 1.0
            if token_count < 50:
                length_penalty = 0.8
            elif token_count > 1000:
                length_penalty = 0.9
            
            # Keyword match boost
            keyword_boost = 1.0
            if query_text and 'keywords' in metadata:
                try:
                    keywords = json.loads(metadata['keywords'])
                    query_words = set(query_text.lower().split())
                    keyword_matches = len(query_words.intersection(set(keywords)))
                    keyword_boost = 1.0 + (keyword_matches * 0.1)
                except:
                    pass
            
            # Combined score
            base_score = result['score']
            enhanced_score = (
                base_score * alpha +
                semantic_score * (1 - alpha) * 0.3 +
                (section_boost - 1.0) * 0.2 +
                (length_penalty - 1.0) * 0.1 +
                (keyword_boost - 1.0) * 0.4
            )
            
            result['enhanced_score'] = enhanced_score
            result['ranking_signals'] = {
                'semantic_score': semantic_score,
                'section_boost': section_boost,
                'length_penalty': length_penalty,
                'keyword_boost': keyword_boost
            }
        
        # Sort by enhanced score
        results.sort(key=lambda x: x.get('enhanced_score', x['score']), reverse=True)
        
        return results
    
    def _simple_query(self, query_embedding: List[float], top_k: int, filters: Optional[Dict] = None) -> List[Dict]:
        """Fallback simple query."""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filters
            )
            
            return [{
                'id': match.id,
                'score': match.score,
                'metadata': match.metadata
            } for match in results.matches]
            
        except Exception as e:
            logger.error(f"Simple query failed: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}

# Initialize global vector store instance
vector_store = EnhancedVectorStore()

# Backward compatibility functions
def upsert_chunks(chunks: List[DocumentChunk]):
    """Backward compatible upsert function."""
    return vector_store.upsert_chunks_enhanced(chunks)

def query_vectorstore(query_embedding: List[float], top_k: int = 8, query_text: str = "") -> List[Dict]:
    """Backward compatible query function."""
    results = vector_store.query_enhanced(query_embedding, top_k, query_text)
    
    # Convert to expected format
    formatted_results = []
    for result in results:
        formatted_results.append({
            'id': result['id'],
            'score': result.get('enhanced_score', result['score']),
            'metadata': result['metadata']
        })
    
    return formatted_results

def query_similar_chunks(query_embedding: list[float], top_k: int = 10):
    """
    Enhanced Pinecone query with comprehensive error handling
    """
    if not query_embedding:
        print("❌ No query embedding provided")
        return []
        
    if len(query_embedding) != 1536:
        print(f"❌ Invalid query embedding dimension: {len(query_embedding)}")
        return []
    
    try:
        # Use the global vector store instance
        response = vector_store.index.query(
            vector=query_embedding,
            top_k=min(top_k, 100),  # Allow larger retrieval set for better reranking
            include_metadata=True,
            include_values=False,  # Don't need embedding values, saves bandwidth
            namespace=""  # Use default namespace
        )
        
        if not response or not response.matches:
            print("⚠️ No matches found in Pinecone")
            return []
        
        # Filter out very low similarity matches early
        filtered_matches = [
            match for match in response.matches 
            if match.score > 0.05  # Lower threshold for more results
        ]
        
        print(f"📊 Pinecone returned {len(response.matches)} matches, {len(filtered_matches)} above threshold")
        return filtered_matches
        
    except Exception as e:
        print(f"❌ Pinecone query failed: {e}")
        # Return empty list to allow the system to continue gracefully
        return []
