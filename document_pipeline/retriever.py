from document_pipeline.vectorstore import query_vectorstore
from document_pipeline.chunk_schema import DocumentChunk
from document_pipeline.embedding_cache import embed_with_cache
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class EnhancedRetriever:
    """Enhanced retrieval system with improved context understanding and ranking."""
    
    def __init__(self):
        self.domain_keywords = {
            'insurance': {
                'coverage', 'policy', 'premium', 'claim', 'benefit', 'deductible',
                'exclusion', 'liability', 'renewal', 'grace period'
            },
            'medical': {
                'treatment', 'diagnosis', 'procedure', 'medication', 'hospital',
                'doctor', 'patient', 'condition', 'symptoms', 'therapy'
            },
            'financial': {
                'payment', 'cost', 'fee', 'amount', 'price', 'charge',
                'refund', 'billing', 'invoice', 'expense'
            },
            'legal': {
                'contract', 'agreement', 'terms', 'conditions', 'obligation',
                'liability', 'responsibility', 'rights', 'dispute'
            }
        }
        
        self.query_intent_patterns = {
            'definition': ['what is', 'define', 'definition of', 'meaning of', 'explain'],
            'procedure': ['how to', 'procedure', 'process', 'steps', 'method'],
            'comparison': ['difference', 'compare', 'versus', 'vs', 'between'],
            'requirement': ['required', 'need', 'must', 'should', 'eligibility'],
            'coverage': ['covered', 'cover', 'include', 'benefit', 'protection'],
            'exclusion': ['not covered', 'exclude', 'exception', 'limitation'],
            'timeframe': ['when', 'how long', 'period', 'duration', 'time'],
            'amount': ['how much', 'cost', 'price', 'amount', 'fee']
        }
    
    def retrieve_relevant_chunks_enhanced(self, query: str, top_k: int = 10, 
                                        context_expansion: bool = True) -> List[Dict[str, Any]]:
        """
        Enhanced retrieval with better context understanding and multi-stage ranking.
        """
        logger.info(f"Enhanced retrieval for query: {query}")
        
        # Analyze query intent and domain
        query_analysis = self._analyze_query(query)
        logger.debug(f"Query analysis: {query_analysis}")
        
        # Generate query embedding
        query_embedding = self._get_query_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Multi-stage retrieval
        # Stage 1: Broad semantic search
        broad_results = query_vectorstore(
            query_embedding, 
            top_k=min(top_k * 3, 50),  # Get more results for reranking
            query_text=query
        )
        
        if not broad_results:
            logger.warning("No results from vector search")
            return []
        
        logger.info(f"Vector search returned {len(broad_results)} results")
        
        # Stage 2: Enhanced reranking
        reranked_results = self._rerank_results_enhanced(
            broad_results, query, query_analysis
        )
        
        # Stage 3: Context expansion (optional)
        if context_expansion and reranked_results:
            expanded_results = self._expand_context(reranked_results, top_k)
        else:
            expanded_results = reranked_results[:top_k]
        
        # Stage 4: Final formatting
        formatted_results = self._format_results(expanded_results, query_analysis)
        
        logger.info(f"Returning {len(formatted_results)} enhanced results")
        return formatted_results
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand intent and domain."""
        query_lower = query.lower()
        
        # Detect intent
        detected_intent = 'general'
        intent_confidence = 0.0
        
        for intent, patterns in self.query_intent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    detected_intent = intent
                    intent_confidence = 1.0 / len(patterns)  # Simple confidence scoring
                    break
            if intent_confidence > 0:
                break
        
        # Detect domain
        detected_domain = 'general'
        domain_score = 0.0
        
        for domain, keywords in self.domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            score = matches / len(keywords)
            if score > domain_score:
                domain_score = score
                detected_domain = domain
        
        # Extract key terms
        query_terms = self._extract_key_terms(query)
        
        return {
            'intent': detected_intent,
            'intent_confidence': intent_confidence,
            'domain': detected_domain,
            'domain_score': domain_score,
            'key_terms': query_terms,
            'is_multi_part': '?' in query or 'and' in query_lower or 'or' in query_lower
        }
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for better matching."""
        # Remove stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'what', 'when',
            'where', 'why', 'how', 'which', 'who', 'whom', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'
        }
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for query."""
        try:
            temp_chunk = DocumentChunk(
                chunk_id="query_temp",
                text=query,
                token_count=len(query.split()),
                char_range=(0, len(query)),
                embedding=[]
            )
            
            embedded_query = embed_with_cache(temp_chunk)
            return embedded_query.embedding if embedded_query.embedding else None
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None
    
    def _rerank_results_enhanced(self, results: List[Dict], query: str, 
                               query_analysis: Dict) -> List[Dict]:
        """Enhanced reranking with multiple signals."""
        if not results:
            return results
        
        query_lower = query.lower()
        key_terms = query_analysis['key_terms']
        domain = query_analysis['domain']
        intent = query_analysis['intent']
        
        for result in results:
            text = result.get('metadata', {}).get('text', '').lower()
            base_score = result.get('score', 0.0)
            
            # Term frequency scoring
            tf_score = self._calculate_tf_score(text, key_terms)
            
            # Domain relevance scoring
            domain_score = self._calculate_domain_score(text, domain)
            
            # Intent-specific scoring
            intent_score = self._calculate_intent_score(text, intent, query_lower)
            
            # Position and length scoring
            position_score = self._calculate_position_score(text, key_terms)
            length_score = self._calculate_length_score(text)
            
            # Semantic coherence (simple version)
            coherence_score = self._calculate_coherence_score(text, query_lower)
            
            # Combined enhanced score
            enhanced_score = (
                base_score * 0.4 +           # Base semantic similarity
                tf_score * 0.25 +            # Term frequency
                domain_score * 0.15 +        # Domain relevance
                intent_score * 0.1 +         # Intent matching
                position_score * 0.05 +      # Term position
                length_score * 0.03 +        # Content length
                coherence_score * 0.02       # Semantic coherence
            )
            
            result['enhanced_score'] = enhanced_score
            result['scoring_breakdown'] = {
                'base_score': base_score,
                'tf_score': tf_score,
                'domain_score': domain_score,
                'intent_score': intent_score,
                'position_score': position_score,
                'length_score': length_score,
                'coherence_score': coherence_score
            }
        
        # Sort by enhanced score
        results.sort(key=lambda x: x.get('enhanced_score', 0), reverse=True)
        
        return results
    
    def _calculate_tf_score(self, text: str, key_terms: List[str]) -> float:
        """Calculate term frequency score."""
        if not key_terms:
            return 0.0
        
        text_words = text.split()
        total_words = len(text_words)
        
        if total_words == 0:
            return 0.0
        
        tf_scores = []
        for term in key_terms:
            count = text.count(term)
            tf = count / total_words
            # Apply log normalization
            tf_score = math.log(1 + tf) if tf > 0 else 0
            tf_scores.append(tf_score)
        
        return sum(tf_scores) / len(key_terms)
    
    def _calculate_domain_score(self, text: str, domain: str) -> float:
        """Calculate domain-specific relevance score."""
        if domain == 'general' or domain not in self.domain_keywords:
            return 0.0
        
        domain_keywords = self.domain_keywords[domain]
        matches = sum(1 for keyword in domain_keywords if keyword in text)
        
        return matches / len(domain_keywords)
    
    def _calculate_intent_score(self, text: str, intent: str, query: str) -> float:
        """Calculate intent-specific score."""
        intent_signals = {
            'definition': ['definition', 'means', 'refers to', 'defined as', 'is a'],
            'procedure': ['steps', 'procedure', 'process', 'method', 'follow'],
            'comparison': ['difference', 'compare', 'versus', 'unlike', 'while'],
            'requirement': ['required', 'must', 'need to', 'should', 'mandatory'],
            'coverage': ['covered', 'includes', 'benefits', 'protection', 'covers'],
            'exclusion': ['excluded', 'not covered', 'except', 'limitation'],
            'timeframe': ['days', 'months', 'years', 'period', 'duration', 'when'],
            'amount': ['amount', 'cost', 'price', 'fee', 'charge', 'sum']
        }
        
        if intent not in intent_signals:
            return 0.0
        
        signals = intent_signals[intent]
        matches = sum(1 for signal in signals if signal in text)
        
        return matches / len(signals)
    
    def _calculate_position_score(self, text: str, key_terms: List[str]) -> float:
        """Score based on term positions (early terms weighted more)."""
        if not key_terms:
            return 0.0
        
        position_scores = []
        text_length = len(text)
        
        for term in key_terms:
            pos = text.find(term)
            if pos >= 0:
                # Earlier positions get higher scores
                position_score = 1.0 - (pos / text_length)
                position_scores.append(position_score)
        
        return sum(position_scores) / len(key_terms) if position_scores else 0.0
    
    def _calculate_length_score(self, text: str) -> float:
        """Score based on content length (optimal length gets highest score)."""
        length = len(text)
        # Optimal length around 200-500 characters
        if 200 <= length <= 500:
            return 1.0
        elif 100 <= length < 200 or 500 < length <= 800:
            return 0.8
        elif 50 <= length < 100 or 800 < length <= 1200:
            return 0.6
        else:
            return 0.4
    
    def _calculate_coherence_score(self, text: str, query: str) -> float:
        """Simple coherence scoring based on sentence structure."""
        # Simple heuristics for coherence
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Check for proper sentence structure
        proper_sentences = sum(1 for s in sentences if len(s.strip()) > 10 and ' ' in s.strip())
        coherence = proper_sentences / len(sentences)
        
        return coherence
    
    def _expand_context(self, results: List[Dict], top_k: int) -> List[Dict]:
        """Expand context by including related chunks."""
        if not results:
            return results
        
        # For now, return top results (can be enhanced with chunk relationship analysis)
        return results[:top_k]
    
    def _format_results(self, results: List[Dict], query_analysis: Dict) -> List[Dict]:
        """Format results with enhanced metadata."""
        formatted = []
        
        for i, result in enumerate(results):
            formatted_result = {
                'text': result.get('metadata', {}).get('text', ''),
                'score': result.get('enhanced_score', result.get('score', 0)),
                'rank': i + 1,
                'metadata': result.get('metadata', {}),
                'relevance_signals': result.get('scoring_breakdown', {}),
                'query_intent': query_analysis['intent'],
                'query_domain': query_analysis['domain']
            }
            formatted.append(formatted_result)
        
        return formatted

# Global instance
enhanced_retriever = EnhancedRetriever()

# Backward compatibility function
def retrieve_relevant_chunks(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Backward compatible retrieval function with enhanced capabilities.
    """
    try:
        enhanced_results = enhanced_retriever.retrieve_relevant_chunks_enhanced(
            query, top_k
        )
        
        # Convert to expected format
        compatible_results = []
        for result in enhanced_results:
            compatible_result = {
                'text': result['text'],
                'score': result['score'],
                'metadata': result['metadata']
            }
            compatible_results.append(compatible_result)
        
        return compatible_results
        
    except Exception as e:
        logger.error(f"Enhanced retrieval failed, falling back to simple retrieval: {e}")
        # Fallback to simple retrieval logic
        return _simple_retrieve_fallback(query, top_k)

def _simple_retrieve_fallback(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Simple fallback retrieval."""
    try:
        # Generate query embedding
        temp_chunk = DocumentChunk(
            chunk_id="query_fallback",
            text=query,
            token_count=len(query.split()),
            char_range=(0, len(query)),
            embedding=[]
        )
        
        embedded_query = embed_with_cache(temp_chunk)
        if not embedded_query.embedding:
            return []
        
        # Simple vector search
        results = query_vectorstore(embedded_query.embedding, top_k * 2)
        
        # Basic filtering and formatting
        filtered_results = []
        for result in results[:top_k]:
            filtered_results.append({
                'text': result.get('metadata', {}).get('text', ''),
                'score': result.get('score', 0),
                'metadata': result.get('metadata', {})
            })
        
        return filtered_results
        
    except Exception as e:
        logger.error(f"Fallback retrieval also failed: {e}")
        return []
