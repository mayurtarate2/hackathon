from document_pipeline.vectorstore import query_similar_chunks
from document_pipeline.chunk_schema import DocumentChunk
import re
from typing import List, Dict, Any

def retrieve_relevant_chunks(query: str, top_k: int = 8):
    """
    Advanced retrieval system with improved query processing and semantic matching
    """
    print(f"🔍 Searching for: {query}")
    
    # Step 1: Enhanced query preprocessing
    enhanced_queries = generate_query_variations(query)
    
    # Step 2: Get embeddings for all query variations
    from document_pipeline.embedding_cache import embed_with_cache
    
    all_matches = []
    
    for i, enhanced_query in enumerate(enhanced_queries):
        print(f"🎯 Query variation {i+1}: {enhanced_query[:50]}...")
        
        temp_chunk = DocumentChunk(
            chunk_id=f"query_{i}", 
            text=enhanced_query, 
            token_count=len(enhanced_query.split()), 
            char_range=(0, len(enhanced_query)), 
            embedding=[]
        )
        
        embedded_query = embed_with_cache(temp_chunk)
        query_vector = embedded_query.embedding
        
        # Search with increased scope
        matches = query_similar_chunks(query_vector, top_k=40)
        
        # Add source query info to matches
        for match in matches:
            match.query_source = i
            match.original_score = match.score
            
        all_matches.extend(matches)
    
    print(f"📊 Found {len(all_matches)} total matches across all query variations")
    
    if not all_matches:
        print("❌ No matches found in vector database")
        return []
    
    # Step 3: Advanced deduplication and reranking
    deduplicated_matches = deduplicate_matches(all_matches)
    print(f"� After deduplication: {len(deduplicated_matches)} unique matches")
    
    # Step 4: Semantic reranking with multiple criteria
    reranked_matches = semantic_rerank(query, deduplicated_matches)
    
    # Step 5: Apply dynamic thresholds
    final_matches = apply_dynamic_threshold(query, reranked_matches)
    
    # Return top results
    result_chunks = final_matches[:top_k]
    print(f"📤 Returning {len(result_chunks)} most relevant chunks")
    
    if result_chunks:
        first_text = result_chunks[0].metadata.get("text", "")[:150]
        print(f"🔍 Top result preview: {first_text}...")
    
    return [
        {
            "id": match.id,
            "score": match.final_score,
            "text": match.metadata.get("text", "No text available")
        }
        for match in result_chunks
    ]

def generate_query_variations(query: str) -> List[str]:
    """Generate multiple query variations for better semantic matching"""
    variations = [query]  # Original query
    query_lower = query.lower().strip()
    
    # Remove question words and focus on key terms
    question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'who', 'is', 'are', 'does', 'do', 'can', 'will']
    key_terms_query = ' '.join([word for word in query_lower.split() if word not in question_words])
    if key_terms_query != query_lower:
        variations.append(key_terms_query)
    
    # Insurance domain replacements for better matching
    domain_replacements = {
        'grace period': ['grace period', 'payment grace', 'premium grace', 'grace time'],
        'waiting period': ['waiting period', 'wait time', 'waiting time', 'period wait'],
        'pre-existing': ['pre-existing', 'preexisting', 'existing condition', 'prior condition'],
        'coverage': ['coverage', 'benefit', 'cover', 'benefits covered'],
        'exclusion': ['exclusion', 'excluded', 'not covered', 'exception'],
        'claim': ['claim', 'claims process', 'filing claim', 'claim procedure'],
        'premium': ['premium', 'payment', 'installment', 'cost'],
        'hospital': ['hospital', 'hospitalization', 'medical facility'],
        'age limit': ['age limit', 'maximum age', 'age restriction', 'age criteria'],
        'sum insured': ['sum insured', 'coverage amount', 'insured amount', 'policy amount'],
        'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery'],
        'renewal': ['renewal', 'renew', 'continuation', 'extend policy']
    }
    
    # Add domain-specific variations
    for key_phrase, replacements in domain_replacements.items():
        if key_phrase in query_lower:
            for replacement in replacements:
                if replacement != key_phrase:
                    variations.append(query_lower.replace(key_phrase, replacement))
    
    # Add noun phrase extraction
    important_terms = extract_key_terms(query)
    if important_terms:
        variations.append(' '.join(important_terms))
    
    # Remove duplicates while preserving order
    unique_variations = []
    seen = set()
    for var in variations:
        if var not in seen:
            unique_variations.append(var)
            seen.add(var)
    
    return unique_variations[:5]  # Limit to top 5 variations

def extract_key_terms(query: str) -> List[str]:
    """Extract key terms from query using simple NLP rules"""
    # Remove common stop words
    stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                  'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                  'to', 'was', 'will', 'with', 'what', 'how', 'when', 'where', 'this'}
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    key_terms = [word for word in words if word not in stop_words]
    
    return key_terms

def deduplicate_matches(matches: List) -> List:
    """Remove duplicate matches based on chunk content"""
    seen_ids = set()
    unique_matches = []
    
    for match in matches:
        if match.id not in seen_ids:
            unique_matches.append(match)
            seen_ids.add(match.id)
    
    return unique_matches

def semantic_rerank(query: str, matches: List) -> List:
    """Advanced semantic reranking using multiple signals"""
    query_lower = query.lower()
    query_terms = set(extract_key_terms(query))
    
    for match in matches:
        text = match.metadata.get("text", "").lower()
        base_score = getattr(match, 'original_score', match.score)
        
        # Exact phrase matching boost
        exact_phrases = extract_phrases(query_lower)
        phrase_boost = sum(0.15 for phrase in exact_phrases if phrase in text)
        
        # Term frequency boost
        text_terms = set(extract_key_terms(text))
        term_overlap = len(query_terms.intersection(text_terms))
        term_boost = (term_overlap / max(len(query_terms), 1)) * 0.1
        
        # Length penalty for very short chunks
        length_penalty = 0.05 if len(text) < 100 else 0
        
        # Domain context boost
        domain_boost = calculate_domain_boost(query_lower, text)
        
        # Question type alignment boost
        question_boost = calculate_question_type_boost(query_lower, text)
        
        # Calculate final score
        final_score = (base_score + phrase_boost + term_boost + 
                      domain_boost + question_boost - length_penalty)
        
        match.final_score = final_score
    
    # Sort by final score
    matches.sort(key=lambda x: x.final_score, reverse=True)
    return matches

def extract_phrases(text: str, min_words: int = 2) -> List[str]:
    """Extract meaningful phrases from text"""
    words = text.split()
    phrases = []
    
    for i in range(len(words) - min_words + 1):
        phrase = ' '.join(words[i:i + min_words])
        if len(phrase) > 5:  # Skip very short phrases
            phrases.append(phrase)
    
    return phrases

def calculate_domain_boost(query: str, text: str) -> float:
    """Calculate boost based on insurance domain relevance"""
    domain_terms = {
        'high_value': ['grace period', 'waiting period', 'sum insured', 'premium payment', 
                      'claim procedure', 'policy renewal', 'coverage amount'],
        'medium_value': ['insurance', 'policy', 'coverage', 'benefit', 'claim', 'premium',
                        'hospital', 'medical', 'treatment', 'condition', 'age', 'family'],
        'context': ['amount', 'limit', 'period', 'time', 'payment', 'process', 'procedure']
    }
    
    boost = 0
    for term in domain_terms['high_value']:
        if term in query and term in text:
            boost += 0.08
    
    for term in domain_terms['medium_value']:
        if term in query and term in text:
            boost += 0.04
    
    for term in domain_terms['context']:
        if term in query and term in text:
            boost += 0.02
    
    return min(boost, 0.2)  # Cap the boost

def calculate_question_type_boost(query: str, text: str) -> float:
    """Boost based on question type and text content alignment"""
    boost = 0
    
    if query.startswith(('what is', 'what are')):
        if any(indicator in text for indicator in ['means', 'defined as', 'refers to', 'is the']):
            boost += 0.05
    
    elif query.startswith('how'):
        if any(indicator in text for indicator in ['process', 'procedure', 'steps', 'method']):
            boost += 0.05
    
    elif 'period' in query:
        if any(indicator in text for indicator in ['days', 'months', 'years', 'period', 'time']):
            boost += 0.06
    
    elif 'amount' in query or 'limit' in query:
        if any(indicator in text for indicator in ['rupees', 'amount', 'limit', 'maximum', 'minimum']):
            boost += 0.05
    
    return boost

def apply_dynamic_threshold(query: str, matches: List) -> List:
    """Apply dynamic threshold based on query complexity and match quality"""
    if not matches:
        return matches
    
    query_lower = query.lower()
    
    # Determine base threshold based on query type
    if any(term in query_lower for term in ['what is', 'define', 'definition']):
        base_threshold = 0.25  # Higher threshold for definitions
    elif any(term in query_lower for term in ['how', 'procedure', 'process']):
        base_threshold = 0.20  # Medium threshold for procedures
    elif len(query.split()) > 8:  # Complex queries
        base_threshold = 0.18
    else:
        base_threshold = 0.15  # Lower threshold for simple queries
    
    # Adaptive threshold based on score distribution
    scores = [match.final_score for match in matches]
    max_score = max(scores)
    score_gap = max_score - min(scores) if len(scores) > 1 else 0
    
    # If there's a clear winner, be more selective
    if score_gap > 0.3:
        adaptive_threshold = max_score * 0.8
    else:
        adaptive_threshold = base_threshold
    
    # Apply threshold
    filtered_matches = [match for match in matches if match.final_score >= adaptive_threshold]
    
    # Ensure we return at least some results
    if not filtered_matches and matches:
        print("⚠️ No matches above threshold, returning top matches")
        filtered_matches = matches[:8]
    
    print(f"✅ {len(filtered_matches)} matches above threshold {adaptive_threshold:.3f}")
    return filtered_matches
