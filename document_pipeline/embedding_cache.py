import json
import os
import time
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from document_pipeline.chunk_schema import DocumentChunk

class MaximumAccuracyEmbedder:
    """NO-CACHE embedder for maximum accuracy - generates fresh embeddings every time"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.logger = logging.getLogger(__name__)
        self.embedding_model = "text-embedding-3-small"
        
    def create_question_optimized_embeddings(self, chunks: List[str], questions: List[str]) -> List[List[float]]:
        """Generate fresh embeddings optimized for specific questions - NO CACHING"""
        
        self.logger.info(f"🚀 Generating FRESH embeddings for {len(chunks)} chunks (NO CACHE)")
        embeddings = []
        
        # Analyze question patterns for optimization
        question_context = self._build_question_context(questions)
        
        for i, chunk in enumerate(chunks):
            try:
                start_time = time.time()
                
                # Create question-aware embedding input
                optimized_input = self._optimize_chunk_for_embedding(chunk, question_context)
                
                # Generate fresh embedding (always new)
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=optimized_input,
                    dimensions=1536
                )
                
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                
                processing_time = time.time() - start_time
                
                self.logger.info(f"✅ Fresh embedding {i+1}/{len(chunks)} generated in {processing_time:.2f}s")
                
                # Rate limiting for API stability
                if i < len(chunks) - 1:  # Don't sleep after last chunk
                    time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"❌ Embedding generation failed for chunk {i+1}: {e}")
                # Retry once with basic chunk
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=chunk[:8000],
                        dimensions=1536
                    )
                    embeddings.append(response.data[0].embedding)
                    self.logger.info(f"✅ Fallback embedding {i+1} generated")
                except Exception as retry_error:
                    self.logger.error(f"❌ Retry failed for chunk {i+1}: {retry_error}")
                    raise
        
        self.logger.info(f"🎉 All {len(embeddings)} embeddings generated FRESH (NO CACHE)")
        return embeddings
    
    def _build_question_context(self, questions: List[str]) -> Dict:
        """Analyze questions to build optimization context"""
        all_questions = ' '.join(questions).lower()
        
        return {
            "question_types": {
                "comparative": any(word in all_questions for word in ["compare", "difference", "versus", "vs", "between"]),
                "definitional": any(word in all_questions for word in ["what is", "define", "definition", "meaning"]),
                "procedural": any(word in all_questions for word in ["how to", "process", "procedure", "steps", "method"]),
                "quantitative": any(word in all_questions for word in ["how much", "cost", "amount", "percentage", "number"]),
                "temporal": any(word in all_questions for word in ["when", "duration", "period", "time", "deadline"]),
                "coverage": any(word in all_questions for word in ["cover", "include", "exclude", "benefit", "eligible"]),
                "conditional": any(word in all_questions for word in ["if", "when", "condition", "requirement", "criteria"])
            },
            "domain_indicators": {
                "insurance": any(word in all_questions for word in ["policy", "premium", "claim", "coverage", "deductible"]),
                "medical": any(word in all_questions for word in ["treatment", "diagnosis", "medical", "health", "disease"]),
                "financial": any(word in all_questions for word in ["payment", "cost", "fee", "charge", "financial"]),
                "legal": any(word in all_questions for word in ["legal", "law", "regulation", "compliance", "terms"])
            },
            "complexity_level": max(len(q.split()) for q in questions),
            "question_count": len(questions)
        }
    
    def _optimize_chunk_for_embedding(self, chunk: str, question_context: Dict) -> str:
        """Optimize chunk text based on question analysis"""
        
        # Build context-aware prompt for embedding
        context_parts = []
        
        # Add domain context
        if question_context["domain_indicators"]["insurance"]:
            context_parts.append("INSURANCE POLICY DOCUMENT")
        elif question_context["domain_indicators"]["medical"]:
            context_parts.append("MEDICAL/HEALTH DOCUMENT")
        elif question_context["domain_indicators"]["financial"]:
            context_parts.append("FINANCIAL DOCUMENT")
        elif question_context["domain_indicators"]["legal"]:
            context_parts.append("LEGAL/REGULATORY DOCUMENT")
        else:
            context_parts.append("DOCUMENT CONTENT")
        
        # Add question type context
        question_types = question_context["question_types"]
        if question_types["comparative"]:
            context_parts.append("COMPARISON ANALYSIS")
        if question_types["definitional"]:
            context_parts.append("DEFINITIONS AND EXPLANATIONS")
        if question_types["procedural"]:
            context_parts.append("PROCEDURES AND PROCESSES")
        if question_types["coverage"]:
            context_parts.append("COVERAGE AND BENEFITS")
        
        # Build optimized input
        context_header = " | ".join(context_parts)
        optimized_input = f"[{context_header}]\n\n{chunk}"
        
        # Ensure token limit compliance
        return optimized_input[:8000]

def embed_with_cache(chunk: DocumentChunk) -> DocumentChunk:
    """
    NO-CACHE embedding function - always generates fresh embeddings for maximum accuracy
    """
    try:
        if not chunk.text or not chunk.text.strip():
            logger.warning(f"Empty text for chunk {chunk.chunk_id}")
            chunk.embedding = []
            return chunk
        
        # Always generate fresh embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk.text[:8000],
            dimensions=1536
        )
        
        if response.data and len(response.data) > 0:
            embedding = response.data[0].embedding
            if len(embedding) == 1536:
                chunk.embedding = embedding
                logger.debug(f"Generated fresh embedding for chunk {chunk.chunk_id}")
            else:
                logger.warning(f"Invalid embedding dimension {len(embedding)} for chunk {chunk.chunk_id}")
                chunk.embedding = []
        else:
            logger.warning(f"No embedding data for chunk {chunk.chunk_id}")
            chunk.embedding = []
        
        return chunk
        
    except Exception as e:
        logger.error(f"Embedding failed for chunk {chunk.chunk_id}: {e}")
        chunk.embedding = []
        return chunk

def embed_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """
    NO-CACHE batch embedding function - always generates fresh embeddings
    """
    if not chunks:
        logger.warning("No chunks to embed")
        return []
        
    logger.info(f"🚀 Generating FRESH embeddings for {len(chunks)} chunks (NO CACHE)")
    
    results = []
    successful_embeddings = 0
    
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, DocumentChunk):
            logger.error(f"Invalid chunk type at index {i}: {type(chunk)}")
            continue
            
        embedded_chunk = embed_with_cache(chunk)  # This now generates fresh embeddings
        results.append(embedded_chunk)
        
        if embedded_chunk.embedding:
            successful_embeddings += 1
        
        # Progress logging for large batches
        if (i + 1) % 10 == 0:
            logger.info(f"📊 Generated {i + 1}/{len(chunks)} fresh embeddings ({successful_embeddings} successful)")

    logger.info(f"✅ Fresh embedding generation complete: {successful_embeddings}/{len(chunks)} chunks successful (NO CACHE)")
    return results
