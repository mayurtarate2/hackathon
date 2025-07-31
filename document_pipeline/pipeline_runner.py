# document_pipeline/pipeline_runner.py

from .parser import EnhancedDocumentParser, extract_text_from_pdf  # Backward compatibility
from .cleaner import remove_common_headers_footers, normalize_whitespace
from .chunker import EnhancedChunker, recursive_split  # Backward compatibility
from .embedding_cache import MaximumAccuracyEmbedder, embed_chunks  # NO-CACHE embedding
from .chunk_schema import DocumentChunk
from .vectorstore import EnhancedVectorStore, upsert_chunks  # Enhanced vector store
from .retriever import EnhancedRetriever, retrieve_relevant_chunks  # Enhanced retrieval
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MaximumAccuracyPipelineRunner:
    """NO-CACHE document processing pipeline for maximum accuracy - always fresh processing."""
    
    def __init__(self):
        self.parser = EnhancedDocumentParser()
        self.chunker = EnhancedChunker(chunk_size=600, overlap=200)
        self.embedder = MaximumAccuracyEmbedder()  # NO-CACHE embedder
        self.vector_store = EnhancedVectorStore()
        self.retriever = EnhancedRetriever()
        
    def run_maximum_accuracy_pipeline(self,
                            document_path: str,
                            doc_id: Optional[str] = None,
                            pipeline_version: str = "v3.0_NO_CACHE",
                            cleanup_temp: bool = True) -> Dict[str, Any]:
        """
        NO-CACHE pipeline for maximum accuracy - always generates fresh embeddings.
        
        Returns:
            Dict with pipeline results, statistics, and any warnings/errors
        """
        pipeline_start = datetime.utcnow()
        
        try:
            logger.info(f"🚀 Starting MAXIMUM ACCURACY NO-CACHE pipeline for: {document_path}")
            
            # Step 1: Document Detection and Parsing
            document_result = self._parse_document_enhanced(document_path)
            if not document_result['success']:
                return {
                    'success': False,
                    'error': document_result['error'],
                    'stage': 'document_parsing'
                }
            
            pages = document_result['pages']
            metadata = document_result['metadata']
            
            # Step 2: Text Cleaning and Preprocessing
            cleaned_result = self._clean_and_preprocess(pages, metadata)
            full_text = cleaned_result['text']
            
            if not full_text.strip():
                return {
                    'success': False,
                    'error': 'No meaningful text found after cleaning',
                    'stage': 'text_cleaning'
                }
            
            # Step 3: Enhanced Chunking
            chunking_result = self._chunk_text_enhanced(full_text, metadata)
            if not chunking_result['success']:
                return {
                    'success': False,
                    'error': chunking_result['error'],
                    'stage': 'text_chunking'
                }
            
            raw_chunks = chunking_result['chunks']
            
            # Step 4: Create DocumentChunk objects
            document_chunks = self._create_document_chunks(
                raw_chunks, doc_id, pipeline_version, metadata
            )
            
            # Step 5: NO-CACHE Maximum Accuracy Embedding Generation
            embedding_result = self._generate_embeddings_maximum_accuracy(document_chunks)
            if not embedding_result['success']:
                return {
                    'success': False,
                    'error': embedding_result['error'],
                    'stage': 'embedding_generation'
                }
            
            embedded_chunks = embedding_result['chunks']
            
            # Step 6: Enhanced Vector Store Upsert
            upsert_result = self._upsert_to_vector_store_enhanced(embedded_chunks)
            
            # Step 7: Cleanup
            if cleanup_temp and document_path.startswith('/tmp'):
                try:
                    os.unlink(document_path)
                    logger.debug(f"Cleaned up temporary file: {document_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")
            
            # Calculate pipeline statistics
            pipeline_end = datetime.utcnow()
            processing_time = (pipeline_end - pipeline_start).total_seconds()
            
            result = {
                'success': True,
                'chunks': embedded_chunks,
                'statistics': {
                    'document_format': metadata.get('format', 'unknown'),
                    'total_pages': len(pages),
                    'total_chunks': len(embedded_chunks),
                    'successful_embeddings': len([c for c in embedded_chunks if c.embedding]),
                    'processing_time_seconds': processing_time,
                    'average_chunk_size': sum(c.token_count for c in embedded_chunks) / len(embedded_chunks) if embedded_chunks else 0
                },
                'metadata': metadata,
                'upsert_result': upsert_result,
                'embedding_stats': embedding_result.get('stats', {}),
                'chunking_stats': chunking_result.get('stats', {})
            }
            
            logger.info(f"Enhanced pipeline completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': f"Pipeline failed: {str(e)}",
                'stage': 'unknown'
            }
    
    def _parse_document_enhanced(self, document_path: str) -> Dict[str, Any]:
        """Enhanced document parsing supporting multiple formats."""
        try:
            result = self.parser.extract_text_from_document(document_path)
            return {
                'success': True,
                'pages': result['pages'],
                'metadata': result['metadata']
            }
        except Exception as e:
            logger.error(f"Document parsing failed: {e}")
            return {
                'success': False,
                'error': f"Failed to parse document: {str(e)}"
            }
    
    def _clean_and_preprocess(self, pages: List[str], metadata: Dict) -> Dict[str, Any]:
        """Enhanced text cleaning and preprocessing."""
        try:
            # Apply existing cleaning functions
            cleaned_pages = remove_common_headers_footers(pages)
            normalized_pages = [normalize_whitespace(p) for p in cleaned_pages]
            
            # Additional format-specific cleaning
            document_format = metadata.get('format', '').lower()
            
            if document_format == 'email':
                # Email-specific cleaning
                normalized_pages = self._clean_email_content(normalized_pages)
            elif document_format == 'docx':
                # DOCX-specific cleaning
                normalized_pages = self._clean_docx_content(normalized_pages)
            
            full_text = "\n".join(normalized_pages)
            
            return {
                'success': True,
                'text': full_text,
                'stats': {
                    'original_pages': len(pages),
                    'cleaned_pages': len(normalized_pages),
                    'final_length': len(full_text)
                }
            }
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return {
                'success': False,
                'error': f"Text cleaning failed: {str(e)}",
                'text': "\n".join(pages)  # Fallback to original
            }
    
    def _clean_email_content(self, pages: List[str]) -> List[str]:
        """Email-specific content cleaning."""
        cleaned = []
        for page in pages:
            # Remove email headers that are not content
            lines = page.split('\n')
            content_lines = []
            
            for line in lines:
                # Skip technical email headers
                if any(header in line.lower() for header in [
                    'content-type:', 'content-transfer-encoding:', 'mime-version:',
                    'x-mailer:', 'message-id:', 'received:', 'return-path:'
                ]):
                    continue
                content_lines.append(line)
            
            cleaned.append('\n'.join(content_lines))
        
        return cleaned
    
    def _clean_docx_content(self, pages: List[str]) -> List[str]:
        """DOCX-specific content cleaning."""
        cleaned = []
        for page in pages:
            # Remove empty table cells and formatting artifacts
            import re
            
            # Remove multiple spaces/tabs
            page = re.sub(r'[ \t]+', ' ', page)
            
            # Remove isolated formatting characters
            page = re.sub(r'\n\s*\n\s*\n', '\n\n', page)
            
            cleaned.append(page)
        
        return cleaned
    
    def _chunk_text_enhanced(self, text: str, metadata: Dict) -> Dict[str, Any]:
        """Enhanced chunking with document-aware strategies."""
        try:
            chunks = self.chunker.enhanced_recursive_split(text, metadata)
            
            return {
                'success': True,
                'chunks': chunks,
                'stats': {
                    'total_chunks': len(chunks),
                    'average_chunk_size': sum(c.get('token_count', 0) for c in chunks) / len(chunks) if chunks else 0,
                    'chunk_types': list(set(c.get('section_type', 'body') for c in chunks))
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced chunking failed: {e}")
            # Fallback to basic chunking
            try:
                chunks = recursive_split(text)
                return {
                    'success': True,
                    'chunks': chunks,
                    'stats': {'fallback_used': True}
                }
            except Exception as e2:
                return {
                    'success': False,
                    'error': f"Both enhanced and fallback chunking failed: {str(e2)}"
                }
    
    def _create_document_chunks(self, raw_chunks: List[Dict], doc_id: str, 
                              pipeline_version: str, metadata: Dict) -> List[DocumentChunk]:
        """Create DocumentChunk objects with enhanced metadata."""
        document_chunks = []
        now = datetime.utcnow()
        
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        for i, chunk_data in enumerate(raw_chunks):
            try:
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_data["text"],
                    token_count=chunk_data["token_count"],
                    char_range=chunk_data["char_range"],
                    embedding=[],  # Will be filled by embedding step
                    doc_id=doc_id,
                    pipeline_version=pipeline_version,
                    page_num=chunk_data.get("page_num"),
                    section_title=chunk_data.get("section_title"),
                    created_at=now
                )
                
                # Add enhanced metadata if available
                if hasattr(chunk, '__dict__'):
                    for key in ['section_type', 'semantic_score', 'keywords', 'entity_count']:
                        if key in chunk_data:
                            setattr(chunk, key, chunk_data[key])
                
                document_chunks.append(chunk)
                
            except Exception as e:
                logger.warning(f"Failed to create chunk {i}: {e}")
                continue
        
        return document_chunks
    
    def _generate_embeddings_maximum_accuracy(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """NO-CACHE embedding generation for maximum accuracy - always fresh."""
        try:
            logger.info(f"🚀 Generating FRESH embeddings for {len(chunks)} chunks (NO CACHE)")
            
            # Use no-cache embedding function
            embedded_chunks = embed_chunks(chunks)  # This now generates fresh embeddings
            
            successful = len([c for c in embedded_chunks if c.embedding])
            
            return {
                'success': successful > 0,
                'chunks': embedded_chunks,
                'stats': {
                    'total_chunks': len(chunks),
                    'successful_embeddings': successful,
                    'processing_mode': 'MAXIMUM_ACCURACY_NO_CACHE',
                    'accuracy_boost': 'FRESH_EMBEDDINGS_EVERY_TIME'
                },
                'error': f"No embeddings generated" if successful == 0 else None
            }
            
        except Exception as e:
            logger.error(f"Maximum accuracy embedding generation failed: {e}")
            return {
                'success': False,
                'error': f"Embedding generation failed: {str(e)}",
                'chunks': chunks
            }
    
    def _upsert_to_vector_store_enhanced(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Enhanced vector store upsert with detailed results."""
        try:
            result = self.vector_store.upsert_chunks_enhanced(chunks)
            logger.info(f"Vector store upsert result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Vector store upsert failed: {e}")
            return {
                'success': False,
                'error': f"Vector store upsert failed: {str(e)}"
            }

# Global instance - NO-CACHE for maximum accuracy
maximum_accuracy_pipeline = MaximumAccuracyPipelineRunner()

def run_pipeline(pdf_path: str, doc_id: Optional[str] = None, 
                pipeline_version: str = "v3.0_NO_CACHE") -> List[DocumentChunk]:
    """
    NO-CACHE pipeline function for maximum accuracy - always generates fresh embeddings.
    """
    try:
        logger.info(f"🚀 Starting MAXIMUM ACCURACY NO-CACHE pipeline")
        result = maximum_accuracy_pipeline.run_maximum_accuracy_pipeline(
            pdf_path, doc_id, pipeline_version
        )
        
        if result['success']:
            logger.info(f"✅ Maximum accuracy pipeline completed successfully (NO CACHE)")
            return result['chunks']
        else:
            logger.error(f"❌ Maximum accuracy pipeline failed: {result.get('error', 'Unknown error')}")
            # Fallback to original pipeline
            return _run_original_pipeline_fallback(pdf_path, doc_id, pipeline_version)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return _run_original_pipeline_fallback(pdf_path, doc_id, pipeline_version)

def _run_original_pipeline_fallback(pdf_path: str, doc_id: Optional[str] = None,
                                  pipeline_version: str = "v1.0") -> List[DocumentChunk]:
    """Fallback to original pipeline logic."""
    try:
        logger.warning("Using fallback pipeline")
        
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            raise ValueError("No text extracted from PDF")

        cleaned_pages = remove_common_headers_footers(pages)
        full_text = "\n".join(normalize_whitespace(p) for p in cleaned_pages)
        
        if not full_text.strip():
            raise ValueError("No meaningful text found after cleaning")

        raw_chunks = recursive_split(full_text)
        
        if not raw_chunks:
            raise ValueError("No chunks created from text")

        now = datetime.utcnow()
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        document_chunks = []
        for i, chunk_data in enumerate(raw_chunks):
            try:
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_data["text"],
                    token_count=chunk_data["token_count"],
                    char_range=chunk_data["char_range"],
                    embedding=[],
                    doc_id=doc_id,
                    pipeline_version=pipeline_version,
                    page_num=None,
                    section_title=None,
                    created_at=now
                )
                document_chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to create chunk {i}: {e}")
                continue

        if not document_chunks:
            raise ValueError("No valid chunks created")

        embedded_chunks = embed_chunks(document_chunks)
        
        if not embedded_chunks:
            raise ValueError("No chunks were successfully embedded")

        upsert_result = upsert_chunks(embedded_chunks)
        
        logger.info(f"Fallback pipeline completed: {len(embedded_chunks)} chunks processed")
        return embedded_chunks
        
    except Exception as e:
        logger.error(f"Fallback pipeline failed: {str(e)}")
        raise Exception(f"All pipeline methods failed: {str(e)}")
