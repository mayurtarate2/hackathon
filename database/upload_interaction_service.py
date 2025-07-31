"""
Enhanced Upload and Interaction Logging Service for HackRX 6.0 Project
Comprehensive logging for document uploads and user interactions with optimized performance
"""

import os
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv
import uuid

load_dotenv()

# Try to import SQLAlchemy components and models
try:
    from sqlalchemy import create_engine, func, desc, asc
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.exc import SQLAlchemyError
    from database.upload_interaction_models import (
        Base, DocumentUploadLog, UserInteractionLog, ModelVersionLog
    )
    POSTGRES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PostgreSQL dependencies not available: {e}")
    print("📝 Running in fallback mode without enhanced logging")
    POSTGRES_AVAILABLE = False

class UploadInteractionLogger:
    """
    Enhanced logging service for document uploads and user interactions
    Provides comprehensive tracking while maintaining high performance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.postgres_enabled = POSTGRES_AVAILABLE and self._initialize_database()
        
        # Performance optimization: batch logging for high-volume scenarios
        self.batch_size = 100
        self.pending_interactions = []
        self.pending_uploads = []
        
        if self.postgres_enabled:
            self.logger.info("✅ Enhanced upload and interaction logging initialized")
        else:
            self.logger.warning("⚠️ PostgreSQL not available, using fallback logging")
    
    def _initialize_database(self) -> bool:
        """Initialize database connection and create tables"""
        try:
            self.database_url = os.getenv("DATABASE_URL")
            if not self.database_url:
                self.logger.error("DATABASE_URL environment variable not set")
                return False
            
            self.engine = create_engine(
                self.database_url, 
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                echo=False
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            # Test connection
            with self.SessionLocal() as db:
                db.execute("SELECT 1")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    def log_document_upload(
        self,
        document_url: str,
        document_name: Optional[str] = None,
        original_filename: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        file_type: Optional[str] = None,
        uploader_id: Optional[str] = None,
        uploader_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        upload_source: str = "api",
        upload_method: str = "url_fetch",
        storage_info: Optional[Dict] = None,
        custom_metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Log document upload with comprehensive metadata
        
        Args:
            document_url: URL or storage path of the document
            document_name: Display name of the document
            original_filename: Original filename from upload
            file_size_bytes: File size in bytes
            file_type: File type (pdf, docx, txt, etc.)
            uploader_id: User ID if available
            uploader_ip: IP address of uploader
            user_agent: Browser user agent
            upload_source: Source of upload (web, api, mobile)
            upload_method: Method used (direct_upload, url_fetch)
            storage_info: Cloud storage information
            custom_metadata: Additional metadata
            
        Returns:
            Upload ID if successful, None otherwise
        """
        
        if not self.postgres_enabled:
            return self._fallback_log_upload(document_url, uploader_id, file_type)
        
        upload_id = f"upload_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        try:
            with self.SessionLocal() as db:
                # Generate document hash for duplicate detection
                document_hash = hashlib.sha256(document_url.encode()).hexdigest()
                
                # Check for duplicates
                existing_upload = db.query(DocumentUploadLog).filter(
                    DocumentUploadLog.document_hash == document_hash
                ).first()
                
                is_duplicate = existing_upload is not None
                original_upload_id = existing_upload.upload_id if existing_upload else None
                
                # Detect file type if not provided
                if not file_type:
                    file_type = self._detect_file_type(document_url)
                
                # Extract storage information
                storage_provider = "unknown"
                storage_path = document_url
                storage_bucket = None
                storage_region = None
                
                if storage_info:
                    storage_provider = storage_info.get("provider", "unknown")
                    storage_bucket = storage_info.get("bucket")
                    storage_region = storage_info.get("region")
                
                # Create upload log entry
                upload_log = DocumentUploadLog(
                    upload_id=upload_id,
                    document_url=document_url,
                    document_name=document_name or self._extract_filename(document_url),
                    original_filename=original_filename,
                    file_size_bytes=file_size_bytes,
                    file_type=file_type,
                    document_hash=document_hash,
                    
                    # Upload metadata
                    upload_timestamp=datetime.utcnow(),
                    upload_source=upload_source,
                    upload_method=upload_method,
                    
                    # Uploader information
                    uploader_id=uploader_id,
                    uploader_ip=uploader_ip,
                    uploader_user_agent=user_agent,
                    uploader_session=f"session_{int(time.time())}",
                    
                    # Storage information
                    storage_provider=storage_provider,
                    storage_path=storage_path,
                    storage_bucket=storage_bucket,
                    storage_region=storage_region,
                    
                    # Processing status
                    processing_status="pending",
                    
                    # Duplicate detection
                    is_duplicate=is_duplicate,
                    original_upload_id=original_upload_id,
                    
                    # Metadata
                    document_category=self._classify_document_category(document_name or document_url),
                    document_language="en",  # Default, can be detected later
                    custom_metadata=custom_metadata or {}
                )
                
                db.add(upload_log)
                db.commit()
                
                self.logger.info(f"✅ Document upload logged: {upload_id}")
                return upload_id
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Failed to log document upload: {e}")
            return None
    
    def update_upload_processing_status(
        self,
        upload_id: str,
        status: str,
        chunks_created: Optional[int] = None,
        embeddings_generated: Optional[int] = None,
        processing_duration: Optional[float] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ) -> bool:
        """Update the processing status of a document upload"""
        
        if not self.postgres_enabled:
            return self._fallback_update_upload(upload_id, status)
        
        try:
            with self.SessionLocal() as db:
                upload_log = db.query(DocumentUploadLog).filter(
                    DocumentUploadLog.upload_id == upload_id
                ).first()
                
                if not upload_log:
                    self.logger.warning(f"Upload log not found: {upload_id}")
                    return False
                
                # Update processing status
                upload_log.processing_status = status
                
                if status == "processing" and not upload_log.processing_started_at:
                    upload_log.processing_started_at = datetime.utcnow()
                elif status in ["completed", "failed"]:
                    upload_log.processing_completed_at = datetime.utcnow()
                    if processing_duration:
                        upload_log.processing_duration_seconds = int(processing_duration)
                
                # Update results
                if chunks_created is not None:
                    upload_log.chunks_created = chunks_created
                if embeddings_generated is not None:
                    upload_log.embeddings_generated = embeddings_generated
                
                # Update errors and warnings
                if errors:
                    upload_log.processing_errors = errors
                if warnings:
                    upload_log.processing_warnings = warnings
                
                db.commit()
                
                self.logger.info(f"✅ Upload status updated: {upload_id} -> {status}")
                return True
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Failed to update upload status: {e}")
            return False
    
    def log_user_interaction(
        self,
        user_input: str,
        model_output: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        document_upload_id: Optional[str] = None,
        document_url: Optional[str] = None,
        model_version: str = "gpt-3.5-turbo",
        pipeline_version: str = "v3.0_NO_CACHE",
        processing_mode: str = "no_cache_maximum_accuracy",
        performance_metrics: Optional[Dict] = None,
        quality_metrics: Optional[Dict] = None,
        api_usage: Optional[Dict] = None,
        user_context: Optional[Dict] = None,
        error_info: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Log comprehensive user interaction with optimized performance
        
        Args:
            user_input: User's question or input
            model_output: Model's response
            user_id: User identifier if available
            session_id: Session identifier for grouping interactions
            document_upload_id: Reference to document upload
            document_url: Document used for this interaction
            model_version: Model version used
            pipeline_version: Pipeline version used
            processing_mode: Processing mode
            performance_metrics: Processing time metrics
            quality_metrics: Relevance and quality scores
            api_usage: OpenAI API usage statistics
            user_context: User context information
            error_info: Error information if interaction failed
            
        Returns:
            Interaction ID if successful, None otherwise
        """
        
        if not self.postgres_enabled:
            return self._fallback_log_interaction(user_input, model_output, user_id)
        
        interaction_id = f"interaction_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        try:
            # Extract metrics with defaults
            perf = performance_metrics or {}
            quality = quality_metrics or {}
            api = api_usage or {}
            context = user_context or {}
            error = error_info or {}
            
            # Calculate processing times in milliseconds
            total_time_ms = int((perf.get("total_processing_time", 0)) * 1000)
            embedding_time_ms = int((perf.get("embedding_time", 0)) * 1000)
            retrieval_time_ms = int((perf.get("retrieval_time", 0)) * 1000)
            llm_time_ms = int((perf.get("llm_time", 0)) * 1000)
            
            # Quality scores (0-100)
            relevance_score = int(quality.get("relevance_score", 0) * 100) if quality.get("relevance_score") else None
            confidence_score = int(quality.get("confidence_score", 80))  # Default 80%
            
            # API usage
            tokens_input = api.get("tokens_input", 0)
            tokens_output = api.get("tokens_output", 0)
            tokens_total = tokens_input + tokens_output
            api_calls = api.get("api_calls", 1)
            cost_cents = int((api.get("estimated_cost_usd", 0)) * 100)  # Convert to cents
            
            # User information
            user_ip = context.get("ip_address")
            user_agent = context.get("user_agent")
            timezone = context.get("timezone", "UTC")
            
            # Create interaction log (optimized for batch processing)
            interaction_data = {
                "interaction_id": interaction_id,
                "session_id": session_id or f"session_{int(time.time())}",
                "user_id": user_id,
                "user_ip": user_ip,
                "user_agent": user_agent,
                "user_timezone": timezone,
                
                "interaction_timestamp": datetime.utcnow(),
                "interaction_type": "question_answer",
                "interaction_source": context.get("source", "api"),
                
                "user_input": user_input,
                "input_language": self._detect_language(user_input),
                "input_length_chars": len(user_input),
                "input_word_count": len(user_input.split()),
                
                "model_version": model_version,
                "pipeline_version": pipeline_version,
                "processing_mode": processing_mode,
                
                "document_upload_id": document_upload_id,
                "document_url": document_url,
                "document_chunks_used": quality.get("chunks_used", 0),
                
                "model_output": model_output,
                "output_language": self._detect_language(model_output),
                "output_length_chars": len(model_output),
                "output_word_count": len(model_output.split()),
                
                "total_processing_time_ms": total_time_ms,
                "embedding_time_ms": embedding_time_ms,
                "retrieval_time_ms": retrieval_time_ms,
                "llm_time_ms": llm_time_ms,
                
                "relevance_score": relevance_score,
                "confidence_score": confidence_score,
                "answer_completeness": int(quality.get("completeness", 80)),
                
                "openai_tokens_input": tokens_input,
                "openai_tokens_output": tokens_output,
                "openai_tokens_total": tokens_total,
                "openai_api_calls": api_calls,
                "estimated_cost_usd": cost_cents,
                
                "interaction_success": not bool(error),
                "error_type": error.get("type"),
                "error_message": error.get("message"),
                "error_stage": error.get("stage"),
                
                "question_category": self._classify_question_category(user_input),
                "question_type": self._classify_question_type(user_input),
                "answer_type": self._classify_answer_type(model_output)
            }
            
            # Use batch processing for high performance
            if len(self.pending_interactions) < self.batch_size:
                self.pending_interactions.append(interaction_data)
                
                # Process batch when full or after timeout
                if len(self.pending_interactions) >= self.batch_size:
                    self._flush_interaction_batch()
            else:
                # Direct insert for immediate consistency
                self._insert_interaction_direct(interaction_data)
            
            self.logger.info(f"✅ User interaction logged: {interaction_id}")
            return interaction_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log user interaction: {e}")
            return None
    
    def _insert_interaction_direct(self, interaction_data: Dict) -> bool:
        """Insert interaction directly to database"""
        try:
            with self.SessionLocal() as db:
                interaction_log = UserInteractionLog(**interaction_data)
                db.add(interaction_log)
                db.commit()
                return True
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Direct interaction insert failed: {e}")
            return False
    
    def _flush_interaction_batch(self) -> bool:
        """Flush pending interactions to database in batch"""
        if not self.pending_interactions:
            return True
        
        try:
            with self.SessionLocal() as db:
                # Bulk insert for performance
                interaction_objects = [UserInteractionLog(**data) for data in self.pending_interactions]
                db.bulk_save_objects(interaction_objects)
                db.commit()
                
                count = len(self.pending_interactions)
                self.pending_interactions.clear()
                
                self.logger.info(f"✅ Batch logged {count} interactions")
                return True
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Batch interaction logging failed: {e}")
            self.pending_interactions.clear()  # Clear to prevent memory buildup
            return False
    
    def get_upload_statistics(self, days: int = 7) -> Dict:
        """Get document upload statistics"""
        if not self.postgres_enabled:
            return {"message": "PostgreSQL not available"}
        
        try:
            with self.SessionLocal() as db:
                since = datetime.utcnow() - timedelta(days=days)
                
                # Basic upload stats
                total_uploads = db.query(DocumentUploadLog).filter(
                    DocumentUploadLog.upload_timestamp >= since
                ).count()
                
                successful_uploads = db.query(DocumentUploadLog).filter(
                    DocumentUploadLog.upload_timestamp >= since,
                    DocumentUploadLog.processing_status == "completed"
                ).count()
                
                # File type distribution
                file_types = db.query(
                    DocumentUploadLog.file_type,
                    func.count(DocumentUploadLog.id).label('count')
                ).filter(
                    DocumentUploadLog.upload_timestamp >= since
                ).group_by(DocumentUploadLog.file_type).all()
                
                # Storage size stats
                size_stats = db.query(
                    func.sum(DocumentUploadLog.file_size_bytes).label('total_size'),
                    func.avg(DocumentUploadLog.file_size_bytes).label('avg_size'),
                    func.max(DocumentUploadLog.file_size_bytes).label('max_size')
                ).filter(
                    DocumentUploadLog.upload_timestamp >= since
                ).first()
                
                return {
                    "period_days": days,
                    "total_uploads": total_uploads,
                    "successful_uploads": successful_uploads,
                    "success_rate": (successful_uploads / total_uploads * 100) if total_uploads > 0 else 0,
                    "file_type_distribution": {ft: count for ft, count in file_types},
                    "storage_stats": {
                        "total_size_bytes": int(size_stats.total_size or 0),
                        "avg_size_bytes": int(size_stats.avg_size or 0),
                        "max_size_bytes": int(size_stats.max_size or 0)
                    }
                }
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Failed to get upload statistics: {e}")
            return {"error": str(e)}
    
    def get_interaction_statistics(self, days: int = 7) -> Dict:
        """Get user interaction statistics"""
        if not self.postgres_enabled:
            return {"message": "PostgreSQL not available"}
        
        try:
            with self.SessionLocal() as db:
                since = datetime.utcnow() - timedelta(days=days)
                
                # Basic interaction stats
                total_interactions = db.query(UserInteractionLog).filter(
                    UserInteractionLog.interaction_timestamp >= since
                ).count()
                
                successful_interactions = db.query(UserInteractionLog).filter(
                    UserInteractionLog.interaction_timestamp >= since,
                    UserInteractionLog.interaction_success == True
                ).count()
                
                # Unique users
                unique_users = db.query(UserInteractionLog.user_id).filter(
                    UserInteractionLog.interaction_timestamp >= since,
                    UserInteractionLog.user_id.isnot(None)
                ).distinct().count()
                
                # Performance metrics
                perf_stats = db.query(
                    func.avg(UserInteractionLog.total_processing_time_ms).label('avg_time'),
                    func.avg(UserInteractionLog.relevance_score).label('avg_relevance'),
                    func.sum(UserInteractionLog.openai_tokens_total).label('total_tokens'),
                    func.sum(UserInteractionLog.estimated_cost_usd).label('total_cost_cents')
                ).filter(
                    UserInteractionLog.interaction_timestamp >= since,
                    UserInteractionLog.interaction_success == True
                ).first()
                
                return {
                    "period_days": days,
                    "total_interactions": total_interactions,
                    "successful_interactions": successful_interactions,
                    "success_rate": (successful_interactions / total_interactions * 100) if total_interactions > 0 else 0,
                    "unique_users": unique_users,
                    "performance": {
                        "avg_processing_time_ms": int(perf_stats.avg_time or 0),
                        "avg_relevance_score": int(perf_stats.avg_relevance or 0),
                        "total_tokens_used": int(perf_stats.total_tokens or 0),
                        "total_cost_usd": (perf_stats.total_cost_cents or 0) / 100
                    }
                }
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Failed to get interaction statistics: {e}")
            return {"error": str(e)}
    
    # Utility methods
    def _detect_file_type(self, document_url: str) -> str:
        """Detect file type from URL"""
        url_lower = document_url.lower()
        if '.pdf' in url_lower:
            return 'pdf'
        elif '.docx' in url_lower:
            return 'docx'
        elif '.doc' in url_lower:
            return 'doc'
        elif '.txt' in url_lower:
            return 'txt'
        elif '.eml' in url_lower:
            return 'email'
        else:
            return 'unknown'
    
    def _extract_filename(self, document_url: str) -> str:
        """Extract filename from URL"""
        try:
            return document_url.split('/')[-1]
        except:
            return "unknown_document"
    
    def _classify_document_category(self, document_name: str) -> str:
        """Classify document category"""
        name_lower = document_name.lower()
        if any(word in name_lower for word in ['insurance', 'policy', 'premium', 'coverage']):
            return 'insurance'
        elif any(word in name_lower for word in ['legal', 'contract', 'agreement', 'terms']):
            return 'legal'
        elif any(word in name_lower for word in ['hr', 'employee', 'handbook', 'benefits']):
            return 'hr'
        elif any(word in name_lower for word in ['compliance', 'regulation', 'audit']):
            return 'compliance'
        else:
            return 'general'
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (can be enhanced with proper language detection)"""
        # Basic implementation - can be enhanced with langdetect library
        return "en"  # Default to English
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type"""
        q_lower = question.lower().strip()
        if q_lower.startswith(('what', 'what is', 'what are')):
            return 'what'
        elif q_lower.startswith(('how', 'how to', 'how much', 'how many')):
            return 'how'
        elif q_lower.startswith(('when', 'what time')):
            return 'when'
        elif q_lower.startswith(('where', 'which')):
            return 'where'
        elif q_lower.startswith(('why', 'why is')):
            return 'why'
        elif q_lower.startswith(('can', 'could', 'may', 'might')):
            return 'can'
        elif q_lower.startswith(('is', 'are', 'was', 'were')):
            return 'yes_no'
        else:
            return 'other'
    
    def _classify_question_category(self, question: str) -> str:
        """Classify question category"""
        q_lower = question.lower()
        if any(word in q_lower for word in ['premium', 'payment', 'cost', 'fee']):
            return 'premium'
        elif any(word in q_lower for word in ['coverage', 'cover', 'benefit']):
            return 'coverage'
        elif any(word in q_lower for word in ['claim', 'process']):
            return 'claims'
        elif any(word in q_lower for word in ['exclude', 'exclusion']):
            return 'exclusions'
        else:
            return 'general'
    
    def _classify_answer_type(self, answer: str) -> str:
        """Classify answer type"""
        if len(answer) < 50:
            return 'brief'
        elif len(answer) < 200:
            return 'concise'
        else:
            return 'detailed'
    
    # Fallback methods for when PostgreSQL is not available
    def _fallback_log_upload(self, document_url: str, uploader_id: Optional[str], file_type: Optional[str]) -> Optional[str]:
        """Fallback upload logging"""
        upload_id = f"fallback_upload_{int(time.time())}"
        self.logger.info(f"📝 [FALLBACK] Upload logged: {upload_id} - {document_url}")
        return upload_id
    
    def _fallback_update_upload(self, upload_id: str, status: str) -> bool:
        """Fallback upload status update"""
        self.logger.info(f"📝 [FALLBACK] Upload status: {upload_id} -> {status}")
        return True
    
    def _fallback_log_interaction(self, user_input: str, model_output: str, user_id: Optional[str]) -> Optional[str]:
        """Fallback interaction logging"""
        interaction_id = f"fallback_interaction_{int(time.time())}"
        self.logger.info(f"📝 [FALLBACK] Interaction logged: {interaction_id}")
        self.logger.info(f"  Input: {user_input[:100]}...")
        self.logger.info(f"  Output: {model_output[:100]}...")
        return interaction_id

# Create singleton instance
try:
    upload_interaction_logger = UploadInteractionLogger()
except Exception as e:
    print(f"⚠️ Failed to initialize upload interaction logger: {e}")
    upload_interaction_logger = None
