"""
Enhanced PostgreSQL Database Models for HackRX 6.0 Project
Comprehensive tracking of document processing, performance metrics, and system analytics
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime
from typing import Dict, List, Optional

Base = declarative_base()

class EnhancedQuerySession(Base):
    """Enhanced query session tracking with comprehensive metrics"""
    __tablename__ = "enhanced_query_sessions"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), unique=True, index=True)
    
    # Document information
    document_url = Column(Text, nullable=False)
    document_type = Column(String(50))  # pdf, docx, txt, email, etc.
    document_size_bytes = Column(Integer)
    document_hash = Column(String(64), index=True)  # For duplicate detection and analytics
    document_name = Column(String(500))
    
    # Question and answer data
    questions = Column(JSONB)  # Store as JSONB for better querying
    answers = Column(JSONB)
    question_count = Column(Integer)
    
    # Processing configuration
    processing_mode = Column(String(50), default="no_cache_maximum_accuracy")
    pipeline_version = Column(String(20), default="v3.0_NO_CACHE")
    
    # Performance metrics (all times in seconds)
    total_processing_time = Column(Float)
    document_download_time = Column(Float)
    document_parsing_time = Column(Float)
    text_cleaning_time = Column(Float)
    chunking_time = Column(Float)
    embedding_generation_time = Column(Float)
    vector_store_time = Column(Float)
    vector_search_time = Column(Float)
    answer_generation_time = Column(Float)
    
    # Quality and accuracy metrics
    chunks_generated = Column(Integer)
    embeddings_created = Column(Integer)
    successful_embeddings = Column(Integer)
    average_relevance_score = Column(Float)
    max_relevance_score = Column(Float)
    min_relevance_score = Column(Float)
    questions_answered = Column(Integer)
    successful_answers = Column(Integer)
    
    # API usage and cost tracking
    openai_tokens_used = Column(Integer)
    openai_api_calls = Column(Integer)
    embedding_api_calls = Column(Integer)
    chat_completion_calls = Column(Integer)
    estimated_cost_usd = Column(Float)
    
    # Error handling and debugging
    processing_errors = Column(JSONB)
    warnings = Column(JSONB)
    success_status = Column(Boolean, default=True)
    error_stage = Column(String(100))  # Where processing failed if applicable
    
    # Request metadata
    user_agent = Column(String(500))
    ip_address = Column(String(45))  # IPv6 compatible
    request_source = Column(String(100))  # API, web, mobile, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Analytics fields
    is_duplicate_document = Column(Boolean, default=False)
    cache_hits = Column(Integer, default=0)
    cache_misses = Column(Integer, default=0)
    
    # Indexes for better query performance
    __table_args__ = (
        Index('idx_document_url_hash', 'document_url', 'document_hash'),
        Index('idx_created_at_success', 'created_at', 'success_status'),
        Index('idx_processing_mode_type', 'processing_mode', 'document_type'),
        Index('idx_performance_metrics', 'total_processing_time', 'average_relevance_score'),
        Index('idx_cost_tracking', 'openai_tokens_used', 'estimated_cost_usd'),
    )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON responses"""
        return {
            "session_id": self.session_id,
            "document_url": self.document_url,
            "document_type": self.document_type,
            "processing_mode": self.processing_mode,
            "question_count": self.question_count,
            "total_processing_time": self.total_processing_time,
            "chunks_generated": self.chunks_generated,
            "average_relevance_score": self.average_relevance_score,
            "openai_tokens_used": self.openai_tokens_used,
            "estimated_cost_usd": self.estimated_cost_usd,
            "success_status": self.success_status,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class DocumentProcessingStats(Base):
    """Aggregated statistics for each unique document"""
    __tablename__ = "document_processing_stats"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_hash = Column(String(64), unique=True, index=True)
    document_url = Column(Text)
    document_type = Column(String(50))
    document_name = Column(String(500))
    
    # Processing statistics
    total_requests = Column(Integer, default=1)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    # Performance averages
    avg_processing_time = Column(Float)
    avg_download_time = Column(Float)
    avg_embedding_time = Column(Float)
    avg_search_time = Column(Float)
    avg_answer_time = Column(Float)
    
    # Quality averages
    avg_relevance_score = Column(Float)
    avg_chunks_generated = Column(Float)
    avg_questions_per_session = Column(Float)
    
    # Cost tracking
    total_tokens_used = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    avg_cost_per_session = Column(Float)
    
    # Usage patterns
    common_question_types = Column(JSONB)
    peak_usage_hours = Column(JSONB)
    processing_modes_used = Column(JSONB)
    
    # Timestamps
    first_processed = Column(DateTime, default=datetime.utcnow)
    last_processed = Column(DateTime, default=datetime.utcnow)
    
    # Performance tracking
    best_relevance_score = Column(Float)
    fastest_processing_time = Column(Float)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON responses"""
        return {
            "document_hash": self.document_hash,
            "document_type": self.document_type,
            "total_requests": self.total_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "avg_processing_time": self.avg_processing_time,
            "avg_relevance_score": self.avg_relevance_score,
            "total_cost_usd": self.total_cost_usd,
            "first_processed": self.first_processed.isoformat() if self.first_processed else None,
            "last_processed": self.last_processed.isoformat() if self.last_processed else None
        }

class SystemPerformanceMetrics(Base):
    """Hourly system performance aggregations"""
    __tablename__ = "system_performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, index=True)  # Rounded to hour
    
    # Request volume metrics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    # Performance metrics
    avg_response_time = Column(Float)
    median_response_time = Column(Float)
    p95_response_time = Column(Float)
    p99_response_time = Column(Float)
    fastest_response_time = Column(Float)
    slowest_response_time = Column(Float)
    
    # Quality metrics
    avg_relevance_score = Column(Float)
    documents_processed = Column(Integer, default=0)
    unique_documents = Column(Integer, default=0)
    duplicate_requests = Column(Integer, default=0)
    
    # Resource usage
    total_openai_tokens = Column(Integer, default=0)
    total_embedding_calls = Column(Integer, default=0)
    total_chat_calls = Column(Integer, default=0)
    total_estimated_cost = Column(Float, default=0.0)
    
    # Error tracking
    error_breakdown = Column(JSONB)  # {"embedding_errors": 5, "download_errors": 2}
    warning_counts = Column(JSONB)
    
    # Document type distribution
    document_types = Column(JSONB)  # {"pdf": 45, "docx": 12, "txt": 3}
    processing_modes = Column(JSONB)  # {"no_cache_maximum_accuracy": 60}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON responses"""
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "total_requests": self.total_requests,
            "success_rate": success_rate,
            "avg_response_time": self.avg_response_time,
            "p95_response_time": self.p95_response_time,
            "avg_relevance_score": self.avg_relevance_score,
            "total_openai_tokens": self.total_openai_tokens,
            "total_estimated_cost": self.total_estimated_cost,
            "document_types": self.document_types,
            "error_breakdown": self.error_breakdown
        }

class QuestionAnalytics(Base):
    """Analytics for question patterns and answer quality"""
    __tablename__ = "question_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_hash = Column(String(64), index=True)  # Hash of normalized question
    question_text = Column(Text)
    question_type = Column(String(100))  # What, How, When, Where, Why, etc.
    question_category = Column(String(100))  # insurance, coverage, premium, etc.
    
    # Answer quality metrics
    times_asked = Column(Integer, default=1)
    avg_relevance_score = Column(Float)
    avg_answer_length = Column(Integer)
    avg_processing_time = Column(Float)
    
    # Answer consistency
    unique_answers_count = Column(Integer, default=1)
    most_common_answer = Column(Text)
    answer_variants = Column(JSONB)
    
    # Performance tracking
    best_relevance_score = Column(Float)
    fastest_answer_time = Column(Float)
    
    # Timestamps
    first_asked = Column(DateTime, default=datetime.utcnow)
    last_asked = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON responses"""
        return {
            "question_type": self.question_type,
            "question_category": self.question_category,
            "times_asked": self.times_asked,
            "avg_relevance_score": self.avg_relevance_score,
            "avg_processing_time": self.avg_processing_time,
            "answer_consistency": self.unique_answers_count,
            "first_asked": self.first_asked.isoformat() if self.first_asked else None,
            "last_asked": self.last_asked.isoformat() if self.last_asked else None
        }

class SystemHealth(Base):
    """System health monitoring and alerts"""
    __tablename__ = "system_health"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Component health status
    api_status = Column(String(20), default="healthy")  # healthy, degraded, down
    database_status = Column(String(20), default="healthy")
    openai_status = Column(String(20), default="healthy")
    pinecone_status = Column(String(20), default="healthy")
    
    # Performance indicators
    avg_response_time_5min = Column(Float)
    error_rate_5min = Column(Float)
    throughput_requests_per_min = Column(Float)
    
    # Resource utilization
    memory_usage_percent = Column(Float)
    cpu_usage_percent = Column(Float)
    disk_usage_percent = Column(Float)
    
    # Alerts and issues
    active_alerts = Column(JSONB)
    resolved_alerts = Column(JSONB)
    
    # Maintenance flags
    maintenance_mode = Column(Boolean, default=False)
    scheduled_maintenance = Column(DateTime)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON responses"""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "api_status": self.api_status,
            "database_status": self.database_status,
            "openai_status": self.openai_status,
            "pinecone_status": self.pinecone_status,
            "avg_response_time_5min": self.avg_response_time_5min,
            "error_rate_5min": self.error_rate_5min,
            "throughput_requests_per_min": self.throughput_requests_per_min,
            "active_alerts": self.active_alerts,
            "maintenance_mode": self.maintenance_mode
        }
