"""
Enhanced Upload and User Interaction Logging Models for HackRX 6.0 Project
Comprehensive tracking of document uploads and user interactions
"""

from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from typing import Dict, Optional

Base = declarative_base()

class DocumentUploadLog(Base):
    """
    Comprehensive logging for every document upload
    Tracks file uploads with metadata, uploader information, and processing status
    """
    __tablename__ = "document_upload_logs"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    upload_id = Column(String(100), unique=True, index=True)  # Unique upload identifier
    
    # Document information
    document_url = Column(Text, nullable=False)  # URL or cloud storage path
    document_name = Column(String(500))
    original_filename = Column(String(500))  # Original file name from upload
    file_size_bytes = Column(Integer)
    file_type = Column(String(50))  # pdf, docx, txt, email, etc.
    mime_type = Column(String(100))  # MIME type detection
    document_hash = Column(String(64), index=True)  # SHA-256 hash for duplicate detection
    
    # Upload metadata
    upload_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    upload_source = Column(String(100))  # web, api, mobile, etc.
    upload_method = Column(String(50))  # direct_upload, url_fetch, etc.
    
    # Uploader information
    uploader_id = Column(String(100), index=True)  # User ID if available
    uploader_ip = Column(String(45))  # IPv6 compatible
    uploader_user_agent = Column(String(500))
    uploader_session = Column(String(100))  # Session identifier
    
    # Cloud storage information (if applicable)
    storage_provider = Column(String(50))  # aws_s3, google_cloud, azure, local, etc.
    storage_path = Column(Text)  # Full storage path
    storage_bucket = Column(String(200))  # Bucket/container name
    storage_region = Column(String(50))  # Storage region
    
    # Processing status
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    processing_started_at = Column(DateTime)
    processing_completed_at = Column(DateTime)
    processing_duration_seconds = Column(Integer)
    
    # Processing results
    chunks_created = Column(Integer, default=0)
    embeddings_generated = Column(Integer, default=0)
    processing_errors = Column(JSONB)  # Array of error messages
    processing_warnings = Column(JSONB)  # Array of warnings
    
    # Access and security
    is_public = Column(Boolean, default=False)
    access_permissions = Column(JSONB)  # User/group access permissions
    encryption_status = Column(String(50))  # encrypted, plain, etc.
    
    # Analytics and tracking
    download_count = Column(Integer, default=0)  # How many times downloaded
    query_count = Column(Integer, default=0)  # How many times queried
    last_accessed = Column(DateTime)
    is_duplicate = Column(Boolean, default=False)  # Duplicate of existing document
    original_upload_id = Column(String(100))  # Reference to original if duplicate
    
    # Metadata and tags
    document_tags = Column(JSONB)  # User-defined tags
    document_category = Column(String(100))  # insurance, legal, hr, etc.
    document_language = Column(String(20))  # en, es, fr, etc.
    custom_metadata = Column(JSONB)  # Additional metadata
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_upload_timestamp_status', 'upload_timestamp', 'processing_status'),
        Index('idx_uploader_document', 'uploader_id', 'document_hash'),
        Index('idx_document_category_type', 'document_category', 'file_type'),
        Index('idx_storage_provider_path', 'storage_provider', 'storage_path'),
    )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON responses"""
        return {
            "upload_id": self.upload_id,
            "document_url": self.document_url,
            "document_name": self.document_name,
            "original_filename": self.original_filename,
            "file_size_bytes": self.file_size_bytes,
            "file_type": self.file_type,
            "upload_timestamp": self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            "uploader_id": self.uploader_id,
            "processing_status": self.processing_status,
            "chunks_created": self.chunks_created,
            "embeddings_generated": self.embeddings_generated,
            "download_count": self.download_count,
            "query_count": self.query_count,
            "document_category": self.document_category,
            "is_duplicate": self.is_duplicate
        }

class UserInteractionLog(Base):
    """
    Comprehensive logging for every user interaction with the model
    Tracks input/output with detailed metadata and performance metrics
    """
    __tablename__ = "user_interaction_logs"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interaction_id = Column(String(100), unique=True, index=True)
    session_id = Column(String(100), index=True)  # Groups related interactions
    
    # User information
    user_id = Column(String(100), index=True)  # User identifier if available
    user_ip = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    user_session = Column(String(100))  # Browser/app session
    user_timezone = Column(String(50))  # User's timezone
    
    # Interaction metadata
    interaction_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    interaction_type = Column(String(50), default="question_answer")  # question_answer, document_upload, etc.
    interaction_source = Column(String(100))  # web, api, mobile, cli, etc.
    
    # Input data
    user_input = Column(Text, nullable=False)  # User's question/input
    input_language = Column(String(20))  # Detected language
    input_length_chars = Column(Integer)
    input_word_count = Column(Integer)
    input_processing_time_ms = Column(Integer)  # Time to process input
    
    # Model and processing information
    model_version = Column(String(50))  # GPT-3.5-turbo, GPT-4, etc.
    pipeline_version = Column(String(50))  # v3.0_NO_CACHE, etc.
    processing_mode = Column(String(50))  # no_cache_maximum_accuracy, etc.
    
    # Document reference
    document_upload_id = Column(String(100))  # Reference to DocumentUploadLog
    document_url = Column(Text)  # Document used for this interaction
    document_chunks_used = Column(Integer)  # Number of chunks retrieved
    
    # Model output
    model_output = Column(Text, nullable=False)  # Model's response/answer
    output_language = Column(String(20))  # Output language
    output_length_chars = Column(Integer)
    output_word_count = Column(Integer)
    output_generation_time_ms = Column(Integer)  # Time to generate output
    
    # Performance metrics
    total_processing_time_ms = Column(Integer)  # Total interaction time
    embedding_time_ms = Column(Integer)  # Time for embeddings if applicable
    retrieval_time_ms = Column(Integer)  # Time for document retrieval
    llm_time_ms = Column(Integer)  # Time for LLM generation
    
    # Quality metrics
    relevance_score = Column(Integer)  # 0-100 relevance score
    confidence_score = Column(Integer)  # 0-100 confidence score
    retrieval_accuracy = Column(Integer)  # 0-100 retrieval accuracy
    answer_completeness = Column(Integer)  # 0-100 completeness score
    
    # API usage tracking
    openai_tokens_input = Column(Integer)  # Tokens in prompt
    openai_tokens_output = Column(Integer)  # Tokens in response
    openai_tokens_total = Column(Integer)  # Total tokens used
    openai_api_calls = Column(Integer)  # Number of API calls made
    estimated_cost_usd = Column(Integer)  # Cost in cents (avoid floating point)
    
    # Success and error tracking
    interaction_success = Column(Boolean, default=True)
    error_type = Column(String(100))  # Type of error if failed
    error_message = Column(Text)  # Detailed error message
    error_stage = Column(String(50))  # Where in pipeline error occurred
    retry_count = Column(Integer, default=0)  # Number of retries
    
    # Context and conversation
    conversation_turn = Column(Integer, default=1)  # Turn number in conversation
    previous_interaction_id = Column(String(100))  # Previous interaction in conversation
    context_maintained = Column(Boolean, default=False)  # Whether context was used
    
    # User feedback (if available)
    user_rating = Column(Integer)  # 1-5 user rating
    user_feedback = Column(Text)  # Textual feedback
    feedback_timestamp = Column(DateTime)
    
    # Analysis and categorization
    question_category = Column(String(100))  # insurance, legal, hr, etc.
    question_type = Column(String(50))  # what, how, when, where, why, etc.
    answer_type = Column(String(50))  # factual, procedural, explanatory, etc.
    interaction_tags = Column(JSONB)  # User-defined or auto-generated tags
    
    # Privacy and compliance
    pii_detected = Column(Boolean, default=False)  # Personally identifiable information
    data_retention_days = Column(Integer, default=365)  # How long to retain this data
    anonymized = Column(Boolean, default=False)  # Whether PII has been removed
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_interaction_timestamp_user', 'interaction_timestamp', 'user_id'),
        Index('idx_session_conversation', 'session_id', 'conversation_turn'),
        Index('idx_document_interaction', 'document_upload_id', 'interaction_timestamp'),
        Index('idx_performance_metrics', 'total_processing_time_ms', 'relevance_score'),
        Index('idx_model_version_type', 'model_version', 'interaction_type'),
        Index('idx_success_error', 'interaction_success', 'error_type'),
    )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON responses"""
        return {
            "interaction_id": self.interaction_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "interaction_timestamp": self.interaction_timestamp.isoformat() if self.interaction_timestamp else None,
            "interaction_type": self.interaction_type,
            "user_input": self.user_input,
            "model_output": self.model_output,
            "model_version": self.model_version,
            "processing_mode": self.processing_mode,
            "document_url": self.document_url,
            "total_processing_time_ms": self.total_processing_time_ms,
            "relevance_score": self.relevance_score,
            "confidence_score": self.confidence_score,
            "openai_tokens_total": self.openai_tokens_total,
            "estimated_cost_usd": self.estimated_cost_usd / 100 if self.estimated_cost_usd else 0,  # Convert cents to dollars
            "interaction_success": self.interaction_success,
            "question_category": self.question_category,
            "question_type": self.question_type,
            "user_rating": self.user_rating
        }

class ModelVersionLog(Base):
    """
    Track different model versions and their performance
    """
    __tablename__ = "model_version_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version = Column(String(50), unique=True, index=True)  # GPT-3.5-turbo-1106, etc.
    pipeline_version = Column(String(50))  # v3.0_NO_CACHE, etc.
    
    # Version information
    deployment_date = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    deprecation_date = Column(DateTime)
    
    # Performance metrics
    total_interactions = Column(Integer, default=0)
    avg_processing_time_ms = Column(Integer)
    avg_relevance_score = Column(Integer)
    avg_user_rating = Column(Integer)
    success_rate = Column(Integer)  # Percentage (0-100)
    
    # Cost metrics
    total_tokens_used = Column(Integer, default=0)
    total_cost_usd = Column(Integer, default=0)  # In cents
    avg_cost_per_interaction = Column(Integer)  # In cents
    
    # Configuration
    model_config = Column(JSONB)  # Model configuration parameters
    deployment_notes = Column(Text)
    
    def to_dict(self) -> Dict:
        return {
            "model_version": self.model_version,
            "pipeline_version": self.pipeline_version,
            "deployment_date": self.deployment_date.isoformat() if self.deployment_date else None,
            "is_active": self.is_active,
            "total_interactions": self.total_interactions,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "avg_relevance_score": self.avg_relevance_score,
            "success_rate": self.success_rate,
            "total_cost_usd": self.total_cost_usd / 100 if self.total_cost_usd else 0,
            "avg_cost_per_interaction": self.avg_cost_per_interaction / 100 if self.avg_cost_per_interaction else 0
        }
