"""
Enhanced Database Service for HackRX 6.0 Project
Comprehensive database operations with advanced analytics and monitoring
"""

import os
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv

load_dotenv()

# Try to import SQLAlchemy components
try:
    from sqlalchemy import create_engine, func, desc, asc, text
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.exc import SQLAlchemyError
    from database.enhanced_models import (
        Base, EnhancedQuerySession, DocumentProcessingStats, 
        SystemPerformanceMetrics, QuestionAnalytics, SystemHealth
    )
    POSTGRES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PostgreSQL dependencies not available: {e}")
    print("📝 Running in fallback mode without PostgreSQL")
    POSTGRES_AVAILABLE = False

class EnhancedDatabaseService:
    """Enhanced database service with comprehensive analytics and monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.postgres_enabled = POSTGRES_AVAILABLE and self._initialize_database()
        
        if self.postgres_enabled:
            self.logger.info("✅ Enhanced PostgreSQL database service initialized")
        else:
            self.logger.warning("⚠️ PostgreSQL not available, using fallback mode")
    
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
                db.execute(text("SELECT 1"))
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    def log_enhanced_query_session(
        self,
        document_url: str,
        questions: List[str],
        answers: List[str],
        performance_metrics: Dict,
        quality_metrics: Dict,
        processing_details: Dict,
        request_info: Dict = None
    ) -> Optional[str]:
        """Log comprehensive query session with enhanced metrics"""
        
        if not self.postgres_enabled:
            self._fallback_log_session(document_url, questions, answers, performance_metrics)
            return None
        
        session_id = f"session_{int(time.time())}_{hashlib.md5(document_url.encode()).hexdigest()[:8]}"
        
        try:
            with self.SessionLocal() as db:
                # Calculate document hash for analytics
                document_hash = hashlib.sha256(document_url.encode()).hexdigest()
                
                # Create enhanced session record
                session_record = EnhancedQuerySession(
                    session_id=session_id,
                    document_url=document_url,
                    document_type=self._detect_document_type(document_url),
                    document_size_bytes=processing_details.get("document_size"),
                    document_hash=document_hash,
                    document_name=self._extract_document_name(document_url),
                    
                    # Question and answer data
                    questions=questions,
                    answers=answers,
                    question_count=len(questions),
                    
                    # Processing configuration
                    processing_mode=processing_details.get("mode", "no_cache_maximum_accuracy"),
                    pipeline_version=processing_details.get("pipeline_version", "v3.0_NO_CACHE"),
                    
                    # Performance metrics (all times in seconds)
                    total_processing_time=performance_metrics.get("total_time", 0),
                    document_download_time=performance_metrics.get("download_time", 0),
                    document_parsing_time=performance_metrics.get("parsing_time", 0),
                    text_cleaning_time=performance_metrics.get("cleaning_time", 0),
                    chunking_time=performance_metrics.get("chunking_time", 0),
                    embedding_generation_time=performance_metrics.get("embedding_time", 0),
                    vector_store_time=performance_metrics.get("vector_store_time", 0),
                    vector_search_time=performance_metrics.get("search_time", 0),
                    answer_generation_time=performance_metrics.get("answer_time", 0),
                    
                    # Quality metrics
                    chunks_generated=quality_metrics.get("chunks_count", 0),
                    embeddings_created=quality_metrics.get("embeddings_count", 0),
                    successful_embeddings=quality_metrics.get("successful_embeddings", 0),
                    average_relevance_score=quality_metrics.get("avg_relevance", 0),
                    max_relevance_score=quality_metrics.get("max_relevance", 0),
                    min_relevance_score=quality_metrics.get("min_relevance", 0),
                    questions_answered=len([a for a in answers if a and a.strip()]),
                    successful_answers=quality_metrics.get("successful_answers", len(answers)),
                    
                    # API usage and cost tracking
                    openai_tokens_used=processing_details.get("tokens_used", 0),
                    openai_api_calls=processing_details.get("total_api_calls", 0),
                    embedding_api_calls=processing_details.get("embedding_calls", 0),
                    chat_completion_calls=processing_details.get("chat_calls", 0),
                    estimated_cost_usd=processing_details.get("estimated_cost", 0),
                    
                    # Error handling
                    processing_errors=processing_details.get("errors", []),
                    warnings=processing_details.get("warnings", []),
                    success_status=processing_details.get("success", True),
                    error_stage=processing_details.get("error_stage"),
                    
                    # Request metadata
                    user_agent=request_info.get("user_agent") if request_info else None,
                    ip_address=request_info.get("ip_address") if request_info else None,
                    request_source=request_info.get("source", "api") if request_info else "api",
                    
                    # Timestamps
                    started_at=datetime.fromtimestamp(processing_details.get("start_time", time.time())),
                    completed_at=datetime.utcnow(),
                    
                    # Analytics
                    is_duplicate_document=self._check_duplicate_document(db, document_hash),
                    cache_hits=processing_details.get("cache_hits", 0),
                    cache_misses=processing_details.get("cache_misses", 0)
                )
                
                db.add(session_record)
                db.commit()
                
                # Update aggregated statistics
                self._update_document_stats(db, document_hash, document_url, session_record)
                self._update_system_metrics(db, session_record)
                self._update_question_analytics(db, questions, answers, quality_metrics)
                
                self.logger.info(f"✅ Enhanced session logged: {session_id}")
                return session_id
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Enhanced database logging failed: {e}")
            return None
    
    def get_document_insights(self, document_url: str) -> Dict:
        """Get comprehensive insights about document processing history"""
        if not self.postgres_enabled:
            return {"message": "PostgreSQL not available - using fallback mode"}
        
        try:
            with self.SessionLocal() as db:
                document_hash = hashlib.sha256(document_url.encode()).hexdigest()
                
                # Get processing stats
                stats = db.query(DocumentProcessingStats).filter(
                    DocumentProcessingStats.document_hash == document_hash
                ).first()
                
                if not stats:
                    return {"message": "No processing history found for this document"}
                
                # Get recent sessions
                recent_sessions = db.query(EnhancedQuerySession).filter(
                    EnhancedQuerySession.document_hash == document_hash
                ).order_by(desc(EnhancedQuerySession.created_at)).limit(10).all()
                
                # Calculate trends
                performance_trend = self._calculate_performance_trend(recent_sessions)
                
                return {
                    "document_hash": document_hash,
                    "document_type": stats.document_type,
                    "total_requests": stats.total_requests,
                    "successful_requests": stats.successful_requests,
                    "success_rate": (stats.successful_requests / stats.total_requests * 100) if stats.total_requests > 0 else 0,
                    "avg_processing_time": stats.avg_processing_time,
                    "avg_relevance_score": stats.avg_relevance_score,
                    "total_cost_usd": stats.total_cost_usd,
                    "avg_cost_per_session": stats.avg_cost_per_session,
                    "first_processed": stats.first_processed.isoformat(),
                    "last_processed": stats.last_processed.isoformat(),
                    "common_question_types": stats.common_question_types,
                    "processing_modes_used": stats.processing_modes_used,
                    "recent_sessions_count": len(recent_sessions),
                    "performance_trend": performance_trend,
                    "best_relevance_score": stats.best_relevance_score,
                    "fastest_processing_time": stats.fastest_processing_time
                }
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Failed to get document insights: {e}")
            return {"error": str(e)}
    
    def get_system_analytics(self, hours: int = 24) -> Dict:
        """Get comprehensive system analytics for specified time period"""
        if not self.postgres_enabled:
            return {"message": "PostgreSQL not available - using fallback mode"}
        
        try:
            with self.SessionLocal() as db:
                since = datetime.utcnow() - timedelta(hours=hours)
                
                # Basic session statistics
                total_sessions = db.query(EnhancedQuerySession).filter(
                    EnhancedQuerySession.created_at >= since
                ).count()
                
                successful_sessions = db.query(EnhancedQuerySession).filter(
                    EnhancedQuerySession.created_at >= since,
                    EnhancedQuerySession.success_status == True
                ).count()
                
                # Performance metrics aggregation
                performance_stats = db.query(
                    func.avg(EnhancedQuerySession.total_processing_time).label('avg_time'),
                    func.avg(EnhancedQuerySession.average_relevance_score).label('avg_relevance'),
                    func.sum(EnhancedQuerySession.openai_tokens_used).label('total_tokens'),
                    func.sum(EnhancedQuerySession.estimated_cost_usd).label('total_cost'),
                    func.avg(EnhancedQuerySession.chunks_generated).label('avg_chunks'),
                    func.avg(EnhancedQuerySession.question_count).label('avg_questions')
                ).filter(
                    EnhancedQuerySession.created_at >= since,
                    EnhancedQuerySession.success_status == True
                ).first()
                
                # Document type distribution
                doc_types = db.query(
                    EnhancedQuerySession.document_type,
                    func.count(EnhancedQuerySession.id).label('count')
                ).filter(
                    EnhancedQuerySession.created_at >= since
                ).group_by(EnhancedQuerySession.document_type).all()
                
                # Processing mode distribution
                processing_modes = db.query(
                    EnhancedQuerySession.processing_mode,
                    func.count(EnhancedQuerySession.id).label('count')
                ).filter(
                    EnhancedQuerySession.created_at >= since
                ).group_by(EnhancedQuerySession.processing_mode).all()
                
                # Error analysis
                error_sessions = db.query(EnhancedQuerySession).filter(
                    EnhancedQuerySession.created_at >= since,
                    EnhancedQuerySession.success_status == False
                ).all()
                
                error_breakdown = {}
                for session in error_sessions:
                    stage = session.error_stage or "unknown"
                    error_breakdown[stage] = error_breakdown.get(stage, 0) + 1
                
                # Hourly distribution
                hourly_stats = self._get_hourly_distribution(db, since)
                
                return {
                    "time_period_hours": hours,
                    "analysis_period": {
                        "start": since.isoformat(),
                        "end": datetime.utcnow().isoformat()
                    },
                    "request_summary": {
                        "total_requests": total_sessions,
                        "successful_requests": successful_sessions,
                        "failed_requests": total_sessions - successful_sessions,
                        "success_rate": (successful_sessions / total_sessions * 100) if total_sessions > 0 else 0
                    },
                    "performance_metrics": {
                        "avg_processing_time": float(performance_stats.avg_time or 0),
                        "avg_relevance_score": float(performance_stats.avg_relevance or 0),
                        "avg_chunks_per_document": float(performance_stats.avg_chunks or 0),
                        "avg_questions_per_session": float(performance_stats.avg_questions or 0)
                    },
                    "cost_analysis": {
                        "total_tokens_used": int(performance_stats.total_tokens or 0),
                        "total_estimated_cost": float(performance_stats.total_cost or 0),
                        "avg_cost_per_session": float(performance_stats.total_cost or 0) / max(successful_sessions, 1)
                    },
                    "document_distribution": {
                        "types": {doc_type: count for doc_type, count in doc_types},
                        "unique_documents": db.query(EnhancedQuerySession.document_hash).filter(
                            EnhancedQuerySession.created_at >= since
                        ).distinct().count()
                    },
                    "processing_modes": {mode: count for mode, count in processing_modes},
                    "error_analysis": {
                        "error_breakdown": error_breakdown,
                        "error_rate": ((total_sessions - successful_sessions) / total_sessions * 100) if total_sessions > 0 else 0
                    },
                    "hourly_distribution": hourly_stats
                }
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Failed to get system analytics: {e}")
            return {"error": str(e)}
    
    def get_question_analytics(self) -> Dict:
        """Get analytics about question patterns and answer quality"""
        if not self.postgres_enabled:
            return {"message": "PostgreSQL not available - using fallback mode"}
        
        try:
            with self.SessionLocal() as db:
                # Top question categories
                top_categories = db.query(
                    QuestionAnalytics.question_category,
                    func.sum(QuestionAnalytics.times_asked).label('total_asked'),
                    func.avg(QuestionAnalytics.avg_relevance_score).label('avg_relevance')
                ).group_by(QuestionAnalytics.question_category).order_by(
                    desc('total_asked')
                ).limit(10).all()
                
                # Question types distribution
                question_types = db.query(
                    QuestionAnalytics.question_type,
                    func.count(QuestionAnalytics.id).label('count'),
                    func.avg(QuestionAnalytics.avg_relevance_score).label('avg_relevance')
                ).group_by(QuestionAnalytics.question_type).all()
                
                # Answer quality metrics
                quality_stats = db.query(
                    func.avg(QuestionAnalytics.avg_relevance_score).label('overall_avg_relevance'),
                    func.avg(QuestionAnalytics.avg_processing_time).label('overall_avg_time'),
                    func.sum(QuestionAnalytics.times_asked).label('total_questions')
                ).first()
                
                return {
                    "top_question_categories": [
                        {
                            "category": cat,
                            "times_asked": total,
                            "avg_relevance": float(avg_rel or 0)
                        }
                        for cat, total, avg_rel in top_categories
                    ],
                    "question_type_distribution": [
                        {
                            "type": qtype,
                            "count": count,
                            "avg_relevance": float(avg_rel or 0)
                        }
                        for qtype, count, avg_rel in question_types
                    ],
                    "overall_quality": {
                        "avg_relevance_score": float(quality_stats.overall_avg_relevance or 0),
                        "avg_processing_time": float(quality_stats.overall_avg_time or 0),
                        "total_questions_analyzed": int(quality_stats.total_questions or 0)
                    }
                }
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Failed to get question analytics: {e}")
            return {"error": str(e)}
    
    def get_system_health(self) -> Dict:
        """Get current system health status"""
        if not self.postgres_enabled:
            return {
                "overall_status": "degraded",
                "database_status": "unavailable",
                "message": "PostgreSQL not available - using fallback mode"
            }
        
        try:
            with self.SessionLocal() as db:
                # Get latest health record
                latest_health = db.query(SystemHealth).order_by(
                    desc(SystemHealth.timestamp)
                ).first()
                
                # Calculate current metrics from recent sessions
                recent_sessions = db.query(EnhancedQuerySession).filter(
                    EnhancedQuerySession.created_at >= datetime.utcnow() - timedelta(minutes=5)
                ).all()
                
                # Calculate current performance indicators
                current_metrics = self._calculate_current_metrics(recent_sessions)
                
                if latest_health:
                    health_data = latest_health.to_dict()
                else:
                    health_data = {
                        "api_status": "healthy",
                        "database_status": "healthy",
                        "openai_status": "healthy",
                        "pinecone_status": "healthy"
                    }
                
                # Add current metrics
                health_data.update(current_metrics)
                health_data["overall_status"] = self._determine_overall_status(health_data)
                
                return health_data
                
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Failed to get system health: {e}")
            return {
                "overall_status": "degraded",
                "database_status": "error",
                "error": str(e)
            }
    
    def _fallback_log_session(self, document_url: str, questions: List[str], 
                             answers: List[str], performance_metrics: Dict):
        """Fallback logging when PostgreSQL is not available"""
        self.logger.info(f"📝 [FALLBACK] Session processed:")
        self.logger.info(f"  Document: {document_url}")
        self.logger.info(f"  Questions: {len(questions)}")
        self.logger.info(f"  Processing time: {performance_metrics.get('total_time', 0):.2f}s")
        self.logger.info(f"  Mode: NO_CACHE_MAXIMUM_ACCURACY")
    
    def _detect_document_type(self, document_url: str) -> str:
        """Detect document type from URL"""
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
    
    def _extract_document_name(self, document_url: str) -> str:
        """Extract document name from URL"""
        try:
            return document_url.split('/')[-1]
        except:
            return "unknown_document"
    
    def _check_duplicate_document(self, db: Session, document_hash: str) -> bool:
        """Check if document has been processed before"""
        existing = db.query(DocumentProcessingStats).filter(
            DocumentProcessingStats.document_hash == document_hash
        ).first()
        return existing is not None
    
    def _update_document_stats(self, db: Session, document_hash: str, 
                              document_url: str, session: EnhancedQuerySession):
        """Update document processing statistics"""
        stats = db.query(DocumentProcessingStats).filter(
            DocumentProcessingStats.document_hash == document_hash
        ).first()
        
        if not stats:
            stats = DocumentProcessingStats(
                document_hash=document_hash,
                document_url=document_url,
                document_type=session.document_type,
                document_name=session.document_name
            )
            db.add(stats)
        
        # Update statistics
        stats.total_requests += 1
        if session.success_status:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
        
        # Update running averages
        total_successful = stats.successful_requests
        if total_successful > 0:
            # Processing time average
            if stats.avg_processing_time:
                stats.avg_processing_time = (
                    (stats.avg_processing_time * (total_successful - 1) + session.total_processing_time) 
                    / total_successful
                )
            else:
                stats.avg_processing_time = session.total_processing_time
            
            # Relevance score average
            if session.average_relevance_score and session.average_relevance_score > 0:
                if stats.avg_relevance_score:
                    stats.avg_relevance_score = (
                        (stats.avg_relevance_score * (total_successful - 1) + session.average_relevance_score) 
                        / total_successful
                    )
                else:
                    stats.avg_relevance_score = session.average_relevance_score
        
        # Update cost tracking
        stats.total_tokens_used += session.openai_tokens_used or 0
        stats.total_cost_usd += session.estimated_cost_usd or 0
        if stats.successful_requests > 0:
            stats.avg_cost_per_session = stats.total_cost_usd / stats.successful_requests
        
        # Update performance records
        if session.average_relevance_score:
            if not stats.best_relevance_score or session.average_relevance_score > stats.best_relevance_score:
                stats.best_relevance_score = session.average_relevance_score
        
        if session.total_processing_time:
            if not stats.fastest_processing_time or session.total_processing_time < stats.fastest_processing_time:
                stats.fastest_processing_time = session.total_processing_time
        
        stats.last_processed = datetime.utcnow()
        db.commit()
    
    def _update_system_metrics(self, db: Session, session: EnhancedQuerySession):
        """Update hourly system performance metrics"""
        current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        
        metrics = db.query(SystemPerformanceMetrics).filter(
            SystemPerformanceMetrics.timestamp == current_hour
        ).first()
        
        if not metrics:
            metrics = SystemPerformanceMetrics(timestamp=current_hour)
            db.add(metrics)
        
        # Update request counts
        metrics.total_requests += 1
        if session.success_status:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # Update resource usage
        metrics.total_openai_tokens += session.openai_tokens_used or 0
        metrics.total_embedding_calls += session.embedding_api_calls or 0
        metrics.total_chat_calls += session.chat_completion_calls or 0
        metrics.total_estimated_cost += session.estimated_cost_usd or 0
        
        db.commit()
    
    def _update_question_analytics(self, db: Session, questions: List[str], 
                                  answers: List[str], quality_metrics: Dict):
        """Update question analytics for pattern recognition"""
        for i, question in enumerate(questions):
            question_hash = hashlib.sha256(question.lower().encode()).hexdigest()
            
            analytics = db.query(QuestionAnalytics).filter(
                QuestionAnalytics.question_hash == question_hash
            ).first()
            
            if not analytics:
                analytics = QuestionAnalytics(
                    question_hash=question_hash,
                    question_text=question,
                    question_type=self._classify_question_type(question),
                    question_category=self._classify_question_category(question)
                )
                db.add(analytics)
            
            analytics.times_asked += 1
            analytics.last_asked = datetime.utcnow()
            
        db.commit()
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question by type (What, How, When, etc.)"""
        question_lower = question.lower().strip()
        if question_lower.startswith(('what', 'what is', 'what are')):
            return 'what'
        elif question_lower.startswith(('how', 'how to', 'how much', 'how many')):
            return 'how'
        elif question_lower.startswith(('when', 'what time')):
            return 'when'
        elif question_lower.startswith(('where', 'which')):
            return 'where'
        elif question_lower.startswith(('why', 'why is')):
            return 'why'
        elif question_lower.startswith(('who', 'whose')):
            return 'who'
        elif question_lower.startswith(('can', 'could', 'may', 'might')):
            return 'can'
        elif question_lower.startswith(('is', 'are', 'was', 'were', 'will', 'would')):
            return 'yes_no'
        else:
            return 'other'
    
    def _classify_question_category(self, question: str) -> str:
        """Classify question by category (insurance domain)"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['premium', 'payment', 'cost', 'fee', 'price']):
            return 'premium'
        elif any(word in question_lower for word in ['coverage', 'cover', 'benefit', 'include']):
            return 'coverage'
        elif any(word in question_lower for word in ['claim', 'process', 'procedure', 'submit']):
            return 'claims'
        elif any(word in question_lower for word in ['exclude', 'exclusion', 'not covered']):
            return 'exclusions'
        elif any(word in question_lower for word in ['age', 'eligible', 'qualification']):
            return 'eligibility'
        elif any(word in question_lower for word in ['policy', 'document', 'terms']):
            return 'policy_terms'
        elif any(word in question_lower for word in ['medical', 'health', 'treatment', 'hospital']):
            return 'medical'
        else:
            return 'general'
    
    def _calculate_performance_trend(self, sessions: List[EnhancedQuerySession]) -> Dict:
        """Calculate performance trends from recent sessions"""
        if not sessions:
            return {"trend": "no_data"}
        
        # Sort by timestamp
        sessions.sort(key=lambda x: x.created_at)
        
        # Calculate trends
        times = [s.total_processing_time for s in sessions if s.total_processing_time]
        relevance_scores = [s.average_relevance_score for s in sessions if s.average_relevance_score]
        
        trend_data = {
            "session_count": len(sessions),
            "time_range": {
                "earliest": sessions[0].created_at.isoformat(),
                "latest": sessions[-1].created_at.isoformat()
            }
        }
        
        if times:
            trend_data["processing_time"] = {
                "average": sum(times) / len(times),
                "trend": "improving" if len(times) > 1 and times[-1] < times[0] else "stable"
            }
        
        if relevance_scores:
            trend_data["relevance_score"] = {
                "average": sum(relevance_scores) / len(relevance_scores),
                "trend": "improving" if len(relevance_scores) > 1 and relevance_scores[-1] > relevance_scores[0] else "stable"
            }
        
        return trend_data
    
    def _get_hourly_distribution(self, db: Session, since: datetime) -> List[Dict]:
        """Get hourly request distribution"""
        hourly_data = db.query(
            func.date_trunc('hour', EnhancedQuerySession.created_at).label('hour'),
            func.count(EnhancedQuerySession.id).label('requests'),
            func.sum(func.case([(EnhancedQuerySession.success_status == True, 1)], else_=0)).label('successful')
        ).filter(
            EnhancedQuerySession.created_at >= since
        ).group_by('hour').order_by('hour').all()
        
        return [
            {
                "hour": hour.isoformat(),
                "total_requests": requests,
                "successful_requests": successful,
                "success_rate": (successful / requests * 100) if requests > 0 else 0
            }
            for hour, requests, successful in hourly_data
        ]
    
    def _calculate_current_metrics(self, recent_sessions: List[EnhancedQuerySession]) -> Dict:
        """Calculate current performance metrics from recent sessions"""
        if not recent_sessions:
            return {
                "current_throughput": 0,
                "current_avg_response_time": 0,
                "current_error_rate": 0
            }
        
        successful_sessions = [s for s in recent_sessions if s.success_status]
        
        # Calculate metrics
        throughput = len(recent_sessions)  # requests per 5 minutes
        avg_response_time = 0
        if successful_sessions:
            response_times = [s.total_processing_time for s in successful_sessions if s.total_processing_time]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
        
        error_rate = ((len(recent_sessions) - len(successful_sessions)) / len(recent_sessions) * 100) if recent_sessions else 0
        
        return {
            "current_throughput": throughput,
            "current_avg_response_time": avg_response_time,
            "current_error_rate": error_rate,
            "active_sessions_5min": len(recent_sessions)
        }
    
    def _determine_overall_status(self, health_data: Dict) -> str:
        """Determine overall system status from component statuses"""
        statuses = [
            health_data.get("api_status", "unknown"),
            health_data.get("database_status", "unknown"),
            health_data.get("openai_status", "unknown"),
            health_data.get("pinecone_status", "unknown")
        ]
        
        if "down" in statuses:
            return "down"
        elif "degraded" in statuses or health_data.get("current_error_rate", 0) > 10:
            return "degraded"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"

# Create singleton instance
try:
    enhanced_db_service = EnhancedDatabaseService()
except Exception as e:
    print(f"⚠️ Failed to initialize enhanced database service: {e}")
    enhanced_db_service = None
