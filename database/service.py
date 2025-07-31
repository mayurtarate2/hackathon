"""
Database Service for HackRX 6.0 Project
Integrates PostgreSQL with the existing document processing pipeline
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Try to import SQLAlchemy components
try:
    from database.models import (
        DocumentMetadata, 
        QueryLog, 
        ProcessingStats, 
        get_db_session, 
        create_tables,
        test_connection
    )
    POSTGRES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PostgreSQL dependencies not available: {e}")
    print("📝 Running in fallback mode without PostgreSQL")
    POSTGRES_AVAILABLE = False

class DatabaseService:
    """Service for PostgreSQL database operations"""
    
    def __init__(self):
        self.postgres_enabled = POSTGRES_AVAILABLE and self._test_postgres()
        if self.postgres_enabled:
            print("✅ PostgreSQL database service initialized")
        else:
            print("⚠️ PostgreSQL not available, using fallback mode")
    
    def _test_postgres(self) -> bool:
        """Test if PostgreSQL is available and working"""
        try:
            return test_connection()
        except Exception as e:
            print(f"❌ PostgreSQL test failed: {e}")
            return False
    
    def log_document_processing(
        self,
        document_url: str,
        document_name: str = None,
        file_size: int = None,
        chunks_created: int = 0,
        processing_time: float = 0.0,
        status: str = "completed",
        error_message: str = None
    ) -> Optional[int]:
        """Log document processing to PostgreSQL"""
        
        if not self.postgres_enabled:
            # Fallback: just print the log
            print(f"📝 [FALLBACK] Document processed: {document_url}")
            print(f"  Chunks: {chunks_created}, Time: {processing_time:.2f}s, Status: {status}")
            return None
        
        try:
            db = get_db_session()
            if not db:
                return None
                
            doc_metadata = DocumentMetadata(
                document_url=document_url,
                document_name=document_name or document_url.split('/')[-1],
                file_size=file_size,
                mime_type="application/pdf",
                processing_status=status,
                chunks_created=chunks_created,
                processing_time=processing_time,
                error_message=error_message
            )
            
            db.add(doc_metadata)
            db.commit()
            doc_id = doc_metadata.id
            db.close()
            
            print(f"✅ Logged document processing to PostgreSQL (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            print(f"❌ Error logging to PostgreSQL: {e}")
            return None
    
    def get_document_by_url(self, document_url: str):
        """Check if document already exists and processed"""
        if not self.postgres_enabled:
            return None
            
        try:
            db = get_db_session()
            doc = db.query(DocumentMetadata).filter(
                DocumentMetadata.document_url == document_url,
                DocumentMetadata.processing_status == "completed"
            ).first()
            db.close()
            
            return doc
            
        except Exception as e:
            print(f"❌ Error checking existing document: {e}")
            return None
    
    def log_query_session(
        self,
        document_id: int,
        questions: List[str],
        answers: List[str],
        response_time: float = 0.0,
        user_session: str = None,
        confidence_scores: List[str] = None
    ) -> Optional[int]:
        """Log question-answer session to PostgreSQL"""
        
        if not self.postgres_enabled:
            # Fallback: just print the log
            print(f"📝 [FALLBACK] Query session logged:")
            print(f"  Questions: {len(questions)}, Response time: {response_time:.2f}s")
            return None
        
        try:
            db = get_db_session()
            if not db:
                return None
                
            query_log = QueryLog(
                document_id=document_id or 0,  # Use 0 for unknown document
                questions=questions,
                answers=answers,
                response_time=response_time,
                user_session=user_session,
                confidence_scores=confidence_scores
            )
            
            db.add(query_log)
            db.commit()
            query_id = query_log.id
            db.close()
            
            print(f"✅ Logged query session to PostgreSQL (ID: {query_id})")
            return query_id
            
        except Exception as e:
            print(f"❌ Error logging query to PostgreSQL: {e}")
            return None
    
    def get_document_history(self, limit: int = 10) -> List[Dict]:
        """Get recent document processing history"""
        
        if not self.postgres_enabled:
            return [{
                "message": "PostgreSQL not available",
                "fallback_mode": True,
                "note": "Install psycopg2-binary and configure DATABASE_URL to enable PostgreSQL"
            }]
        
        try:
            db = get_db_session()
            if not db:
                return []
                
            docs = db.query(DocumentMetadata)\
                     .order_by(DocumentMetadata.upload_timestamp.desc())\
                     .limit(limit)\
                     .all()
            
            result = [doc.to_dict() for doc in docs]
            db.close()
            return result
            
        except Exception as e:
            print(f"❌ Error getting document history: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system processing statistics"""
        
        if not self.postgres_enabled:
            return {
                "postgresql_status": "unavailable",
                "message": "PostgreSQL integration disabled",
                "fallback_mode": True
            }
        
        try:
            db = get_db_session()
            if not db:
                return {"error": "Database connection failed"}
            
            # Count total documents
            total_docs = db.query(DocumentMetadata).count()
            
            # Count total queries
            total_queries = db.query(QueryLog).count()
            
            # Calculate average processing time
            avg_processing = db.query(DocumentMetadata.processing_time).filter(
                DocumentMetadata.processing_time.isnot(None)
            ).all()
            
            avg_proc_time = sum(t[0] for t in avg_processing) / len(avg_processing) if avg_processing else 0
            
            # Calculate average response time
            avg_response = db.query(QueryLog.response_time).filter(
                QueryLog.response_time.isnot(None)
            ).all()
            
            avg_resp_time = sum(t[0] for t in avg_response) / len(avg_response) if avg_response else 0
            
            db.close()
            
            return {
                "postgresql_status": "active",
                "total_documents_processed": total_docs,
                "total_queries_answered": total_queries,
                "average_processing_time": round(avg_proc_time, 2),
                "average_response_time": round(avg_resp_time, 2),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error getting system stats: {e}")
            return {"error": str(e)}
    
    def setup_database(self) -> bool:
        """Initialize PostgreSQL database tables"""
        if not self.postgres_enabled:
            print("⚠️ PostgreSQL not available for setup")
            return False
            
        try:
            return create_tables()
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            return False

# Global database service instance
db_service = DatabaseService()

def get_database_service() -> DatabaseService:
    """Get the global database service instance"""
    return db_service
