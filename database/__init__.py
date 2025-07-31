"""
Database Package for HackRX 6.0 Project
PostgreSQL integration for document processing metadata
"""

from .service import get_database_service, db_service

__all__ = ['get_database_service', 'db_service']
