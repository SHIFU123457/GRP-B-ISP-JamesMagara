from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging

from config.settings import settings
from src.data.models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager following singleton pattern"""
    
    _instance = None
    _engine = None
    _SessionLocal = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize database connection"""
        try:
            # Create engine with connection pooling
            self._engine = create_engine(
                settings.DATABASE_URL,
                echo=settings.DEBUG,  # Log SQL queries in debug mode
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=300,    # Recycle connections every 5 minutes
            )
            
            # Create session factory
            self._SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self._engine
            )
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self._engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def get_engine(self):
        """Get database engine"""
        return self._engine
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self._SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session_sync(self) -> Session:
        """Get database session for synchronous operations"""
        return self._SessionLocal()

# Global database manager instance
db_manager = DatabaseManager()

# Dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """Database dependency for FastAPI"""
    with db_manager.get_session() as session:
        yield session
