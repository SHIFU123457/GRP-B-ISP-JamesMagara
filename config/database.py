from sqlalchemy import create_engine, text, inspect
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
            # Create engine with connection pooling configured for Supabase (free-tier) limits
            self._engine = create_engine(
                settings.DATABASE_URL,
                echo=settings.DEBUG,  # Log SQL queries in debug mode
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=300,    # Recycle connections every 5 minutes
                pool_size=3,         # Max 3 persistent connections (Supabase has ~15 max)
                max_overflow=2,      # Allow 2 additional connections temporarily (total max: 5)
                pool_timeout=30,     # Wait up to 30s for a connection
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
    
    def migrate_schema(self):
        """Add missing columns to existing tables and create new tables"""
        try:
            inspector = inspect(self._engine)
            
            # 1. Add new columns to users table (lines 25-42 from models.py)
            if inspector.has_table('users'):
                columns = [col['name'] for col in inspector.get_columns('users')]
                
                # OAuth credentials columns
                if 'google_credentials' not in columns:
                    logger.info("Adding google_credentials column to users table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE users ADD COLUMN google_credentials TEXT"))
                    logger.info("Successfully added google_credentials column to users table")
                
                if 'moodle_credentials' not in columns:
                    logger.info("Adding moodle_credentials column to users table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE users ADD COLUMN moodle_credentials TEXT"))
                    logger.info("Successfully added moodle_credentials column to users table")
                
                # LMS connection status columns
                if 'google_classroom_connected' not in columns:
                    logger.info("Adding google_classroom_connected column to users table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE users ADD COLUMN google_classroom_connected BOOLEAN DEFAULT FALSE"))
                    logger.info("Successfully added google_classroom_connected column to users table")
                
                if 'moodle_connected' not in columns:
                    logger.info("Adding moodle_connected column to users table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE users ADD COLUMN moodle_connected BOOLEAN DEFAULT FALSE"))
                    logger.info("Successfully added moodle_connected column to users table")
                
                if 'last_oauth_refresh' not in columns:
                    logger.info("Adding last_oauth_refresh column to users table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE users ADD COLUMN last_oauth_refresh TIMESTAMP WITH TIME ZONE"))
                    logger.info("Successfully added last_oauth_refresh column to users table")
                
                # Privacy preferences columns
                if 'allow_data_collection' not in columns:
                    logger.info("Adding allow_data_collection column to users table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE users ADD COLUMN allow_data_collection BOOLEAN DEFAULT TRUE"))
                    logger.info("Successfully added allow_data_collection column to users table")
                
                if 'notification_preferences' not in columns:
                    logger.info("Adding notification_preferences column to users table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE users ADD COLUMN notification_preferences JSON"))
                    logger.info("Successfully added notification_preferences column to users table")
            
            # 2. Create user_lms_connections table (lines 53-77 from models.py)
            if not inspector.has_table('user_lms_connections'):
                logger.info("Creating user_lms_connections table")
                with self._engine.begin() as conn:
                    conn.execute(text("""
                        CREATE TABLE user_lms_connections (
                            id SERIAL NOT NULL,
                            user_id INTEGER NOT NULL,
                            lms_platform VARCHAR NOT NULL,
                            is_active BOOLEAN DEFAULT TRUE,
                            credentials_encrypted TEXT,
                            last_sync TIMESTAMP WITH TIME ZONE,
                            sync_status VARCHAR DEFAULT 'pending',
                            last_error TEXT,
                            error_count INTEGER DEFAULT 0,
                            connected_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                            updated_at TIMESTAMP WITH TIME ZONE,
                            PRIMARY KEY (id),
                            FOREIGN KEY(user_id) REFERENCES users (id)
                        )
                    """))
                    conn.execute(text("CREATE INDEX ix_user_lms_connections_id ON user_lms_connections (id)"))
                logger.info("Successfully created user_lms_connections table")
            
            # 3. Add is_public column to courses table
            if inspector.has_table('courses'):
                columns = [col['name'] for col in inspector.get_columns('courses')]
                
                if 'is_public' not in columns:
                    logger.info("Adding is_public column to courses table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE courses ADD COLUMN is_public BOOLEAN DEFAULT FALSE"))
                    logger.info("Successfully added is_public column to courses table")
            
            # 4. Add enrolled_via_platform column to course_enrollments table
            if inspector.has_table('course_enrollments'):
                columns = [col['name'] for col in inspector.get_columns('course_enrollments')]
                
                if 'enrolled_via_platform' not in columns:
                    logger.info("Adding enrolled_via_platform column to course_enrollments table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE course_enrollments ADD COLUMN enrolled_via_platform VARCHAR"))
                    logger.info("Successfully added enrolled_via_platform column to course_enrollments table")
            
            # 5. Add privacy and access control columns to documents table (lines 144-146)
            if inspector.has_table('documents'):
                columns = [col['name'] for col in inspector.get_columns('documents')]
                
                if 'is_public' not in columns:
                    logger.info("Adding is_public column to documents table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE documents ADD COLUMN is_public BOOLEAN DEFAULT FALSE"))
                    logger.info("Successfully added is_public column to documents table")
                
                if 'access_level' not in columns:
                    logger.info("Adding access_level column to documents table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE documents ADD COLUMN access_level VARCHAR DEFAULT 'enrolled'"))
                    logger.info("Successfully added access_level column to documents table")

                # Notification tracking columns (Phase 1 - Hybrid Approach)
                if 'notification_sent' not in columns:
                    logger.info("Adding notification_sent column to documents table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE documents ADD COLUMN notification_sent BOOLEAN DEFAULT FALSE"))
                    logger.info("Successfully added notification_sent column to documents table")

                if 'notification_sent_at' not in columns:
                    logger.info("Adding notification_sent_at column to documents table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE documents ADD COLUMN notification_sent_at TIMESTAMP WITH TIME ZONE"))
                    logger.info("Successfully added notification_sent_at column to documents table")

                if 'material_type' not in columns:
                    logger.info("Adding material_type column to documents table")
                    with self._engine.begin() as conn:
                        conn.execute(text("ALTER TABLE documents ADD COLUMN material_type VARCHAR"))
                    logger.info("Successfully added material_type column to documents table")

            logger.info("Schema migration completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to migrate schema: {e}")
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
