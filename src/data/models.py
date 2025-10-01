from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    """User model representing a student"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, index=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True)
    
    # Learning preferences (will be updated by personalization engine)
    learning_style = Column(String, default="adaptive")  # visual, auditory, kinesthetic, adaptive
    difficulty_preference = Column(String, default="medium")  # easy, medium, hard
    interaction_frequency = Column(Float, default=0.0)  # interactions per day

    # OAuth credentials (encrypted JSON)
    google_credentials = Column(Text)  # Store encrypted Google OAuth credentials
    moodle_credentials = Column(Text)  # Future: for Moodle user-level auth
    
    # LMS connection status
    google_classroom_connected = Column(Boolean, default=False)
    moodle_connected = Column(Boolean, default=False)
    last_oauth_refresh = Column(DateTime(timezone=True))
    
    # Privacy preferences
    allow_data_collection = Column(Boolean, default=True)
    notification_preferences = Column(JSON, default=lambda: {
        'new_materials': True,
        'assignments': True,
        'announcements': True,
        'reminders': True
    })
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    interactions = relationship("UserInteraction", back_populates="user")
    enrollments = relationship("CourseEnrollment", back_populates="user")

class UserLMSConnection(Base):
    """Track LMS connections per user"""
    __tablename__ = "user_lms_connections"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    lms_platform = Column(String, nullable=False)  # 'google_classroom', 'moodle'
    
    # Connection details
    is_active = Column(Boolean, default=True)
    credentials_encrypted = Column(Text)  # Encrypted credentials
    last_sync = Column(DateTime(timezone=True))
    sync_status = Column(String, default="pending")  # pending, active, failed, expired
    
    # Error tracking
    last_error = Column(Text)
    error_count = Column(Integer, default=0)
    
    # Metadata
    connected_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User")

class Course(Base):
    """Course model representing academic courses"""
    __tablename__ = "courses"
    
    id = Column(Integer, primary_key=True, index=True)
    course_code = Column(String, unique=True, index=True, nullable=False)
    course_name = Column(String, nullable=False)
    description = Column(Text)
    semester = Column(String)
    year = Column(Integer)
    
    # Privacy settings
    is_public = Column(Boolean, default=False)  # Whether course materials are shared

    # LMS Integration
    lms_course_id = Column(String, index=True)  # External LMS course ID
    lms_platform = Column(String)  # moodle, google_classroom, etc.
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    documents = relationship("Document", back_populates="course")
    enrollments = relationship("CourseEnrollment", back_populates="course")

class CourseEnrollment(Base):
    """Many-to-many relationship between users and courses"""
    __tablename__ = "course_enrollments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    enrollment_date = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)

    # Enrollment source tracking
    enrolled_via_platform = Column(String)  # Which LMS this enrollment came from
    
    # Relationships
    user = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")

class Document(Base):
    """Document model for course materials"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    
    # Document details
    title = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String)  # pdf, docx, pptx, etc.
    file_size = Column(Integer)  # in bytes
    
    # Content processing
    content_text = Column(Text)  # extracted text content
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed

    # LMS Integration
    lms_document_id = Column(String, index=True)
    lms_last_modified = Column(DateTime(timezone=True))

    # Notification tracking
    notification_sent = Column(Boolean, default=False)
    notification_sent_at = Column(DateTime(timezone=True))
    material_type = Column(String)  # assignment, quiz, reading, announcement

    # Privacy and access control
    is_public = Column(Boolean, default=False)
    access_level = Column(String, default="enrolled")  # enrolled, public, restricted
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    course = relationship("Course", back_populates="documents")
    embeddings = relationship("DocumentEmbedding", back_populates="document")

class DocumentEmbedding(Base):
    """Vector embeddings for documents (for RAG pipeline)"""
    __tablename__ = "document_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Embedding details
    chunk_text = Column(Text, nullable=False)  # the text chunk that was embedded
    chunk_index = Column(Integer, nullable=False)  # position in document
    embedding_vector_id = Column(String)  # reference to vector store (FAISS)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="embeddings")

class UserInteraction(Base):
    """Track user interactions for personalization"""
    __tablename__ = "user_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Interaction details
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    interaction_type = Column(String)  # question, feedback, command
    
    # Context
    course_context = Column(String)  # which course was being discussed
    documents_referenced = Column(JSON)  # list of document IDs used in response
    
    # User feedback
    user_rating = Column(Integer)  # 1-5 rating of response quality
    was_helpful = Column(Boolean)
    
    # Performance metrics
    response_time_ms = Column(Integer)
    tokens_used = Column(Integer)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="interactions")

class SystemLog(Base):
    """System logs for monitoring and debugging"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    log_level = Column(String, nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    component = Column(String)  # bot, rag, personalization, lms_integration
    message = Column(Text, nullable=False)
    
    # Context
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    additional_data = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class PersonalizationProfile(Base):
    """Detailed personalization profile for each user"""
    __tablename__ = "personalization_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Learning analytics
    avg_session_duration = Column(Float, default=0.0)  # minutes
    preferred_response_length = Column(String, default="medium")  # short, medium, long
    question_complexity_level = Column(Float, default=0.5)  # 0-1 scale
    
    # Interaction patterns
    most_active_hours = Column(JSON)  # list of hours when user is most active
    preferred_subjects = Column(JSON)  # list of subjects user asks about most
    learning_pace = Column(String, default="medium")  # slow, medium, fast
    
    # Performance tracking
    total_interactions = Column(Integer, default=0)
    successful_interactions = Column(Integer, default=0)  # based on user ratings
    last_interaction = Column(DateTime(timezone=True))
    
    # Model data (for ML algorithms)
    feature_vector = Column(JSON)  # cached features for ML models
    last_model_update = Column(DateTime(timezone=True))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())