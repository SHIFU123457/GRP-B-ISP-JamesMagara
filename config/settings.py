import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings"""
    
    # Telegram Bot Settings
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_WEBHOOK_URL: Optional[str] = os.getenv("TELEGRAM_WEBHOOK_URL")
    
    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")   
    
    # AI Model Settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "4096"))
    
    # LMS Settings
    MOODLE_BASE_URL: Optional[str] = os.getenv("MOODLE_BASE_URL")
    MOODLE_API_TOKEN: Optional[str] = os.getenv("MOODLE_API_TOKEN")
    GOOGLE_CLASSROOM_CREDENTIALS: Optional[str] = os.getenv("GOOGLE_CLASSROOM_CREDENTIALS")
    PER_USER_GOOGLE_OAUTH: bool = os.getenv("PER_USER_GOOGLE_OAUTH", "True").lower() == "true"
    OAUTH_REDIRECT_URI: Optional[str] = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8080/oauth/callback")
    
    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")
    
    # Personalization Settings
    MIN_INTERACTIONS_FOR_PERSONALIZATION: int = int(os.getenv("MIN_INTERACTIONS_FOR_PERSONALIZATION", "5"))
    LEARNING_STYLE_UPDATE_INTERVAL: int = int(os.getenv("LEARNING_STYLE_UPDATE_INTERVAL", "24"))
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))  # Conversation session timeout

    # RAG Pipeline Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "300"))  # words per chunk (optimized)
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "75"))  # overlap words (increased)
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))  # chunks to retrieve (increased for comprehensive responses)
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))  # minimum similarity (lowered for better recall)
    CONTEXT_MAX_LENGTH: int = int(os.getenv("CONTEXT_MAX_LENGTH", "4000"))  # max context length for LLM (increased)
    HIGH_QUALITY_THRESHOLD: float = float(os.getenv("HIGH_QUALITY_THRESHOLD", "0.3"))  # threshold for quality filtering
    MAX_RESPONSE_LENGTH: int = int(os.getenv("MAX_RESPONSE_LENGTH", "4000"))  # max LLM response length in characters

    # Document Processing Settings
    SUPPORTED_FILE_TYPES: list = ["pdf", "docx", "txt", "doc", "pptx", "ppt"]
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

    # Scheduler Settings
    LMS_SYNC_INTERVAL_MINUTES: int = int(os.getenv("LMS_SYNC_INTERVAL_MINUTES", "720"))
    DOCUMENT_PROCESSING_INTERVAL_MINUTES: int = int(os.getenv("DOCUMENT_PROCESSING_INTERVAL_MINUTES", "10"))
    ENABLE_NOTIFICATIONS: bool = os.getenv("ENABLE_NOTIFICATIONS", "True").lower() == "true"
    
    # API Rate Limiting
    MOODLE_API_RATE_LIMIT: int = int(os.getenv("MOODLE_API_RATE_LIMIT", "100"))  # requests per hour
    GOOGLE_API_RATE_LIMIT: int = int(os.getenv("GOOGLE_API_RATE_LIMIT", "1000"))

    # LLM Configuration
    HUGGINGFACE_API_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_API_TOKEN")
    USE_LOCAL_LLM: bool = os.getenv("USE_LOCAL_LLM", "False").lower() == "true"
    LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

    # Fallback configuration
    ENABLE_LLM_FALLBACK: bool = os.getenv("ENABLE_LLM_FALLBACK", "True").lower() == "true"
    
    # Validation
    def validate(self) -> bool:
        """Validate required settings"""
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL is required")
        
        # Check if at least one LMS is configured
        has_moodle = self.MOODLE_BASE_URL and self.MOODLE_API_TOKEN
        has_google_classroom = self.GOOGLE_CLASSROOM_CREDENTIALS
        
        if not (has_moodle or has_google_classroom):
            raise ValueError("At least one LMS must be configured (Moodle or Google Classroom)")
        
        return True

# Global settings instance
settings = Settings()
settings.validate()