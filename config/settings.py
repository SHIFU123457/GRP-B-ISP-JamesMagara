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
    GOOGLE_CLASSROOM_CREDENTIALS: Optional[str] = os.getenv("GOOGLE_CLASSROOM_CREDENTIALS")
    
    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")
    
    # Personalization Settings
    MIN_INTERACTIONS_FOR_PERSONALIZATION: int = int(os.getenv("MIN_INTERACTIONS_FOR_PERSONALIZATION", "5"))
    LEARNING_STYLE_UPDATE_INTERVAL: int = int(os.getenv("LEARNING_STYLE_UPDATE_INTERVAL", "24"))
    
    # Validation
    def validate(self) -> bool:
        """Validate required settings"""
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        return True

# Global settings instance
settings = Settings()
settings.validate()