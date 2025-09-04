import logging
import logging.config
import os
from pathlib import Path
from datetime import datetime

from config.settings import settings

def setup_logging():
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    log_filename = f"study_helper_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = logs_dir / log_filename
    
    # Logging configuration
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': settings.LOG_LEVEL,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': str(log_path),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'formatter': 'detailed',
            },
            'error_file': {
                'level': 'ERROR',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': str(logs_dir / 'errors.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 3,
                'formatter': 'detailed',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': settings.LOG_LEVEL,
                'propagate': False
            },
            'telegram': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False
            },
            'sqlalchemy.engine': {
                'handlers': ['file'],
                'level': 'WARNING' if not settings.DEBUG else 'INFO',
                'propagate': False
            },
            'httpx': {
                'handlers': ['file'],
                'level': 'WARNING',
                'propagate': False
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Test logging
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    
    return logger