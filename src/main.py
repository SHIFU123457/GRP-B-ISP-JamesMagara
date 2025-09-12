import asyncio
import logging
import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from bot.handlers import StudyHelperBot
from config.database import db_manager
from config.settings import settings
from utils.logger import setup_logging
from src.services.scheduler import scheduler_service
from src.services.lms_integration import lms_service

from src.data.models import User
logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize database tables"""
    try:
        db_manager.create_tables()
        logging.info("Database initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        return False

def initialize_lms_integration():
    """Initialize LMS integration"""
    try:
        platforms = lms_service.get_available_platforms()
        if platforms:
            logging.info(f"LMS integration initialized with platforms: {platforms}")
            return True
        else:
            logging.warning("No LMS platforms configured or accessible")
            return False
    except Exception as e:
        logging.error(f"Failed to initialize LMS integration: {e}")
        return False

def test_database_connection():
    """Test database connectivity"""
    try:
        with db_manager.get_session() as session:
            # Simple query test
            user_count = session.query(User).count()
            logger.info(f"Database connection successful. Users in DB: {user_count}")
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

def main():
    """Main application function"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Study Helper Agent...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    #test db connectivity
    if not test_database_connection():
        logger.error("Database connection failed. Check your database settings.")
        sys.exit(1)

    # Initialize database
    if not initialize_database():
        logger.error("Failed to initialize database. Exiting.")
        sys.exit(1)

    # Initialize LMS integration
    lms_initialized = initialize_lms_integration()
    if not lms_initialized:
        logger.warning("LMS integration failed - continuing without LMS features")
    
    # Initialize and start scheduler
    try:
        if lms_initialized:
            scheduler_service.start()
            logger.info("Scheduler service started")
        else:
            logger.info("Scheduler not started due to LMS initialization failure")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
    
    
    # Initialize and start bot
    try:
        bot = StudyHelperBot()
        logger.info("Bot initialized successfully")
        
        # Start the bot
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        try:
            scheduler_service.stop()
            logger.info("Scheduler stopped")
        except:
            pass
        logger.info("Study Helper Agent stopped")

if __name__ == "__main__":
    main()