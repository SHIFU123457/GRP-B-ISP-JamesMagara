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

def initialize_database():
    """Initialize database tables"""
    try:
        db_manager.create_tables()
        logging.info("Database initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        return False

def main():
    """Main application function"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Study Helper Agent...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Initialize database
    if not initialize_database():
        logger.error("Failed to initialize database. Exiting.")
        sys.exit(1)
    
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
        logger.info("Study Helper Agent stopped")

if __name__ == "__main__":
    main()