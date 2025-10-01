import asyncio
import logging
import sys
import signal
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.bot.handlers import StudyHelperBot
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
            # Check if per-user OAuth is enabled (Google Classroom)
            if getattr(settings, 'PER_USER_GOOGLE_OAUTH', True):
                logging.info("Per-user Google Classroom OAuth enabled - LMS will work per-user basis")
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

def setup_signal_handlers(bot, scheduler_service):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        try:
            if scheduler_service:
                scheduler_service.stop()
                logger.info("Scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")

        try:
            if bot:
                bot.stop()
                logger.info("Bot stopped")
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")

        logger.info("Graceful shutdown completed")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def validate_environment():
    """Validate environment and configuration"""
    logger = logging.getLogger(__name__)

    # Check critical environment variables
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not configured")
        return False

    if not settings.DATABASE_URL:
        logger.error("DATABASE_URL not configured")
        return False

    # Check vector store directory
    vector_store_path = Path(settings.VECTOR_STORE_PATH)
    try:
        vector_store_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Vector store directory ready: {vector_store_path}")
    except Exception as e:
        logger.error(f"Cannot create vector store directory: {e}")
        return False

    logger.info("✅ Environment validation passed")
    return True

def main():
    """Main application function with improved error handling"""
    bot = None

    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)

        logger.info("🚀 Starting Study Helper Agent...")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Debug mode: {settings.DEBUG}")

        # Validate environment
        if not validate_environment():
            logger.error("❌ Environment validation failed. Exiting.")
            sys.exit(1)

        # Test db connectivity
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
        bot = StudyHelperBot()
        logger.info("✅ Bot initialized successfully")

        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(bot, scheduler_service)

        # Start the bot
        logger.info("🎆 Study Helper Agent is now running!")
        asyncio.run(bot.run())

    except KeyboardInterrupt:
        logger.info("❌ Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"❌ Unexpected error in main: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("💯 Performing cleanup...")
        try:
            if scheduler_service:
                scheduler_service.stop()
                logger.info("✅ Scheduler stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping scheduler: {e}")

        try:
            if bot:
                bot.stop()
                logger.info("✅ Bot stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping bot: {e}")

        logger.info("👋 Study Helper Agent stopped")

if __name__ == "__main__":
    main()