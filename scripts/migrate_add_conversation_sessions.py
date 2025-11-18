"""
Database Migration Script: Add ConversationSession Table

This script adds the conversation_sessions table to support:
- Session tracking with automatic timeout
- Conversation context continuity
- Session analytics for personalization

Run this script to update your database schema.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from sqlalchemy import inspect

from config.database import db_manager
from src.data.models import Base, ConversationSession

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_table_exists(table_name: str) -> bool:
    """Check if table already exists"""
    try:
        with db_manager.get_session() as session:
            inspector = inspect(session.bind)
            return table_name in inspector.get_table_names()
    except Exception as e:
        logger.error(f"Error checking if table exists: {e}")
        return False


def create_conversation_sessions_table():
    """Create the conversation_sessions table"""
    try:
        logger.info("Checking if conversation_sessions table exists...")

        if check_table_exists('conversation_sessions'):
            logger.warning("Table 'conversation_sessions' already exists. Skipping creation.")
            return True

        logger.info("Creating conversation_sessions table...")

        # Get the engine from db_manager
        engine = db_manager.get_engine()

        # Create only the ConversationSession table
        ConversationSession.__table__.create(engine, checkfirst=True)

        logger.info("✅ Successfully created conversation_sessions table")

        # Verify creation
        if check_table_exists('conversation_sessions'):
            logger.info("✅ Verified: conversation_sessions table exists")
            return True
        else:
            logger.error("❌ Failed to verify table creation")
            return False

    except Exception as e:
        logger.error(f"❌ Error creating conversation_sessions table: {e}", exc_info=True)
        return False


def verify_table_schema():
    """Verify the table has all expected columns"""
    try:
        logger.info("Verifying table schema...")

        with db_manager.get_session() as session:
            inspector = inspect(session.bind)
            columns = inspector.get_columns('conversation_sessions')

            column_names = [col['name'] for col in columns]

            expected_columns = [
                'id',
                'user_id',
                'session_id',
                'is_active',
                'session_context',
                'message_count',
                'questions_asked',
                'commands_used',
                'started_at',
                'last_activity_at',
                'ended_at',
                'session_duration_minutes',
                'primary_topic',
                'courses_discussed'
            ]

            missing_columns = [col for col in expected_columns if col not in column_names]

            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                return False

            logger.info(f"✅ All {len(expected_columns)} expected columns found")
            logger.info(f"   Columns: {', '.join(column_names)}")
            return True

    except Exception as e:
        logger.error(f"Error verifying table schema: {e}")
        return False


def main():
    """Main migration function"""
    logger.info("=" * 60)
    logger.info("DATABASE MIGRATION: Add ConversationSession Table")
    logger.info("=" * 60)

    try:
        # Step 1: Create table
        logger.info("\nStep 1: Creating conversation_sessions table...")
        if not create_conversation_sessions_table():
            logger.error("❌ Migration failed at table creation step")
            return False

        # Step 2: Verify schema
        logger.info("\nStep 2: Verifying table schema...")
        if not verify_table_schema():
            logger.warning("⚠️  Table created but schema verification had issues")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("✅ MIGRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("\nThe conversation_sessions table has been added to your database.")
        logger.info("This enables:")
        logger.info("  - Session tracking with automatic timeout")
        logger.info("  - Conversation context continuity")
        logger.info("  - Session analytics for personalization")
        logger.info("\nYou can now restart your bot to use the new features.")

        return True

    except Exception as e:
        logger.error(f"❌ Migration failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
