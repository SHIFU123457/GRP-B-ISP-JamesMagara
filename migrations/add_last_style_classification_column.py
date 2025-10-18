"""
Migration script to add last_style_classification column to users table

This migration adds a timestamp column to track when each user's learning style
was last classified, enabling time-based reclassification every 12 hours.

Run this migration with: python migrations/add_last_style_classification_column.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from sqlalchemy import text
from config.database import db_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate():
    """Add last_style_classification column to users table"""

    logger.info("=" * 80)
    logger.info("Migration: Add last_style_classification column to users table")
    logger.info("=" * 80)

    try:
        with db_manager.get_session() as session:
            # Check if column already exists
            check_query = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='users' AND column_name='last_style_classification'
            """)

            result = session.execute(check_query)
            column_exists = result.fetchone() is not None

            if column_exists:
                logger.info("✅ Column 'last_style_classification' already exists. Skipping migration.")
                return True

            logger.info("Adding 'last_style_classification' column to users table...")

            # Add the new column
            alter_query = text("""
                ALTER TABLE users
                ADD COLUMN last_style_classification TIMESTAMP WITH TIME ZONE
            """)

            session.execute(alter_query)
            session.commit()

            logger.info("✅ Successfully added 'last_style_classification' column")

            # Verify the column was added
            result = session.execute(check_query)
            if result.fetchone():
                logger.info("✅ Migration verified successfully")
                return True
            else:
                logger.error("❌ Migration verification failed")
                return False

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}", exc_info=True)
        return False


def rollback():
    """Remove last_style_classification column (rollback migration)"""

    logger.info("=" * 80)
    logger.info("Rollback: Remove last_style_classification column from users table")
    logger.info("=" * 80)

    try:
        with db_manager.get_session() as session:
            # Check if column exists
            check_query = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='users' AND column_name='last_style_classification'
            """)

            result = session.execute(check_query)
            column_exists = result.fetchone() is not None

            if not column_exists:
                logger.info("✅ Column 'last_style_classification' does not exist. Nothing to rollback.")
                return True

            logger.info("Removing 'last_style_classification' column from users table...")

            # Drop the column
            alter_query = text("""
                ALTER TABLE users
                DROP COLUMN last_style_classification
            """)

            session.execute(alter_query)
            session.commit()

            logger.info("✅ Successfully removed 'last_style_classification' column")

            # Verify the column was removed
            result = session.execute(check_query)
            if not result.fetchone():
                logger.info("✅ Rollback verified successfully")
                return True
            else:
                logger.error("❌ Rollback verification failed")
                return False

    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        success = rollback()
    else:
        success = migrate()

    if success:
        logger.info("\n✅ Migration completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Migration failed!")
        sys.exit(1)
