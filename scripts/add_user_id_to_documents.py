"""
Migration script to add user_id to the documents table

Run this script to add the user_id column for multi-user data isolation:
    python scripts/add_user_id_to_documents.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text, inspect
from config.database import db_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_document_table():
    """Add user_id column to the documents table"""
    try:
        engine = db_manager.get_engine()
        inspector = inspect(engine)

        # Check if the column already exists
        columns = inspector.get_columns('documents')
        if any(c['name'] == 'user_id' for c in columns):
            logger.warning("Column 'user_id' already exists in 'documents' table. Skipping migration.")
            return True

        logger.info("Adding 'user_id' column to 'documents' table...")

        with engine.begin() as conn:
            # Add the user_id column
            conn.execute(text("""
                ALTER TABLE documents
                ADD COLUMN user_id INTEGER;
            """))

            # Add a foreign key constraint to link to the users table
            conn.execute(text("""
                ALTER TABLE documents
                ADD CONSTRAINT fk_documents_user_id
                FOREIGN KEY (user_id) REFERENCES users(id);
            """))

            # Create an index for better performance on queries involving user_id
            conn.execute(text("""
                CREATE INDEX ix_documents_user_id ON documents (user_id);
            """))

        logger.info("✅ Successfully added 'user_id' column and index to 'documents' table.")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to update 'documents' table: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting document table migration to add user_id...")
    success = migrate_document_table()

    if success:
        logger.info("✅ Migration completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Migration failed!")
        sys.exit(1)
