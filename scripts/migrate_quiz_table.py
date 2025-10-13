"""
Migration script to create the quiz_sessions table

Run this script to add quiz functionality to the database:
    python scripts/migrate_quiz_table.py
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


def migrate_quiz_table():
    """Create quiz_sessions table"""
    try:
        engine = db_manager.get_engine()
        inspector = inspect(engine)

        # Check if quiz_sessions table already exists
        if inspector.has_table('quiz_sessions'):
            logger.warning("quiz_sessions table already exists. Skipping migration.")
            return True

        logger.info("Creating quiz_sessions table...")

        with engine.begin() as conn:
            # Create quiz_sessions table
            conn.execute(text("""
                CREATE TABLE quiz_sessions (
                    id SERIAL NOT NULL,
                    user_id INTEGER NOT NULL,
                    document_id INTEGER,
                    topic VARCHAR,
                    questions JSON NOT NULL,
                    current_question_index INTEGER DEFAULT 0,
                    total_questions INTEGER NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    is_paused BOOLEAN DEFAULT FALSE,
                    correct_answers INTEGER DEFAULT 0,
                    wrong_answers INTEGER DEFAULT 0,
                    started_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                    last_interaction_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                    completed_at TIMESTAMP WITH TIME ZONE,
                    PRIMARY KEY (id),
                    FOREIGN KEY(user_id) REFERENCES users (id),
                    FOREIGN KEY(document_id) REFERENCES documents (id)
                )
            """))

            # Create indexes for better performance
            conn.execute(text("CREATE INDEX ix_quiz_sessions_id ON quiz_sessions (id)"))
            conn.execute(text("CREATE INDEX ix_quiz_sessions_user_id ON quiz_sessions (user_id)"))
            conn.execute(text("CREATE INDEX ix_quiz_sessions_is_active ON quiz_sessions (is_active)"))

        logger.info("✅ Successfully created quiz_sessions table with indexes")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to create quiz_sessions table: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting quiz table migration...")
    success = migrate_quiz_table()

    if success:
        logger.info("✅ Migration completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Migration failed!")
        sys.exit(1)
