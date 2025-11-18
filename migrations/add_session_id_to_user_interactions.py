"""
Migration Script: Add session_id column to user_interactions table

This migration adds a foreign key relationship between user_interactions
and conversation_sessions, enabling proper session-based context isolation.

Run this migration with: python migrations/add_session_id_to_user_interactions.py
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from config.database import db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate():
    """Add session_id column to user_interactions table"""
    try:
        with db_manager.get_session() as session:
            logger.info("Starting migration: add session_id to user_interactions")

            # Check if column already exists
            check_query = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='user_interactions' AND column_name='session_id'
            """)

            result = session.execute(check_query)
            column_exists = result.fetchone() is not None

            if column_exists:
                logger.info("✅ Column 'session_id' already exists in user_interactions. Skipping migration.")
                return True

            # Step 1: Add the session_id column (nullable initially)
            logger.info("Adding session_id column to user_interactions table...")
            add_column_query = text("""
                ALTER TABLE user_interactions
                ADD COLUMN session_id VARCHAR
            """)
            session.execute(add_column_query)
            session.commit()
            logger.info("✅ Column added successfully")

            # Step 2: Create index on session_id for performance
            logger.info("Creating index on session_id...")
            create_index_query = text("""
                CREATE INDEX IF NOT EXISTS ix_user_interactions_session_id
                ON user_interactions(session_id)
            """)
            session.execute(create_index_query)
            session.commit()
            logger.info("✅ Index created successfully")

            # Step 3: Backfill session_id for existing records
            # Strategy: For each user_interaction without session_id,
            # find the active session at that time or create a legacy session
            logger.info("Backfilling session_id for existing interactions...")

            backfill_query = text("""
                UPDATE user_interactions ui
                SET session_id = (
                    SELECT cs.session_id
                    FROM conversation_sessions cs
                    WHERE cs.user_id = ui.user_id
                    AND cs.started_at <= ui.created_at
                    AND (cs.ended_at IS NULL OR cs.ended_at >= ui.created_at)
                    ORDER BY cs.started_at DESC
                    LIMIT 1
                )
                WHERE ui.session_id IS NULL
            """)
            result = session.execute(backfill_query)
            session.commit()
            logger.info(f"✅ Backfilled {result.rowcount} existing interactions with session_id")

            # Step 4: Handle orphaned interactions (those without a matching session)
            # Create a "legacy" session for each user's orphaned interactions
            logger.info("Creating legacy sessions for orphaned interactions...")

            # Get users with orphaned interactions
            orphaned_users_query = text("""
                SELECT DISTINCT user_id
                FROM user_interactions
                WHERE session_id IS NULL
            """)
            orphaned_users = session.execute(orphaned_users_query).fetchall()

            for (user_id,) in orphaned_users:
                # Create a legacy session for this user
                import uuid
                legacy_session_id = str(uuid.uuid4())

                create_legacy_session_query = text("""
                    INSERT INTO conversation_sessions
                    (user_id, session_id, is_active, primary_topic, message_count, started_at, last_activity_at)
                    VALUES
                    (:user_id, :session_id, FALSE, 'Legacy Session', 0,
                     (SELECT MIN(created_at) FROM user_interactions WHERE user_id = :user_id AND session_id IS NULL),
                     (SELECT MAX(created_at) FROM user_interactions WHERE user_id = :user_id AND session_id IS NULL))
                """)
                session.execute(create_legacy_session_query, {
                    'user_id': user_id,
                    'session_id': legacy_session_id
                })

                # Link orphaned interactions to this legacy session
                link_orphaned_query = text("""
                    UPDATE user_interactions
                    SET session_id = :session_id
                    WHERE user_id = :user_id AND session_id IS NULL
                """)
                session.execute(link_orphaned_query, {
                    'user_id': user_id,
                    'session_id': legacy_session_id
                })

            session.commit()
            logger.info(f"✅ Created {len(orphaned_users)} legacy sessions for orphaned interactions")

            # Step 5: Add foreign key constraint
            logger.info("Adding foreign key constraint...")
            add_fk_query = text("""
                ALTER TABLE user_interactions
                ADD CONSTRAINT fk_user_interactions_session_id
                FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
                ON DELETE SET NULL
            """)
            session.execute(add_fk_query)
            session.commit()
            logger.info("✅ Foreign key constraint added successfully")

            logger.info("🎉 Migration completed successfully!")
            return True

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}", exc_info=True)
        return False


def rollback():
    """Rollback the migration (remove session_id column)"""
    try:
        with db_manager.get_session() as session:
            logger.info("Starting rollback: remove session_id from user_interactions")

            # Drop foreign key constraint
            logger.info("Dropping foreign key constraint...")
            drop_fk_query = text("""
                ALTER TABLE user_interactions
                DROP CONSTRAINT IF EXISTS fk_user_interactions_session_id
            """)
            session.execute(drop_fk_query)
            session.commit()

            # Drop index
            logger.info("Dropping index...")
            drop_index_query = text("""
                DROP INDEX IF EXISTS ix_user_interactions_session_id
            """)
            session.execute(drop_index_query)
            session.commit()

            # Drop column
            logger.info("Dropping session_id column...")
            drop_column_query = text("""
                ALTER TABLE user_interactions
                DROP COLUMN IF EXISTS session_id
            """)
            session.execute(drop_column_query)
            session.commit()

            # Delete legacy sessions
            logger.info("Deleting legacy sessions...")
            delete_legacy_query = text("""
                DELETE FROM conversation_sessions
                WHERE primary_topic = 'Legacy Session'
            """)
            session.execute(delete_legacy_query)
            session.commit()

            logger.info("✅ Rollback completed successfully!")
            return True

    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migration: Add session_id to user_interactions")
    parser.add_argument('--rollback', action='store_true', help='Rollback the migration')
    args = parser.parse_args()

    if args.rollback:
        success = rollback()
    else:
        success = migrate()

    sys.exit(0 if success else 1)
