"""
Migration: Add Confusion Events and Struggle Topics Tables

Creates two new tables for sentiment/confusion tracking:
1. confusion_events - Track every confusion detection event
2. struggle_topics - Track persistent struggles per topic

Benefits over JSON columns:
- Queryable, indexable, analyzable
- Unlimited history
- Proper normalization
- Better performance
"""

import sys
import os
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from config.database import db_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate():
    """Create confusion_events and struggle_topics tables"""

    logger.info("Starting migration: add_confusion_struggle_tables")

    try:
        with db_manager.get_session() as session:
            # Check if tables already exist
            check_query = text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name IN ('confusion_events', 'struggle_topics')
            """)
            result = session.execute(check_query)
            existing_tables = [row[0] for row in result.fetchall()]

            if 'confusion_events' in existing_tables and 'struggle_topics' in existing_tables:
                logger.info("✅ Tables already exist. Skipping migration.")
                return True

            # Create confusion_events table
            if 'confusion_events' not in existing_tables:
                logger.info("Creating confusion_events table...")

                create_confusion_events = text("""
                    CREATE TABLE confusion_events (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id),
                        session_id VARCHAR REFERENCES conversation_sessions(session_id),
                        interaction_id INTEGER REFERENCES user_interactions(id),

                        -- Confusion details
                        confusion_score FLOAT NOT NULL,
                        confusion_type VARCHAR(50) NOT NULL,
                        confidence FLOAT DEFAULT 0.8,

                        -- Detection signals
                        indicators JSON,
                        detected_patterns JSON,

                        -- Context
                        query_text TEXT,
                        topic VARCHAR(255),
                        previous_topic VARCHAR(255),

                        -- Sentiment indicators
                        wants_more_detail BOOLEAN DEFAULT FALSE,
                        wants_brevity BOOLEAN DEFAULT FALSE,

                        -- Metadata
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                session.execute(create_confusion_events)

                # Create indexes
                logger.info("Creating indexes for confusion_events...")
                session.execute(text("CREATE INDEX idx_confusion_events_user_id ON confusion_events(user_id)"))
                session.execute(text("CREATE INDEX idx_confusion_events_session_id ON confusion_events(session_id)"))
                session.execute(text("CREATE INDEX idx_confusion_events_created_at ON confusion_events(created_at)"))
                session.execute(text("CREATE INDEX idx_confusion_events_type ON confusion_events(confusion_type)"))

                logger.info("✅ confusion_events table created")

            # Create struggle_topics table
            if 'struggle_topics' not in existing_tables:
                logger.info("Creating struggle_topics table...")

                create_struggle_topics = text("""
                    CREATE TABLE struggle_topics (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id),

                        -- Topic details
                        topic VARCHAR(255) NOT NULL,
                        topic_aliases JSON,

                        -- Struggle metrics
                        struggle_score FLOAT NOT NULL,
                        query_count INTEGER DEFAULT 1,
                        confusion_count INTEGER DEFAULT 0,

                        -- Rating context
                        avg_rating FLOAT,
                        low_rating_count INTEGER DEFAULT 0,

                        -- Progression tracking
                        complexity_trend VARCHAR(50),
                        first_query_complexity FLOAT,
                        latest_query_complexity FLOAT,

                        -- Indicators
                        indicators JSON,

                        -- Temporal tracking
                        first_asked_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        last_asked_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        resolution_detected_at TIMESTAMP WITH TIME ZONE,

                        -- Status
                        is_active BOOLEAN DEFAULT TRUE,
                        is_resolved BOOLEAN DEFAULT FALSE,

                        -- Metadata
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE
                    )
                """)
                session.execute(create_struggle_topics)

                # Create indexes
                logger.info("Creating indexes for struggle_topics...")
                session.execute(text("CREATE INDEX idx_struggle_topics_user_id ON struggle_topics(user_id)"))
                session.execute(text("CREATE INDEX idx_struggle_topics_topic ON struggle_topics(topic)"))
                session.execute(text("CREATE INDEX idx_struggle_topics_is_active ON struggle_topics(is_active)"))
                session.execute(text("CREATE INDEX idx_struggle_topics_last_asked ON struggle_topics(last_asked_at)"))
                session.execute(text("CREATE INDEX idx_struggle_topics_user_active ON struggle_topics(user_id, is_active)"))

                # Create unique constraint: one active struggle per user per topic
                session.execute(text("""
                    CREATE UNIQUE INDEX idx_struggle_topics_user_topic_unique
                    ON struggle_topics(user_id, topic)
                    WHERE is_active = TRUE
                """))

                logger.info("✅ struggle_topics table created")

            session.commit()
            logger.info("🎉 Migration completed successfully!")
            return True

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}", exc_info=True)
        return False


def rollback():
    """Drop the tables (rollback migration)"""

    logger.info("Starting rollback: add_confusion_struggle_tables")

    try:
        with db_manager.get_session() as session:
            # Drop tables in reverse order (handle foreign keys)
            session.execute(text("DROP TABLE IF EXISTS confusion_events CASCADE"))
            session.execute(text("DROP TABLE IF EXISTS struggle_topics CASCADE"))
            session.commit()

            logger.info("🔄 Rollback completed successfully!")
            return True

    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Confusion/Struggle tables migration')
    parser.add_argument('--rollback', action='store_true', help='Rollback the migration')
    args = parser.parse_args()

    if args.rollback:
        success = rollback()
    else:
        success = migrate()

    sys.exit(0 if success else 1)
