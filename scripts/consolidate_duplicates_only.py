"""Run only the duplicate consolidation step"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from config.database import db_manager
from src.data.models import Document, DocumentAccess, DocumentEmbedding, UserNotification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def consolidate_duplicates():
    """Consolidate duplicate documents"""

    # Find duplicates
    with db_manager.get_session() as session:
        query = text("""
            SELECT lms_document_id, course_id, COUNT(*) as count, ARRAY_AGG(id ORDER BY id) as document_ids
            FROM documents
            WHERE lms_document_id IS NOT NULL
            GROUP BY lms_document_id, course_id
            HAVING COUNT(*) > 1
            ORDER BY count DESC
        """)
        result = session.execute(query)
        duplicates = result.fetchall()

        logger.info(f"Found {len(duplicates)} groups of duplicates")

        if not duplicates:
            logger.info("No duplicates to consolidate!")
            return True

        # Consolidate each group
        for dup in duplicates:
            document_ids = dup.document_ids
            keep_id = document_ids[0]
            remove_ids = document_ids[1:]

            logger.info(f"\nConsolidating: Keep {keep_id}, Remove {remove_ids}")

            # Verify keep document exists
            keep_doc = session.query(Document).filter(Document.id == keep_id).first()
            if not keep_doc:
                logger.warning(f"  SKIP: Keep document {keep_id} not found")
                continue

            # Migrate access records
            for remove_id in remove_ids:
                access_records = session.query(DocumentAccess).filter(
                    DocumentAccess.document_id == remove_id
                ).all()

                for access in access_records:
                    existing = session.query(DocumentAccess).filter(
                        DocumentAccess.user_id == access.user_id,
                        DocumentAccess.document_id == keep_id
                    ).first()

                    if not existing:
                        access.document_id = keep_id
                    else:
                        session.delete(access)

            # Migrate notifications
            for remove_id in remove_ids:
                notifications = session.query(UserNotification).filter(
                    UserNotification.document_id == remove_id
                ).all()

                for notification in notifications:
                    existing = session.query(UserNotification).filter(
                        UserNotification.user_id == notification.user_id,
                        UserNotification.document_id == keep_id
                    ).first()

                    if not existing:
                        notification.document_id = keep_id
                    else:
                        session.delete(notification)

            # Migrate embeddings
            for remove_id in remove_ids:
                embeddings = session.query(DocumentEmbedding).filter(
                    DocumentEmbedding.document_id == remove_id
                ).all()

                logger.info(f"  Migrating {len(embeddings)} embeddings from {remove_id} to {keep_id}")
                for embedding in embeddings:
                    embedding.document_id = keep_id

                # Flush to ensure embeddings are updated before we delete the document
                session.flush()

            # Delete duplicates
            for remove_id in remove_ids:
                doc = session.query(Document).filter(Document.id == remove_id).first()
                if doc:
                    logger.info(f"  Deleting document {remove_id}: {doc.title}")
                    session.delete(doc)

        session.commit()
        logger.info("\nConsolidation complete!")
        return True

if __name__ == "__main__":
    try:
        consolidate_duplicates()
        print("\n[SUCCESS] Consolidation completed")
    except Exception as e:
        print(f"\n[ERROR] Consolidation failed: {e}")
        sys.exit(1)
