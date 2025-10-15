"""Quick script to verify migration status"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from config.database import db_manager
from src.data.models import Document, DocumentAccess

print("Checking migration status...\n")

with db_manager.get_session() as session:
    # Count DocumentAccess records
    access_count = session.query(DocumentAccess).count()
    print(f"[OK] DocumentAccess records: {access_count}")

    # Check for duplicates
    query = text("""
        SELECT COUNT(*) as duplicate_groups
        FROM (
            SELECT lms_document_id, course_id
            FROM documents
            WHERE lms_document_id IS NOT NULL
            GROUP BY lms_document_id, course_id
            HAVING COUNT(*) > 1
        ) sub
    """)
    result = session.execute(query)
    duplicate_count = result.scalar()
    print(f"[OK] Remaining duplicate groups: {duplicate_count}")

    # Total documents
    total_docs = session.query(Document).count()
    print(f"[OK] Total documents: {total_docs}")

    if duplicate_count == 0:
        print("\n[SUCCESS] MIGRATION SUCCESSFUL - No duplicates!")
    else:
        print(f"\n[WARNING] {duplicate_count} duplicate groups remain")
