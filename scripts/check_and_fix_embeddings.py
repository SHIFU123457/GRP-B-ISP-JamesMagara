"""Check for and fix orphaned embeddings"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.database import db_manager
from src.data.models import DocumentEmbedding

with db_manager.get_session() as session:
    # Find embeddings with NULL document_id
    orphaned = session.query(DocumentEmbedding).filter(
        DocumentEmbedding.document_id == None
    ).all()

    print(f"Found {len(orphaned)} orphaned embeddings (document_id is NULL)")

    if orphaned:
        print(f"\nDeleting {len(orphaned)} orphaned embeddings...")
        for emb in orphaned:
            session.delete(emb)
        session.commit()
        print("[SUCCESS] Orphaned embeddings deleted")
    else:
        print("[OK] No orphaned embeddings found")
