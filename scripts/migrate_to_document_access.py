"""
Migration Script: Transition from user_id-per-document to DocumentAccess table

This script:
1. Creates the document_access table
2. Migrates existing user_id relationships to document_access records
3. Identifies and consolidates duplicate documents
4. Updates embeddings to point to consolidated documents
5. Prepares for removal of user_id column from documents table

RUN THIS BEFORE removing user_id from Document model!
"""

import sys
import logging
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from config.database import db_manager
from src.data.models import Document, DocumentAccess, DocumentEmbedding, UserNotification, Base
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_document_access_table():
    """Step 1: Create the document_access table"""
    logger.info("=" * 80)
    logger.info("STEP 1: Creating document_access table")
    logger.info("=" * 80)

    try:
        # Create table using SQLAlchemy
        engine = db_manager.get_engine()
        DocumentAccess.__table__.create(engine, checkfirst=True)
        logger.info("✅ document_access table created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create document_access table: {e}")
        return False

def migrate_existing_access_from_user_id():
    """Step 2: Create DocumentAccess records from existing Document.user_id"""
    logger.info("=" * 80)
    logger.info("STEP 2: Migrating existing user_id relationships to document_access")
    logger.info("=" * 80)

    try:
        with db_manager.get_session() as session:
            # Get all documents with user_id set
            documents_with_users = session.query(Document).filter(
                Document.user_id.isnot(None)
            ).all()

            logger.info(f"Found {len(documents_with_users)} documents with user_id set")

            created_count = 0
            skipped_count = 0

            for doc in documents_with_users:
                # Check if access record already exists
                existing_access = session.query(DocumentAccess).filter(
                    DocumentAccess.user_id == doc.user_id,
                    DocumentAccess.document_id == doc.id
                ).first()

                if not existing_access:
                    # Create new access record
                    access = DocumentAccess(
                        user_id=doc.user_id,
                        document_id=doc.id,
                        access_source="migration",
                        granted_at=doc.created_at or datetime.now()
                    )
                    session.add(access)
                    created_count += 1
                else:
                    skipped_count += 1

            session.commit()
            logger.info(f"✅ Created {created_count} document_access records")
            logger.info(f"   Skipped {skipped_count} existing records")
            return True

    except Exception as e:
        logger.error(f"❌ Failed to migrate user_id relationships: {e}")
        return False

def find_duplicate_documents():
    """Step 3: Identify duplicate documents (same lms_document_id + course_id)"""
    logger.info("=" * 80)
    logger.info("STEP 3: Identifying duplicate documents")
    logger.info("=" * 80)

    try:
        with db_manager.get_session() as session:
            # Find documents with same lms_document_id and course_id
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

            logger.info(f"Found {len(duplicates)} groups of duplicate documents")

            if duplicates:
                logger.info("\nDuplicate groups:")
                for dup in duplicates:
                    logger.info(f"  LMS ID: {dup.lms_document_id}, Course: {dup.course_id}")
                    logger.info(f"    Document IDs: {dup.document_ids} ({dup.count} duplicates)")

            return duplicates

    except Exception as e:
        logger.error(f"❌ Failed to find duplicates: {e}")
        return []

def consolidate_duplicate_documents(duplicates, dry_run=True):
    """Step 4: Consolidate duplicates - keep oldest, migrate relationships"""
    logger.info("=" * 80)
    logger.info(f"STEP 4: Consolidating duplicate documents ({'DRY RUN' if dry_run else 'LIVE'})")
    logger.info("=" * 80)

    if not duplicates:
        logger.info("No duplicates to consolidate")
        return True

    try:
        with db_manager.get_session() as session:
            for dup in duplicates:
                document_ids = dup.document_ids
                keep_id = document_ids[0]  # Keep the oldest (lowest ID)
                remove_ids = document_ids[1:]  # Remove the rest

                logger.info(f"\nProcessing duplicate group:")
                logger.info(f"  Keeping document ID: {keep_id}")
                logger.info(f"  Removing document IDs: {remove_ids}")

                # Get all affected documents
                keep_doc = session.query(Document).filter(Document.id == keep_id).first()
                remove_docs = session.query(Document).filter(Document.id.in_(remove_ids)).all()

                if not keep_doc:
                    logger.warning(f"  ⚠️  Could not find keep document {keep_id}, skipping")
                    continue

                # Migrate DocumentAccess records
                for remove_id in remove_ids:
                    access_records = session.query(DocumentAccess).filter(
                        DocumentAccess.document_id == remove_id
                    ).all()

                    logger.info(f"  Migrating {len(access_records)} access records from {remove_id} to {keep_id}")

                    for access in access_records:
                        # Check if user already has access to keep_doc
                        existing = session.query(DocumentAccess).filter(
                            DocumentAccess.user_id == access.user_id,
                            DocumentAccess.document_id == keep_id
                        ).first()

                        if not existing:
                            if not dry_run:
                                # Update to point to keep_doc
                                access.document_id = keep_id
                            logger.info(f"    ✓ Migrated access for user {access.user_id}")
                        else:
                            if not dry_run:
                                # Delete duplicate access record
                                session.delete(access)
                            logger.info(f"    ✓ Removed duplicate access for user {access.user_id}")

                # Migrate UserNotification records
                for remove_id in remove_ids:
                    notifications = session.query(UserNotification).filter(
                        UserNotification.document_id == remove_id
                    ).all()

                    logger.info(f"  Migrating {len(notifications)} notifications from {remove_id} to {keep_id}")

                    for notification in notifications:
                        # Check if notification already exists for keep_doc
                        existing = session.query(UserNotification).filter(
                            UserNotification.user_id == notification.user_id,
                            UserNotification.document_id == keep_id
                        ).first()

                        if not existing:
                            if not dry_run:
                                notification.document_id = keep_id
                            logger.info(f"    ✓ Migrated notification for user {notification.user_id}")
                        else:
                            if not dry_run:
                                session.delete(notification)
                            logger.info(f"    ✓ Removed duplicate notification for user {notification.user_id}")

                # Migrate DocumentEmbedding records
                for remove_id in remove_ids:
                    embeddings = session.query(DocumentEmbedding).filter(
                        DocumentEmbedding.document_id == remove_id
                    ).all()

                    logger.info(f"  Migrating {len(embeddings)} embeddings from {remove_id} to {keep_id}")

                    if not dry_run:
                        for embedding in embeddings:
                            embedding.document_id = keep_id

                # Delete duplicate documents
                if not dry_run:
                    for remove_doc in remove_docs:
                        logger.info(f"  🗑️  Deleting duplicate document {remove_doc.id}: {remove_doc.title}")
                        session.delete(remove_doc)
                else:
                    logger.info(f"  [DRY RUN] Would delete {len(remove_docs)} duplicate documents")

            if not dry_run:
                session.commit()
                logger.info("\n✅ Consolidation completed and committed")
            else:
                logger.info("\n✅ Dry run completed - no changes made")

            return True

    except Exception as e:
        logger.error(f"❌ Failed to consolidate duplicates: {e}")
        return False

def verify_migration():
    """Step 5: Verify the migration was successful"""
    logger.info("=" * 80)
    logger.info("STEP 5: Verifying migration")
    logger.info("=" * 80)

    try:
        with db_manager.get_session() as session:
            # Count DocumentAccess records
            access_count = session.query(DocumentAccess).count()
            logger.info(f"✓ DocumentAccess records: {access_count}")

            # Check for remaining duplicates
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
            logger.info(f"✓ Remaining duplicate groups: {duplicate_count}")

            # Count documents with user_id still set
            docs_with_user_id = session.query(Document).filter(
                Document.user_id.isnot(None)
            ).count()
            logger.info(f"✓ Documents with user_id set: {docs_with_user_id}")

            # Summary
            logger.info("\n" + "=" * 80)
            if duplicate_count == 0:
                logger.info("✅ MIGRATION SUCCESSFUL - No duplicates remaining")
            else:
                logger.warning(f"⚠️  WARNING - {duplicate_count} duplicate groups still exist")

            if docs_with_user_id > 0:
                logger.info(f"✓ {docs_with_user_id} documents still have user_id (will be removed in final step)")

            return True

    except Exception as e:
        logger.error(f"❌ Failed to verify migration: {e}")
        return False

def main(auto_confirm=False):
    """Main migration workflow"""
    logger.info("\n" + "=" * 80)
    logger.info("DOCUMENT ACCESS MIGRATION SCRIPT")
    logger.info("=" * 80 + "\n")

    # Step 1: Create table
    if not create_document_access_table():
        logger.error("Failed at Step 1. Aborting.")
        return False

    # Step 2: Migrate existing user_id relationships
    if not migrate_existing_access_from_user_id():
        logger.error("Failed at Step 2. Aborting.")
        return False

    # Step 3: Find duplicates
    duplicates = find_duplicate_documents()

    # Step 4: Consolidate duplicates (DRY RUN first)
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING DRY RUN - No changes will be made")
    logger.info("=" * 80)
    if duplicates:
        if not consolidate_duplicate_documents(duplicates, dry_run=True):
            logger.error("Failed at Step 4 (dry run). Aborting.")
            return False

        # Ask for confirmation
        logger.info("\n" + "=" * 80)
        if auto_confirm:
            logger.info("Auto-confirm enabled - proceeding with LIVE consolidation")
            response = 'yes'
        else:
            try:
                response = input("\nProceed with LIVE consolidation? (yes/no): ")
            except EOFError:
                logger.error("Cannot read input. Use --confirm flag for non-interactive mode.")
                return False

        if response.lower() == 'yes':
            if not consolidate_duplicate_documents(duplicates, dry_run=False):
                logger.error("Failed at Step 4 (live). Aborting.")
                return False
        else:
            logger.info("Skipping live consolidation")

    # Step 5: Verify
    if not verify_migration():
        logger.error("Failed at Step 5. Please review.")
        return False

    logger.info("\n" + "=" * 80)
    logger.info("MIGRATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Test DocumentAccess functionality with existing code")
    logger.info("2. Update LMS integration to use DocumentAccess")
    logger.info("3. Update RAG pipeline to filter by DocumentAccess")
    logger.info("4. Remove user_id column from documents table (final step)")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Migrate to DocumentAccess hybrid architecture')
    parser.add_argument('--confirm', action='store_true', help='Auto-confirm consolidation (non-interactive mode)')
    args = parser.parse_args()

    success = main(auto_confirm=args.confirm)
    sys.exit(0 if success else 1)
