"""
Script to remove user_id from ChromaDB embeddings metadata.

This script updates the metadata of existing embeddings in ChromaDB to remove
the user_id field, which is no longer used in the hybrid DocumentAccess architecture.

Key differences from backfill_and_reprocess.py:
- backfill_and_reprocess.py: Re-processed documents (deleted + re-embedded)
- This script: ONLY updates metadata (no re-processing needed)

Why no reprocessing needed:
- ChromaDB supports direct metadata updates via collection.update()
- We're not changing the embeddings themselves, just metadata
- Much faster and more efficient than re-processing
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.database import db_manager
from src.core.rag_pipeline import RAGPipeline
from src.data.models import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cleanup_chromadb_metadata(dry_run: bool = True):
    """Remove user_id from ChromaDB embeddings metadata.

    Args:
        dry_run: If True, only show what would be changed without making changes
    """
    logger.info("Starting ChromaDB metadata cleanup...")
    logger.info(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will update metadata)'}")

    try:
        # Initialize RAG pipeline to access ChromaDB collection
        rag_pipeline = RAGPipeline()
        collection = rag_pipeline.collection

        logger.info(f"Connected to ChromaDB collection: {collection.name}")

        # Get total count
        total_embeddings = collection.count()
        logger.info(f"Total embeddings in collection: {total_embeddings}")

        if total_embeddings == 0:
            logger.info("No embeddings found. Nothing to clean up.")
            return

        # Process in batches to avoid memory issues (similar to backfill_and_reprocess.py approach)
        batch_size = 100
        updated_count = 0
        skipped_count = 0
        error_count = 0

        logger.info(f"Processing in batches of {batch_size}...")

        # Get all embeddings with their metadata
        # Note: ChromaDB get() retrieves all by default if no ids specified
        # But for large collections, we should paginate
        offset = 0

        while offset < total_embeddings:
            logger.info(f"\nProcessing batch: {offset}-{offset + batch_size} of {total_embeddings}")

            try:
                # Get a batch of embeddings
                # ChromaDB doesn't have native pagination, so we get all and slice
                # For very large collections, this approach might need optimization
                batch_results = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=['metadatas']  # Only need metadata, not embeddings
                )

                if not batch_results['ids']:
                    logger.info("No more embeddings to process")
                    break

                batch_ids = batch_results['ids']
                batch_metadatas = batch_results['metadatas']

                logger.info(f"  Retrieved {len(batch_ids)} embeddings in this batch")

                # Track which embeddings need updating
                ids_to_update = []
                new_metadatas = []

                for embedding_id, metadata in zip(batch_ids, batch_metadatas):
                    if 'user_id' in metadata:
                        # This embedding has user_id - needs cleaning
                        updated_count += 1

                        if dry_run:
                            logger.info(f"  [DRY RUN] Would remove user_id from embedding {embedding_id}")
                            logger.info(f"    Current metadata: {metadata}")
                            logger.info(f"    New metadata: {dict((k, v) for k, v in metadata.items() if k != 'user_id')}")
                        else:
                            # Prepare updated metadata (remove user_id)
                            new_metadata = {k: v for k, v in metadata.items() if k != 'user_id'}
                            ids_to_update.append(embedding_id)
                            new_metadatas.append(new_metadata)
                    else:
                        # Already clean (no user_id)
                        skipped_count += 1

                # Update batch if not dry run
                if not dry_run and ids_to_update:
                    logger.info(f"  Updating {len(ids_to_update)} embeddings in this batch...")
                    collection.update(
                        ids=ids_to_update,
                        metadatas=new_metadatas
                    )
                    logger.info(f"  ✓ Updated {len(ids_to_update)} embeddings")

                offset += batch_size

            except Exception as e:
                error_count += len(batch_ids) if 'batch_ids' in locals() else batch_size
                logger.error(f"Error processing batch at offset {offset}: {e}")
                # Continue to next batch instead of failing completely
                offset += batch_size
                continue

        # Summary
        logger.info("\n" + "="*60)
        logger.info("CLEANUP SUMMARY")
        logger.info("="*60)
        logger.info(f"Total embeddings scanned: {total_embeddings}")
        logger.info(f"Embeddings with user_id (updated): {updated_count}")
        logger.info(f"Embeddings without user_id (skipped): {skipped_count}")
        logger.info(f"Errors: {error_count}")

        if dry_run:
            logger.info("\n⚠️  DRY RUN MODE - No changes were made")
            logger.info("Run with --confirm flag to apply changes")
        else:
            logger.info("\n✅ CLEANUP COMPLETE - Metadata updated successfully")

        # Verify the cleanup
        if not dry_run:
            logger.info("\nVerifying cleanup...")
            verify_results = collection.get(include=['metadatas'])
            remaining_with_user_id = sum(1 for m in verify_results['metadatas'] if 'user_id' in m)
            logger.info(f"Embeddings still containing user_id: {remaining_with_user_id}")

            if remaining_with_user_id == 0:
                logger.info("✅ Verification passed: All user_id fields removed")
            else:
                logger.warning(f"⚠️  {remaining_with_user_id} embeddings still have user_id field")

    except Exception as e:
        logger.error(f"Failed to cleanup ChromaDB metadata: {e}")
        raise


def get_metadata_statistics():
    """Get statistics about metadata fields in ChromaDB."""
    logger.info("Gathering metadata statistics...")

    try:
        rag_pipeline = RAGPipeline()
        collection = rag_pipeline.collection

        # Get all embeddings
        all_embeddings = collection.get(include=['metadatas'])
        total = len(all_embeddings['ids'])

        if total == 0:
            logger.info("No embeddings found")
            return

        # Count embeddings with user_id
        with_user_id = sum(1 for m in all_embeddings['metadatas'] if 'user_id' in m)
        without_user_id = total - with_user_id

        # Get sample of each type
        logger.info("\n" + "="*60)
        logger.info("METADATA STATISTICS")
        logger.info("="*60)
        logger.info(f"Total embeddings: {total}")
        logger.info(f"With user_id field: {with_user_id} ({with_user_id/total*100:.1f}%)")
        logger.info(f"Without user_id field: {without_user_id} ({without_user_id/total*100:.1f}%)")

        # Show sample metadata
        if with_user_id > 0:
            logger.info("\nSample embedding WITH user_id:")
            for metadata in all_embeddings['metadatas']:
                if 'user_id' in metadata:
                    logger.info(f"  {metadata}")
                    break

        if without_user_id > 0:
            logger.info("\nSample embedding WITHOUT user_id:")
            for metadata in all_embeddings['metadatas']:
                if 'user_id' not in metadata:
                    logger.info(f"  {metadata}")
                    break

        # Show all metadata fields
        all_fields = set()
        for metadata in all_embeddings['metadatas']:
            all_fields.update(metadata.keys())

        logger.info(f"\nAll metadata fields found: {sorted(all_fields)}")

    except Exception as e:
        logger.error(f"Failed to get metadata statistics: {e}")
        raise


if __name__ == "__main__":
    import argparse
    import traceback

    try:
        parser = argparse.ArgumentParser(description="Clean up ChromaDB metadata by removing user_id fields")
        parser.add_argument('--confirm', action='store_true',
                           help='Confirm to apply changes (default is dry-run)')
        parser.add_argument('--stats', action='store_true',
                           help='Show metadata statistics and exit')

        args = parser.parse_args()

        if args.stats:
            # Just show statistics
            get_metadata_statistics()
        else:
            # Run cleanup
            dry_run = not args.confirm

            logger.info("="*60)
            logger.info("CHROMADB METADATA CLEANUP SCRIPT")
            logger.info("="*60)

            if dry_run:
                logger.info("\n⚠️  Running in DRY RUN mode")
                logger.info("No changes will be made to the database")
                logger.info("Use --confirm flag to apply changes\n")
            else:
                logger.info("\n⚠️  Running in LIVE mode - changes will be applied!")
                logger.info("Press Ctrl+C within 5 seconds to cancel...")
                import time
                try:
                    time.sleep(5)
                except KeyboardInterrupt:
                    logger.info("\n❌ Cancelled by user")
                    sys.exit(0)

            cleanup_chromadb_metadata(dry_run=dry_run)

            logger.info("\n✅ Script completed successfully")

    except Exception as e:
        logger.error("\n" + "="*60)
        logger.error("FATAL ERROR")
        logger.error("="*60)
        logger.error(f"Error: {e}")
        logger.error("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
