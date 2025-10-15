"""
One-time script to backfill user_id for existing documents and re-process them.

This script performs two main functions:
1.  Backfills the `user_id` for documents where it is NULL. It does this by checking
    course enrollments. If a document belongs to a course with only one user, it
    assigns the document to that user. If ownership is ambiguous (multiple users),
    it logs a warning for manual intervention.
2.  Triggers re-processing for all documents that have an assigned user_id, ensuring
    their vector embeddings are updated in ChromaDB with the correct user metadata.


"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.database import db_manager
from src.data.models import Document, CourseEnrollment, User
from src.core.rag_pipeline import RAGPipeline

import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log file to track processed documents to make the script resumable
PROCESSED_LOG_FILE = Path(__file__).parent / 'processed_docs.log'


def get_already_processed_docs() -> set:
    """Reads the log file to return a set of document IDs that have already been processed."""
    if not PROCESSED_LOG_FILE.exists():
        return set()
    with open(PROCESSED_LOG_FILE, 'r') as f:
        return {int(line.strip()) for line in f if line.strip()}


def log_processed_doc(doc_id: int):
    """Appends a successfully processed document ID to the log file."""
    with open(PROCESSED_LOG_FILE, 'a') as f:
        f.write(f"{doc_id}\n")

def backfill_document_user_ids():
    """Assigns user_id to documents based on course enrollment.

    Returns:
        A tuple of (backfilled_count, ambiguous_count)
    """
    logger.info("Starting to backfill user_id for existing documents...")
    backfilled_count = 0
    ambiguous_docs = []

    with db_manager.get_session() as session:
        # Find documents that need a user_id
        docs_to_update = session.query(Document).filter(Document.user_id == None).all()

        if not docs_to_update:
            logger.info("No documents found needing a user_id. Backfill not required.")
            return 0, 0

        logger.info(f"Found {len(docs_to_update)} documents to process.")

        for doc in docs_to_update:
            # Find users enrolled in the document's course
            enrollments = session.query(CourseEnrollment).filter(
                CourseEnrollment.course_id == doc.course_id
            ).all()

            if len(enrollments) == 1:
                # Safe to assign: only one user is in this course
                doc.user_id = enrollments[0].user_id
                backfilled_count += 1
                logger.info(f"Assigned Document ID {doc.id} (\'{doc.title}\') to User ID {doc.user_id}.")
            else:
                # Ambiguous ownership
                user_ids = [e.user_id for e in enrollments]
                ambiguous_docs.append((doc.id, doc.title, user_ids))

        if ambiguous_docs:
            logger.warning("Could not automatically assign a user to the following documents due to ambiguous ownership (multiple enrolled users):")
            for doc_id, title, user_ids in ambiguous_docs:
                logger.warning(f"  - Document ID: {doc_id} (\'{title}\') is enrolled by User IDs: {user_ids}")
            logger.warning("Please assign a user_id to these documents manually in the database.")

        if backfilled_count > 0:
            session.commit()
            logger.info(f"Successfully backfilled user_id for {backfilled_count} documents.")
        else:
            logger.info("No documents were automatically backfilled.")

    return backfilled_count, len(ambiguous_docs)

def reprocess_all_documents():
    """Finds all documents with a user_id and re-processes them for the vector store."""
    logger.info("Starting re-processing of all documents with an assigned user...")
    
    # Get lists of documents to process
    already_processed = get_already_processed_docs()
    doc_ids_to_reprocess = []
    with db_manager.get_session() as session:
        doc_ids_to_reprocess = [item[0] for item in session.query(Document.id).filter(Document.user_id != None).all()]

    # Filter out documents that have already been processed
    docs_to_run = [doc_id for doc_id in doc_ids_to_reprocess if doc_id not in already_processed]

    if not docs_to_run:
        logger.info("All documents have already been processed. Nothing to do.")
        return

    logger.info(f"{len(already_processed)} documents were already processed and will be skipped.")
    logger.info(f"Found {len(docs_to_run)} documents remaining to re-process.")

    try:
        rag_pipeline = RAGPipeline()
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}. Aborting re-processing.")
        return

    processed_count = 0
    failed_count = 0

    for i, doc_id in enumerate(docs_to_run, 1):
        logger.info(f"Processing document {i}/{len(docs_to_run)}: ID {doc_id}...")
        try:
            success = rag_pipeline.process_and_store_document(doc_id)
            if success:
                log_processed_doc(doc_id) # Log success
                processed_count += 1
            else:
                failed_count += 1
                logger.error(f"Failed to re-process document ID: {doc_id}")
        except Exception as e:
            failed_count += 1
            logger.error(f"An exception occurred while re-processing document ID {doc_id}: {e}")


def backfill_document_user_ids():
    """Assigns user_id to documents based on course enrollment.

    Returns:
        A tuple of (backfilled_count, ambiguous_count)
    """
    logger.info("Starting to backfill user_id for existing documents...")
    backfilled_count = 0
    ambiguous_docs = []

    with db_manager.get_session() as session:
        # Find documents that need a user_id
        docs_to_update = session.query(Document).filter(Document.user_id == None).all()

        if not docs_to_update:
            logger.info("No documents found needing a user_id. Backfill not required.")
            return 0, 0

        logger.info(f"Found {len(docs_to_update)} documents to process.")

        for doc in docs_to_update:
            # Find users enrolled in the document's course
            enrollments = session.query(CourseEnrollment).filter(
                CourseEnrollment.course_id == doc.course_id
            ).all()

            if len(enrollments) == 1:
                # Safe to assign: only one user is in this course
                doc.user_id = enrollments[0].user_id
                backfilled_count += 1
                logger.info(f"Assigned Document ID {doc.id} ('{doc.title}') to User ID {doc.user_id}.")
            else:
                # Ambiguous ownership
                user_ids = [e.user_id for e in enrollments]
                ambiguous_docs.append((doc.id, doc.title, user_ids))

        if ambiguous_docs:
            logger.warning("Could not automatically assign a user to the following documents due to ambiguous ownership (multiple enrolled users):")
            for doc_id, title, user_ids in ambiguous_docs:
                logger.warning(f"  - Document ID: {doc_id} ('{title}') is enrolled by User IDs: {user_ids}")
            logger.warning("Please assign a user_id to these documents manually in the database.")

        if backfilled_count > 0:
            session.commit()
            logger.info(f"Successfully backfilled user_id for {backfilled_count} documents.")
        else:
            logger.info("No documents were automatically backfilled.")

    return backfilled_count, len(ambiguous_docs)

def reprocess_all_documents():
    """Finds all documents with a user_id and re-processes them for the vector store."""
    logger.info("Starting re-processing of all documents with an assigned user...")
    
    doc_ids_to_reprocess = []
    with db_manager.get_session() as session:
        # Get all document IDs that have a user assigned. Fetch only IDs to save memory.
        doc_ids_to_reprocess = session.query(Document.id).filter(Document.user_id != None).all()
        # Convert list of tuples to list of ints
        doc_ids_to_reprocess = [item[0] for item in doc_ids_to_reprocess]

    if not doc_ids_to_reprocess:
        logger.info("No documents found to re-process.")
        return

    logger.info(f"Found {len(doc_ids_to_reprocess)} documents to re-process.")

    try:
        rag_pipeline = RAGPipeline()
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}. Aborting re-processing.")
        return

    processed_count = 0
    failed_count = 0

    for i, doc_id in enumerate(doc_ids_to_reprocess, 1):
        # The session is now managed entirely within process_and_store_document
        logger.info(f"Processing document {i}/{len(doc_ids_to_reprocess)}: ID {doc_id}...")
        try:
            success = rag_pipeline.process_and_store_document(doc_id)
            if success:
                processed_count += 1
            else:
                failed_count += 1
                logger.error(f"Failed to re-process document ID: {doc_id}")
        except Exception as e:
            failed_count += 1
            logger.error(f"An exception occurred while re-processing document ID {doc_id}: {e}")

if __name__ == "__main__":
    logger.info("--- Step 1: Backfilling User IDs for Documents ---")
    backfilled, ambiguous = backfill_document_user_ids()
    logger.info("--- Backfill Complete ---")

    logger.info("\n--- Step 2: Re-processing All Documents for Vector Store ---")
    reprocess_all_documents()
    logger.info("--- Re-processing Complete ---")

    logger.info("\n✅ Data migration and re-processing finished.")
    if ambiguous > 0:
        logger.warning(f"Action required: {ambiguous} documents could not be assigned a user and need manual review.")
