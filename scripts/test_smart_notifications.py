#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced intelligent notification system
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.chdir(str(project_root))

from config.database import db_manager
from config.settings import settings
from src.utils.logger import setup_logging
from src.services.scheduler import scheduler_service

def create_sample_documents():
    """Create sample documents to test different material types"""

    logger = logging.getLogger(__name__)

    with db_manager.get_session() as session:
        from src.data.models import Document, Course, User

        # Get a course to add documents to
        course = session.query(Course).first()
        if not course:
            logger.warning("No courses found in database - cannot test notifications")
            return []

        # Get a user to be the owner of these documents
        user = session.query(User).first()
        if not user:
            logger.warning("No users found in database - cannot create documents for test")
            return []

        # Sample documents with different types
        sample_docs = [
            {
                'user_id': user.id, # Assign to the user
                'title': 'Programming Assignment 1 - Data Structures',
                'file_type': 'pdf',
                'course_id': course.id,
                'file_path': './data/documents/sample_assignment.pdf',
                'is_processed': True,
                'processing_status': 'completed'
            },
            {
                'user_id': user.id, # Assign to the user
                'title': 'Midterm Quiz - Database Concepts',
                'file_type': 'pdf',
                'course_id': course.id,
                'file_path': './data/documents/sample_quiz.pdf',
                'is_processed': True,
                'processing_status': 'completed'
            },
            {
                'user_id': user.id, # Assign to the user
                'title': 'Chapter 5 - Advanced SQL Notes',
                'file_type': 'pdf',
                'course_id': course.id,
                'file_path': './data/documents/sample_notes.pdf',
                'is_processed': True,
                'processing_status': 'completed'
            }
        ]

        created_docs = []
        for doc_data in sample_docs:
            # Check if document already exists
            existing = session.query(Document).filter(
                Document.title == doc_data['title'],
                Document.course_id == doc_data['course_id']
            ).first()

            if not existing:
                doc = Document(**doc_data)
                session.add(doc)
                session.flush()
                created_docs.append(doc)
                logger.info(f"📄 Created sample document: {doc.title}")
            else:
                created_docs.append(existing)
                logger.info(f"📄 Using existing document: {existing.title}")

        session.commit()
        return created_docs

async def test_notification_types():
    """Test different types of intelligent notifications"""

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🧪 Testing Enhanced Notification System...")

    try:
        # Get or create sample documents
        logger.info("📄 Setting up sample documents...")
        documents = create_sample_documents()

        if not documents:
            logger.error("❌ No documents available for testing")
            return False

        # Get a user to send notifications to and test within same session
        with db_manager.get_session() as session:
            from src.data.models import User, Course

            user = session.query(User).filter(
                User.telegram_id.isnot(None)
            ).first()

            if not user:
                logger.error("❌ No user with Telegram ID found for testing")
                return False

            # Reload documents in this session to avoid detached instance errors
            doc_ids = [doc.id for doc in documents]
            documents = session.query(Document).filter(Document.id.in_(doc_ids)).all()

            course = session.query(Course).filter(Course.id == documents[0].course_id).first()

            logger.info(f"👤 Testing with user: {user.telegram_id}")
            logger.info(f"📚 Testing with course: {course.course_name}")

            # Test notification generation for each document type
            notification_service = scheduler_service.notification_service

            for doc in documents:
                logger.info(f"\n🔔 Testing notification for: {doc.title}")

                # Generate notification
                await notification_service.notify_new_materials(
                    user.telegram_id,
                    course.course_name,
                    [doc]
                )

        # Get and display generated notifications
        notifications = notification_service.get_pending_notifications()

        logger.info(f"\n📬 Generated {len(notifications)} notifications:")

        for i, notification in enumerate(notifications, 1):
            logger.info(f"\n📧 Notification {i}:")
            logger.info(f"  👤 User: {notification['user_telegram_id']}")
            logger.info(f"  📋 Type: {notification.get('material_type', 'unknown')}")
            logger.info(f"  📄 Document: {notification.get('document_title', 'unknown')}")
            logger.info(f"  🔘 Interactive: {notification.get('interactive', False)}")
            logger.info(f"  💬 Message Preview: {notification['message'][:100]}...")

        if notifications:
            logger.info(f"\n✅ Enhanced notification system test PASSED!")
            logger.info(f"📤 {len(notifications)} intelligent notifications generated with:")
            logger.info("  📝 Assignment help buttons")
            logger.info("  🧠 Quiz study assistance")
            logger.info("  📖 Reading material summaries")
            logger.info("  🤖 Context-aware suggestions")

            logger.info("\n🎯 When users receive these notifications via Telegram, they can:")
            logger.info("  • Click buttons for instant help")
            logger.info("  • Get targeted assistance based on material type")
            logger.info("  • Access AI-powered study support")

            return True
        else:
            logger.warning("⚠️ No notifications were generated")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

async def main():
    """Main test function"""
    success = await test_notification_types()
    return success

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)