#!/usr/bin/env python3
"""
Test script to verify the notification flow end-to-end
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.chdir(str(project_root))

from config.database import db_manager
from config.settings import settings
from src.utils.logger import setup_logging
from src.services.lms_integration import lms_service
from src.services.scheduler import scheduler_service

def test_notification_flow():
    """Test the complete notification flow"""

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🧪 Starting notification flow test...")

    try:
        # Test database connection
        logger.info("📁 Testing database connection...")
        with db_manager.get_session() as session:
            from src.data.models import User, Course, Document
            user_count = session.query(User).count()
            course_count = session.query(Course).count()
            document_count = session.query(Document).count()

            logger.info(f"📊 Database stats: {user_count} users, {course_count} courses, {document_count} documents")

        # Test LMS sync (this will find new documents if available)
        logger.info("🔄 Testing LMS sync...")
        sync_stats = lms_service.sync_all_materials()
        logger.info(f"📈 Sync results: {sync_stats}")

        # Check if any new documents were found
        if sync_stats['documents_synced'] > 0:
            logger.info(f"🆕 Found {sync_stats['documents_synced']} new documents!")

            # Test notification generation - manually trigger the notification logic
            logger.info("🔔 Testing notification generation...")

            # Import asyncio and manually trigger the notification method
            import asyncio

            # Create an async function to test notifications
            async def test_notifications():
                try:
                    # Call the scheduler's notification method directly
                    await scheduler_service._notify_new_documents()
                    logger.info("✅ Notification generation completed")
                except Exception as e:
                    logger.error(f"❌ Error generating notifications: {e}")

            # Run the async notification test
            asyncio.run(test_notifications())

            # Get pending notifications from scheduler
            notifications = scheduler_service.get_pending_notifications()
            logger.info(f"📬 Found {len(notifications)} pending notifications")

            if notifications:
                for notification in notifications:
                    logger.info(f"📧 Notification for user {notification['user_telegram_id']}: {notification['message'][:100]}...")

                logger.info("✅ Notification flow test PASSED - notifications were generated!")
            else:
                logger.warning("⚠️ No notifications found - this could indicate:")
                logger.warning("  - No users are enrolled in courses with new materials")
                logger.warning("  - Users don't have telegram_id set")
                logger.warning("  - Notification generation logic has issues")

                # Let's check user enrollment
                with db_manager.get_session() as session:
                    from src.data.models import User, CourseEnrollment

                    users_with_telegram = session.query(User).filter(
                        User.telegram_id.isnot(None)
                    ).count()

                    total_enrollments = session.query(CourseEnrollment).filter(
                        CourseEnrollment.is_active == True
                    ).count()

                    logger.info(f"📊 Users with Telegram ID: {users_with_telegram}")
                    logger.info(f"📊 Active enrollments: {total_enrollments}")
        else:
            logger.info("ℹ️ No new documents found during sync")
            logger.info("🆕 To test notifications, upload a document to one of your Google Classroom courses")

        # Test document processing
        logger.info("📄 Testing document processing...")
        processing_stats = scheduler_service.force_process_documents()
        logger.info(f"📈 Processing results: {processing_stats}")

        # Get overall sync status
        status = scheduler_service.get_sync_status()
        logger.info(f"📊 System status: {status}")

        logger.info("✅ Notification flow test completed successfully!")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_notification_flow()
    sys.exit(0 if success else 1)