import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import schedule
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings
from config.database import db_manager
from src.data.models import Course, Document, User, CourseEnrollment
from src.services.lms_integration import lms_service
from src.core.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

class NotificationService:
    """Service for sending notifications to users"""

    def __init__(self):
        self.telegram_notifications = []  # Store notifications to be sent

    def notify_new_materials(self, user_telegram_id: str, course_name: str,
                            documents: List[Document], material_type: str = None):
        """
        Queue notification about new materials with interactive buttons

        Args:
            user_telegram_id: Telegram user ID
            course_name: Name of the course
            documents: List of new documents
            material_type: Type of material (assignment, quiz, reading, announcement)
        """
        if not documents:
            return

        # Determine material type from documents if not provided
        if not material_type and documents:
            material_type = documents[0].material_type or 'reading'

        # Create notification message based on material type
        if material_type == 'assignment':
            emoji = "📝"
            action = "Assignment posted"
        elif material_type == 'quiz':
            emoji = "🧠"
            action = "Quiz posted"
        elif material_type == 'announcement':
            emoji = "📢"
            action = "Announcement posted"
        else:
            emoji = "📚"
            action = "New material posted"

        # Create document list
        doc_list = "\n".join([f"• {doc.title}" for doc in documents[:5]])
        extra_count = len(documents) - 5

        message = f"{emoji} **{action} in {course_name}:**\n\n{doc_list}"

        if extra_count > 0:
            message += f"\n\n*...and {extra_count} more documents*"

        # Queue the notification with interactive buttons
        notification = {
            'user_telegram_id': user_telegram_id,
            'message': message,
            'timestamp': datetime.now(),
            'interactive': True,  # Enable interactive buttons
            'material_type': material_type,
            'document_id': documents[0].id if documents else None,
            'document_title': documents[0].title if documents else None,
            'course_name': course_name
        }

        self.telegram_notifications.append(notification)
        logger.info(f"Queued {material_type} notification for user {user_telegram_id} about {len(documents)} materials")

    def get_pending_notifications(self) -> List[Dict[str, Any]]:
        """Get pending notifications without clearing (to avoid race conditions)"""
        return self.telegram_notifications.copy()

    def clear_notification(self, notification: Dict[str, Any]):
        """Clear a specific notification after it's been sent"""
        try:
            self.telegram_notifications.remove(notification)
        except ValueError:
            pass  # Already removed

class SchedulerService:
    """Service for scheduling and managing automated tasks"""

    def __init__(self):
        self.notification_service = NotificationService()
        self.rag_pipeline = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.gmail_monitor = None  # Will be initialized when needed
        self.event_loop = None  # Store event loop for async operations

        # Initialize RAG pipeline
        try:
            self.rag_pipeline = RAGPipeline()
            logger.info("RAG pipeline initialized in scheduler")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for async operations"""
        self.event_loop = loop
        logger.info("Event loop set for scheduler")
    
    def start(self):
        """Start the scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        
        # Schedule tasks
        self._schedule_tasks()
        
        # Start scheduler in a separate thread
        scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Scheduler started successfully")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        schedule.clear()
        self.executor.shutdown(wait=True)
        logger.info("Scheduler stopped")
    
    def _schedule_tasks(self):
        """Set up scheduled tasks"""

        # Schedule Gmail monitoring (every 2 minutes for real-time notifications)
        schedule.every(2).minutes.do(self._check_gmail_notifications)

        # Schedule LMS sync less frequently now (every 12 hours - Gmail handles real-time)
        sync_interval = getattr(settings, 'LMS_SYNC_INTERVAL_MINUTES', 720)
        schedule.every(sync_interval).minutes.do(self._sync_lms_content)

        # Schedule document processing every 3 minutes
        schedule.every(3).minutes.do(self._process_pending_documents)

        # Schedule daily cleanup at 2 AM
        schedule.every().day.at("02:00").do(self._daily_cleanup)

        # Schedule weekly full sync every Sunday at 3 AM
        schedule.every().sunday.at("03:00").do(self._weekly_full_sync)

        logger.info(f"Scheduled tasks configured - Gmail check every 2 min, LMS sync every {sync_interval} min")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        logger.info("Scheduler loop started")
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
    
    def _check_gmail_notifications(self):
        """Check Gmail for new Classroom notifications and trigger immediate sync"""
        try:
            # Lazy initialize Gmail monitor
            if not self.gmail_monitor:
                from src.services.gmail_monitor import GmailMonitorService
                from src.services.oauth_manager import oauth_manager
                self.gmail_monitor = GmailMonitorService(oauth_manager)
                logger.info("Gmail monitor initialized")

            logger.debug("Checking Gmail for Classroom notifications...")

            # Check all connected users for new emails
            user_notifications = self.gmail_monitor.check_all_connected_users()

            if not user_notifications:
                return

            logger.info(f"Found Gmail notifications for {len(user_notifications)} users")

            # Process each user's notifications
            for user_id, notifications in user_notifications.items():
                for notification in notifications:
                    try:
                        # Trigger immediate sync for this specific course
                        sync_result = lms_service.sync_specific_user_course(
                            user_id=user_id,
                            course_name_hint=notification.get('course_name'),
                            material_type=notification.get('material_type')
                        )

                        if sync_result['success'] and sync_result['new_documents']:
                            # Immediately process and notify
                            self._process_pending_documents()
                            self._notify_new_documents_sync(
                                user_id,
                                sync_result['new_documents'],
                                sync_result.get('course_name', 'Unknown Course'),
                                notification.get('material_type')
                            )

                            logger.info(f"✅ Processed Gmail trigger for user {user_id}: {len(sync_result['new_documents'])} new docs")

                    except Exception as e:
                        logger.error(f"Error processing Gmail notification for user {user_id}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error checking Gmail notifications: {e}")

    def _sync_lms_content(self):
        """Sync content from all LMS platforms (backup/fallback method)"""
        try:
            logger.info("Starting scheduled LMS content sync...")

            # Get sync stats
            stats = lms_service.sync_all_materials()

            if stats['documents_synced'] > 0:
                logger.info(f"LMS sync completed: {stats}")

                # Immediately trigger document processing for new documents
                self._process_pending_documents()

                # Send notifications for new documents (use sync version)
                self._schedule_notifications_from_sync()
            else:
                logger.info("LMS sync completed - no new documents")

        except Exception as e:
            logger.error(f"Error during LMS sync: {e}")
    
    def _notify_new_documents_sync(self, user_id: int, documents: List[Document],
                                   course_name: str, material_type: str = None):
        """Send notification for specific user and documents (synchronous)"""
        try:
            with db_manager.get_session() as session:
                # Get user's telegram ID
                user = session.query(User).filter(User.id == user_id).first()
                if not user or not user.telegram_id:
                    logger.warning(f"User {user_id} not found or has no Telegram ID")
                    return

                # Filter out documents that were already notified
                unsent_docs = [doc for doc in documents if not doc.notification_sent]

                if not unsent_docs:
                    logger.debug(f"All documents already notified for user {user_id}")
                    return

                # Send notification
                self.notification_service.notify_new_materials(
                    user.telegram_id,
                    course_name,
                    unsent_docs,
                    material_type
                )

                # Mark documents as notified
                for doc in unsent_docs:
                    doc.notification_sent = True
                    doc.notification_sent_at = datetime.now()

                session.commit()

                logger.info(f"✅ Notification queued for user {user.telegram_id}: {len(unsent_docs)} documents")

        except Exception as e:
            logger.error(f"Error sending notification for user {user_id}: {e}")

    def _schedule_notifications_from_sync(self):
        """Send notifications for recently synced documents (fallback method)"""
        try:
            with db_manager.get_session() as session:
                # Get documents that haven't been notified yet (within last hour)
                recent_cutoff = datetime.now() - timedelta(hours=1)

                recent_docs = session.query(Document).filter(
                    Document.created_at >= recent_cutoff,
                    Document.notification_sent == False
                ).all()

                if not recent_docs:
                    logger.debug("No unnotified recent documents")
                    return

                logger.info(f"Found {len(recent_docs)} unnotified documents")

                # Group by course
                docs_by_course = {}
                for doc in recent_docs:
                    if doc.course_id not in docs_by_course:
                        docs_by_course[doc.course_id] = []
                    docs_by_course[doc.course_id].append(doc)

                # Notify enrolled users
                for course_id, documents in docs_by_course.items():
                    course = session.query(Course).filter(Course.id == course_id).first()
                    if not course:
                        continue

                    enrollments = session.query(CourseEnrollment).filter(
                        CourseEnrollment.course_id == course_id,
                        CourseEnrollment.is_active == True
                    ).all()

                    for enrollment in enrollments:
                        user = session.query(User).filter(User.id == enrollment.user_id).first()
                        if user and user.telegram_id:
                            # Determine material type from documents
                            material_type = documents[0].material_type if documents else 'reading'

                            self.notification_service.notify_new_materials(
                                user.telegram_id,
                                course.course_name,
                                documents,
                                material_type
                            )

                            # Mark as notified
                            for doc in documents:
                                doc.notification_sent = True
                                doc.notification_sent_at = datetime.now()

                session.commit()

        except Exception as e:
            logger.error(f"Error scheduling notifications from sync: {e}")
    
    def _process_pending_documents(self):
        """Process documents that haven't been processed yet"""
        try:
            if not self.rag_pipeline:
                logger.warning("RAG pipeline not available for document processing")
                return

            with db_manager.get_session() as session:
                # Get pending documents (limit to 10 at a time for better throughput)
                pending_docs = session.query(Document).filter(
                    Document.is_processed == False,
                    Document.processing_status == "pending"
                ).order_by(Document.created_at.desc()).limit(10).all()  # Process newest first

                if not pending_docs:
                    # Also check for documents that have been "processing" for too long
                    stuck_cutoff = datetime.now() - timedelta(hours=1)  # Consider stuck after 1 hour

                    stuck_docs = session.query(Document).filter(
                        Document.is_processed == False,
                        Document.processing_status == "processing",
                        Document.updated_at <= stuck_cutoff
                    ).limit(5).all()

                    if stuck_docs:
                        logger.info(f"Found {len(stuck_docs)} stuck processing documents, resetting to pending")
                        for doc in stuck_docs:
                            doc.processing_status = "pending"
                        session.commit()
                        pending_docs = stuck_docs
                    else:
                        return

                logger.info(f"Processing {len(pending_docs)} pending documents...")
                processed_count = 0

                for doc in pending_docs:
                    try:
                        # Verify file exists before processing
                        if not doc.file_path or not Path(doc.file_path).exists():
                            logger.warning(f"File not found for document {doc.title}: {doc.file_path}")
                            doc.processing_status = "failed"
                            session.commit()
                            continue

                        # Update status to processing with timestamp
                        doc.processing_status = "processing"
                        doc.updated_at = datetime.now()
                        session.commit()

                        # Process document
                        logger.info(f"Processing document: {doc.title} ({doc.file_type})")
                        success = self.rag_pipeline.process_and_store_document(doc.id)

                        if success:
                            processed_count += 1
                            logger.info(f"✅ Successfully processed document: {doc.title}")
                        else:
                            doc.processing_status = "failed"
                            doc.updated_at = datetime.now()
                            session.commit()
                            logger.error(f"❌ Failed to process document: {doc.title}")

                    except Exception as e:
                        doc.processing_status = "failed"
                        doc.updated_at = datetime.now()
                        session.commit()
                        logger.error(f"❌ Error processing document {doc.title}: {e}", exc_info=True)

                if processed_count > 0:
                    logger.info(f"✅ Document processing completed: {processed_count}/{len(pending_docs)} successful")
                else:
                    logger.info(f"❌ Document processing completed: 0/{len(pending_docs)} successful")
        
        except Exception as e:
            logger.error(f"Error in document processing task: {e}")
    
    def _daily_cleanup(self):
        """Perform daily cleanup tasks"""
        try:
            logger.info("Starting daily cleanup...")
            
            with db_manager.get_session() as session:
                # Clean up old failed processing attempts (older than 7 days)
                cutoff_date = datetime.now() - timedelta(days=7)
                
                failed_docs = session.query(Document).filter(
                    Document.processing_status == "failed",
                    Document.updated_at <= cutoff_date
                ).all()
                
                # Reset status to pending for retry
                retry_count = 0
                for doc in failed_docs:
                    doc.processing_status = "pending"
                    retry_count += 1
                
                session.commit()
                
                if retry_count > 0:
                    logger.info(f"Reset {retry_count} failed documents for retry")
                
                # Clean up old interaction logs (older than 30 days) to save space
                old_interactions_cutoff = datetime.now() - timedelta(days=30)
                
                from src.data.models import UserInteraction
                old_interactions = session.query(UserInteraction).filter(
                    UserInteraction.created_at <= old_interactions_cutoff
                ).count()
                
                # Delete old interactions (keep last 1000 per user)
                if old_interactions > 0:
                    # This is a simplified cleanup - in production you'd want more sophisticated logic
                    session.query(UserInteraction).filter(
                        UserInteraction.created_at <= old_interactions_cutoff
                    ).delete()
                    session.commit()
                    logger.info(f"Cleaned up {old_interactions} old interaction records")
                
                logger.info("Daily cleanup completed")
        
        except Exception as e:
            logger.error(f"Error during daily cleanup: {e}")
    
    def _weekly_full_sync(self):
        """Perform weekly full synchronization"""
        try:
            logger.info("Starting weekly full sync...")
            
            # Force a complete sync of all LMS content
            stats = lms_service.sync_all_materials()
            
            # Get RAG pipeline statistics
            if self.rag_pipeline:
                rag_stats = self.rag_pipeline.get_vector_store_stats()
                logger.info(f"Vector store stats: {rag_stats}")
            
            # Log sync summary
            logger.info(f"Weekly sync completed - Courses: {stats['courses_synced']}, Documents: {stats['documents_synced']}")
            
        except Exception as e:
            logger.error(f"Error during weekly full sync: {e}")
    
    def force_sync_now(self) -> Dict[str, Any]:
        """Force an immediate sync (useful for testing or manual triggers)"""
        try:
            logger.info("🔄 Forcing immediate LMS sync...")
            stats = lms_service.sync_all_materials()

            # Immediately trigger document processing multiple times to ensure all docs are processed
            processed_docs = 0
            max_attempts = 3  # Process up to 3 rounds

            for attempt in range(max_attempts):
                logger.info(f"🔄 Document processing attempt {attempt + 1}/{max_attempts}")
                initial_pending = self._get_pending_document_count()

                if initial_pending == 0:
                    break

                self._process_pending_documents()

                final_pending = self._get_pending_document_count()
                processed_this_round = initial_pending - final_pending
                processed_docs += processed_this_round

                logger.info(f"📊 Processed {processed_this_round} documents this round, {final_pending} still pending")

                # If no progress was made, break
                if processed_this_round == 0:
                    break

            # Get final processing stats
            processing_stats = self._get_processing_stats()

            return {
                'success': True,
                'stats': stats,
                'documents_processed': processed_docs,
                'processing_stats': processing_stats,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            logger.error(f"Error during forced sync: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status"""
        try:
            with db_manager.get_session() as session:
                # Count documents by processing status
                total_docs = session.query(Document).count()
                processed_docs = session.query(Document).filter(Document.is_processed == True).count()
                pending_docs = session.query(Document).filter(Document.processing_status == "pending").count()
                processing_docs = session.query(Document).filter(Document.processing_status == "processing").count()
                failed_docs = session.query(Document).filter(Document.processing_status == "failed").count()
                
                # Count courses
                total_courses = session.query(Course).count()
                active_courses = session.query(Course).filter(Course.is_active == True).count()
                
                # Get platform connection status
                platforms = lms_service.get_available_platforms()
                
                return {
                    'running': self.running,
                    'connected_platforms': platforms,
                    'documents': {
                        'total': total_docs,
                        'processed': processed_docs,
                        'pending': pending_docs,
                        'processing': processing_docs,
                        'failed': failed_docs
                    },
                    'courses': {
                        'total': total_courses,
                        'active': active_courses
                    },
                    'rag_available': self.rag_pipeline is not None,
                    'last_update': datetime.now()
                }
        
        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return {
                'running': self.running,
                'error': str(e),
                'last_update': datetime.now()
            }
    
    def get_pending_notifications(self) -> List[Dict[str, Any]]:
        """Get pending notifications (for the bot to process)"""
        return self.notification_service.get_pending_notifications()

    def _get_pending_document_count(self) -> int:
        """Get count of pending documents"""
        try:
            with db_manager.get_session() as session:
                return session.query(Document).filter(
                    Document.is_processed == False,
                    Document.processing_status == "pending"
                ).count()
        except Exception as e:
            logger.error(f"Error getting pending document count: {e}")
            return 0

    def _get_processing_stats(self) -> Dict[str, int]:
        """Get comprehensive document processing statistics"""
        try:
            with db_manager.get_session() as session:
                total = session.query(Document).count()
                processed = session.query(Document).filter(Document.is_processed == True).count()
                pending = session.query(Document).filter(
                    Document.is_processed == False,
                    Document.processing_status == "pending"
                ).count()
                processing = session.query(Document).filter(
                    Document.processing_status == "processing"
                ).count()
                failed = session.query(Document).filter(
                    Document.processing_status == "failed"
                ).count()

                return {
                    'total': total,
                    'processed': processed,
                    'pending': pending,
                    'processing': processing,
                    'failed': failed,
                    'success_rate': f"{(processed/total*100):.1f}%" if total > 0 else "0%"
                }
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {'error': str(e)}

    def force_process_documents(self) -> Dict[str, Any]:
        """Force immediate document processing without LMS sync"""
        try:
            logger.info("🔄 Forcing immediate document processing...")

            initial_stats = self._get_processing_stats()
            processed_docs = 0
            max_attempts = 5  # Process up to 5 rounds

            for attempt in range(max_attempts):
                logger.info(f"🔄 Document processing attempt {attempt + 1}/{max_attempts}")
                initial_pending = self._get_pending_document_count()

                if initial_pending == 0:
                    logger.info("✅ No pending documents to process")
                    break

                self._process_pending_documents()

                final_pending = self._get_pending_document_count()
                processed_this_round = initial_pending - final_pending
                processed_docs += processed_this_round

                logger.info(f"📊 Processed {processed_this_round} documents this round, {final_pending} still pending")

                # If no progress was made, break
                if processed_this_round == 0:
                    logger.warning(f"⚠️ No progress in round {attempt + 1}, stopping")
                    break

            final_stats = self._get_processing_stats()

            return {
                'success': True,
                'documents_processed': processed_docs,
                'initial_stats': initial_stats,
                'final_stats': final_stats,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"❌ Error during forced document processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }

# Global scheduler instance
scheduler_service = SchedulerService()