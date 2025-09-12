import logging
import asyncio
from datetime import datetime, timedelta
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
    
    async def notify_new_materials(self, user_telegram_id: str, course_name: str, documents: List[Document]):
        """Queue notification about new materials"""
        if not documents:
            return
        
        # Create notification message
        doc_list = "\n".join([f"• {doc.title}" for doc in documents[:5]])  # Show max 5
        extra_count = len(documents) - 5
        
        message = f"📚 **New materials posted in {course_name}:**\n\n{doc_list}"
        
        if extra_count > 0:
            message += f"\n\n*...and {extra_count} more documents*"
        
        message += "\n\nUse the bot to ask questions about these materials!"
        
        # Queue the notification (will be processed by bot)
        notification = {
            'user_telegram_id': user_telegram_id,
            'message': message,
            'timestamp': datetime.now()
        }
        
        self.telegram_notifications.append(notification)
        logger.info(f"Queued notification for user {user_telegram_id} about {len(documents)} new materials")
    
    def get_pending_notifications(self) -> List[Dict[str, Any]]:
        """Get and clear pending notifications"""
        notifications = self.telegram_notifications.copy()
        self.telegram_notifications.clear()
        return notifications

class SchedulerService:
    """Service for scheduling and managing automated tasks"""
    
    def __init__(self):
        self.notification_service = NotificationService()
        self.rag_pipeline = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize RAG pipeline
        try:
            self.rag_pipeline = RAGPipeline()
            logger.info("RAG pipeline initialized in scheduler")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
    
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
        
        # Schedule LMS sync every 30 minutes
        schedule.every(30).minutes.do(self._sync_lms_content)
        
        # Schedule document processing every 10 minutes
        schedule.every(10).minutes.do(self._process_pending_documents)
        
        # Schedule daily cleanup at 2 AM
        schedule.every().day.at("02:00").do(self._daily_cleanup)
        
        # Schedule weekly full sync every Sunday at 3 AM
        schedule.every().sunday.at("03:00").do(self._weekly_full_sync)
        
        logger.info("Scheduled tasks configured")
    
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
    
    def _sync_lms_content(self):
        """Sync content from all LMS platforms"""
        try:
            logger.info("Starting LMS content sync...")
            
            # Get sync stats
            stats = lms_service.sync_all_materials()
            
            if stats['documents_synced'] > 0:
                logger.info(f"LMS sync completed: {stats}")
                
                # Send notifications for new documents
                asyncio.create_task(self._notify_new_documents())
            else:
                logger.info("LMS sync completed - no new documents")
        
        except Exception as e:
            logger.error(f"Error during LMS sync: {e}")
    
    async def _notify_new_documents(self):
        """Send notifications about newly synced documents"""
        try:
            with db_manager.get_session() as session:
                # Get documents that were recently added (last 35 minutes to account for sync frequency)
                recent_cutoff = datetime.now() - timedelta(minutes=35)
                
                recent_docs = session.query(Document).filter(
                    Document.created_at >= recent_cutoff,
                    Document.is_processed == False
                ).all()
                
                if not recent_docs:
                    return
                
                # Group documents by course
                docs_by_course = {}
                for doc in recent_docs:
                    if doc.course_id not in docs_by_course:
                        docs_by_course[doc.course_id] = []
                    docs_by_course[doc.course_id].append(doc)
                
                # Send notifications to enrolled users
                for course_id, documents in docs_by_course.items():
                    course = session.query(Course).filter(Course.id == course_id).first()
                    if not course:
                        continue
                    
                    # Get enrolled users
                    enrollments = session.query(CourseEnrollment).filter(
                        CourseEnrollment.course_id == course_id,
                        CourseEnrollment.is_active == True
                    ).all()
                    
                    for enrollment in enrollments:
                        user = session.query(User).filter(User.id == enrollment.user_id).first()
                        if user and user.telegram_id:
                            await self.notification_service.notify_new_materials(
                                user.telegram_id,
                                course.course_name,
                                documents
                            )
        
        except Exception as e:
            logger.error(f"Error sending new document notifications: {e}")
    
    def _process_pending_documents(self):
        """Process documents that haven't been processed yet"""
        try:
            if not self.rag_pipeline:
                logger.warning("RAG pipeline not available for document processing")
                return
            
            with db_manager.get_session() as session:
                # Get pending documents (limit to 5 at a time to avoid overwhelming)
                pending_docs = session.query(Document).filter(
                    Document.is_processed == False,
                    Document.processing_status == "pending"
                ).limit(5).all()
                
                if not pending_docs:
                    return
                
                logger.info(f"Processing {len(pending_docs)} pending documents...")
                processed_count = 0
                
                for doc in pending_docs:
                    try:
                        # Update status to processing
                        doc.processing_status = "processing"
                        session.commit()
                        
                        # Process document
                        success = self.rag_pipeline.process_and_store_document(doc.id)
                        
                        if success:
                            processed_count += 1
                            logger.info(f"Successfully processed document: {doc.title}")
                        else:
                            doc.processing_status = "failed"
                            session.commit()
                            logger.error(f"Failed to process document: {doc.title}")
                    
                    except Exception as e:
                        doc.processing_status = "failed"
                        session.commit()
                        logger.error(f"Error processing document {doc.title}: {e}")
                
                if processed_count > 0:
                    logger.info(f"Document processing completed: {processed_count}/{len(pending_docs)} successful")
        
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
            logger.info("Forcing immediate LMS sync...")
            stats = lms_service.sync_all_materials()
            
            # Trigger document processing
            self._process_pending_documents()
            
            return {
                'success': True,
                'stats': stats,
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

# Global scheduler instance
scheduler_service = SchedulerService()