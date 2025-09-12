import logging
from typing import Any, Dict, Optional
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from sqlalchemy.orm import Session

from src.data.models import User, UserInteraction, PersonalizationProfile
from config.database import db_manager
from config.settings import settings
#For RAG functionality
from src.core.rag_pipeline import RAGPipeline
#For LMS integration functionality
from src.services.scheduler import scheduler_service
from src.services.lms_integration import lms_service
from src.bot import commands as bot_commands
from src.data.models import User, UserInteraction, PersonalizationProfile, Course, Document, CourseEnrollment

logger = logging.getLogger(__name__)

class StudyHelperBot:
    """Main bot class encapsulating all bot functionality"""
    
    def __init__(self):
        self.application = None
        self.rag_pipeline = None
        self._setup_application()
        self._initialize_rag()
        self._start_notification_handler()
    
    def _setup_application(self):
        """Initialize the bot application"""
        self.application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
        
        # Add handlers
        self._add_handlers()
        logger.info("Telegram bot application initialized")
        
    def _initialize_rag(self):
        """Initialize the RAG pipeline"""
        try:
            self.rag_pipeline = RAGPipeline()
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            self.rag_pipeline = None

    def _start_notification_handler(self):
        """Start the notification handler job"""
        try:
            # Schedule notification checks every minute
            job_queue = self.application.job_queue
            if job_queue:
                job_queue.run_repeating(
                    self._process_notifications, 
                    interval=60,  # Check every minute
                    first=10  # Start after 10 seconds
                )
                logger.info("Notification handler started")
        except Exception as e:
            logger.error(f"Failed to start notification handler: {e}")
    
    async def _process_notifications(self, context: ContextTypes.DEFAULT_TYPE):
        """Process pending notifications from scheduler"""
        try:
            notifications = scheduler_service.get_pending_notifications()
            
            for notification in notifications:
                try:
                    await context.bot.send_message(
                        chat_id=notification['user_telegram_id'],
                        text=notification['message'],
                        parse_mode='Markdown'
                    )
                    logger.info(f"Sent notification to user {notification['user_telegram_id']}")
                    
                except Exception as e:
                    logger.error(f"Failed to send notification to {notification['user_telegram_id']}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing notifications: {e}")
        
    
    def _add_handlers(self):
        """Add all command and message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("profile", self.profile_command))
        self.application.add_handler(CommandHandler("courses", self.courses_command))
        self.application.add_handler(CommandHandler("settings", self.settings_command))

        # Classroom related
        self.application.add_handler(CommandHandler("connect_classroom", self.connect_classroom_command))
        self.application.add_handler(CommandHandler("disconnect_classroom", self.disconnect_classroom_command))
        self.application.add_handler(CommandHandler("connections", self.connection_status_command))
    
        # Admin/Debug commands
        self.application.add_handler(CommandHandler("sync", self.sync_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
                
        # Message handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_query))
        
        # Callback query handlers (for inline keyboards)
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command (delegates to bot_commands)"""
        await bot_commands.start_command(update, context)
    
    async def sync_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sync command for manual synchronization (delegates)"""
        await bot_commands.sync_command(update, context)
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command (delegates)"""
        await bot_commands.status_command(update, context)
    

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command (delegates)"""
        await bot_commands.help_command(update, context)
    
    async def profile_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /profile command (delegates)"""
        await bot_commands.profile_command(update, context)
    
    async def courses_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /courses command (delegates)"""
        await bot_commands.courses_command(update, context)

    async def _courses_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        return await bot_commands.courses_command(
            update,
            context,
            get_user_by_telegram_id=self._get_user_by_telegram_id,
            log_interaction=self._log_interaction
        )
 
    async def connect_classroom_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /connect_classroom command (delegates)"""
        await bot_commands.connect_classroom_command(update, context)

    async def disconnect_classroom_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /disconnect_classroom command (delegates)"""
        await bot_commands.disconnect_classroom_command(update, context)

    async def connection_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /connections command to show LMS connection status (delegates)"""
        await bot_commands.connection_status_command(update, context)

    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command (delegates)"""
        await bot_commands.settings_command(update, context)
    
    async def handle_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle general text queries - main AI interaction"""
        user_data = update.effective_user
        query_text = update.message.text
        
        try:
            # Show typing indicator
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            with db_manager.get_session() as session:
                user = self._get_or_create_user(session, user_data)
                
                # For now, provide a simple response since we haven't implemented RAG yet
                # This will be replaced with actual AI processing later
                #response_text = await self._process_query_basic(query_text, user)
                response_text = await self._process_query_rag_enhanced(query_text, user)
                
                # Send response
                await update.message.reply_text(response_text, parse_mode='Markdown')
                
                # Create feedback buttons
                keyboard = [
                    [
                        InlineKeyboardButton("👍 Helpful", callback_data=f"feedback_helpful_{len(query_text)}"),
                        InlineKeyboardButton("👎 Not helpful", callback_data=f"feedback_unhelpful_{len(query_text)}"),
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(
                    "Was this response helpful?", 
                    reply_markup=reply_markup
                )
                
                # Log interaction
                #self._log_interaction(
                #   session, user.id, query_text, response_text, "question"
                #)
                self._log_interaction(
                    session, user.id, query_text, response_text, "rag_query"
                )
                
        except Exception as e:
            logger.error(f"Error in handle_query: {e}")
            await update.message.reply_text(
                "Sorry, I encountered an error processing your question. Please try again."
            )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        query = update.callback_query
        await query.answer()  # Acknowledge the callback
        
        callback_data = query.data  
        
        try:
            if callback_data == "view_courses":
                #await self.courses_command(update, context)
                user_data = query.from_user
            
                with db_manager.get_session() as session:
                    user = session.query(User).filter(User.telegram_id == str(user_data.id)).first()
                    
                    if not user:
                        await query.edit_message_text("Please use /start first to create your profile.")
                        return
                    
                    # Simple courses display for now
                    await query.edit_message_text("📚 **Your Courses**\n\nFetching course information...")
                

            elif callback_data == "sync_now":
                await query.edit_message_text("🔄 Starting sync...")
                try:
                    result = scheduler_service.force_sync_now()
                    if result.get('success'):
                        stats = result.get('stats', {})
                        message = f"✅ Sync completed!\nCourses: {stats.get('courses_synced', 0)}, Documents: {stats.get('documents_synced', 0)}"
                    else:
                        message = f"❌ Sync failed: {result.get('error', 'Unknown error')}"
                    await query.edit_message_text(message)
                except Exception as sync_error:
                    logger.error(f"Sync error: {sync_error}")
                    await query.edit_message_text("❌ Sync failed due to internal error.")
            
            #oauth callbacks
            elif callback_data == "confirm_disconnect_gc":
                user_data = update.effective_user
                
                with db_manager.get_session() as session:
                    user = session.query(User).filter(
                        User.telegram_id == str(user_data.id)
                    ).first()
                    
                    if user:
                        # Clear credentials
                        user.google_credentials = None
                        user.google_classroom_connected = False
                        session.commit()
                        
                        await query.edit_message_text(
                            "Google Classroom has been disconnected successfully.\n\n"
                            "Use /connect_classroom to reconnect anytime."
                        )
                    else:
                        await query.edit_message_text("User not found.")
            
            elif callback_data == "cancel_disconnect":
                await query.edit_message_text("Disconnection cancelled.")
            
            elif callback_data == "status":
                #await self.status_command(update, context)
                try:
                    status = scheduler_service.get_sync_status()
                    status_text = f"📊 **System Status**\n\n**Scheduler:** {'🟢 Running' if status.get('running') else '🔴 Stopped'}\n**Documents:** {status.get('documents', {}).get('total', 0)} total"
                    await query.edit_message_text(status_text, parse_mode='Markdown')
                except Exception as status_error:
                    logger.error(f"Status error: {status_error}")
                    await query.edit_message_text("❌ Failed to get system status.")
            
            elif callback_data == "sync_courses":
                await query.edit_message_text("🔄 Refreshing courses...")
                result = scheduler_service.force_sync_now()
                await query.edit_message_text("✅ Courses refreshed! Use /courses to see updated list.")
                    
            elif callback_data == "settings":
                await self.settings_command(update, context)
            
            elif callback_data == "help":
                await self.help_command(update, context)
            
            elif callback_data == "profile":
                await self.profile_command(update, context)
            
            elif callback_data.startswith("feedback_"):
                await self._handle_feedback(update, context, callback_data)
            
            elif callback_data.startswith("setting_"):
                await self._handle_setting_change(update, context, callback_data)
            
            elif callback_data == "enable_notifications":
                await query.edit_message_text(
                    "🔔 Notifications enabled! I'll notify you when new course materials are available.",
                    parse_mode='Markdown'
                )
            
            else:
                await query.edit_message_text("Feature coming soon! Though sifai kujibu hivi mkuu. \n\n Kwa handle_callback")
                
        except Exception as e:
            logger.error(f"Error in handle_callback: {e}")
            await query.edit_message_text("Sorry, something went wrong. Please try again. \n\n Kwa handle_callback superior")


    
    async def _process_query_rag_enhanced(self, query: str, user: User) -> str:
        """Enhanced query processing with RAG pipeline"""
        if not self.rag_pipeline:
            logger.error("RAG pipeline not initialized")
            return await self._process_query_basic(query, user)  # Fallback to basic
        
        try:
            # Determine course context from query
            course_id = self._extract_course_context(query, user)
            
            # Generate context using RAG
            rag_context = self.rag_pipeline.generate_context(query, course_id)
            
            if rag_context['has_relevant_content']:
                response = await self._generate_rag_response(query, rag_context, user)
            else:
                response = await self._generate_fallback_response(query, user)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG-enhanced processing: {e}")
            return await self._process_query_basic(query, user)

    def _extract_course_context(self, query: str, user: User) -> Optional[int]:
        """Extract course context from query (simple keyword matching for now)"""
        query_lower = query.lower()
        
        # Simple course code detection
        course_keywords = {
            'ics201': 1,  # Assuming course ID 1 for ICS201
            'ics301': 2,  # Assuming course ID 2 for ICS301  
            'mat201': 3,  # Assuming course ID 3 for MAT201
            'data structures': 1,
            'software engineering': 2,
            'discrete mathematics': 3
        }
        
        for keyword, course_id in course_keywords.items():
            if keyword in query_lower:
                return course_id
        
        return None

    async def _generate_rag_response(self, query: str, rag_context: Dict[str, Any], user: User) -> str:
        """Generate response using RAG context"""
        context = rag_context['context']
        sources = rag_context['sources']
        
        # For now, create a structured response based on retrieved context
        # Later, you'll replace this with actual LLaMA generation
        
        response_parts = []
        
        # Add main response based on context
        if any(word in query.lower() for word in ['what is', 'define', 'explain']):
            response_parts.append(f"Based on your course materials:\n\n{context[:800]}...")
        else:
            response_parts.append(f"Here's what I found in your course materials:\n\n{context[:600]}...")
        
        # Add sources
        if sources:
            response_parts.append("\n**Sources:**")
            for i, source in enumerate(sources[:3], 1):
                score_percent = int(source['similarity_score'] * 100)
                response_parts.append(f"{i}. {source['title']} ({source['course_code']}) - {score_percent}% match")
        
        # Add personalized note
        response_parts.append(f"\n*This response is tailored for your {user.learning_style} learning style.*")
        
        return "\n".join(response_parts)

    async def _generate_fallback_response(self, query: str, user: User) -> str:
        """Generate fallback response when no relevant content found"""
        return f"""I couldn't find specific information about "{query}" in your uploaded course materials.

**Suggestions:**
• Try asking about topics from your enrolled courses (ICS 201, ICS 301, MAT 201)
• Check if your instructors have uploaded relevant materials to the LMS
• Use more specific keywords related to your coursework

I'm continuously learning from new materials as they're added to your courses. Try asking again later!

*Use /help for tips on asking better questions.*"""
    
    async def _process_query_basic(self, query: str, user: User) -> str:
        """Basic query processing - will be replaced with RAG pipeline later"""
        query_lower = query.lower()
        
        # Simple keyword-based responses for testing
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return f"Hello {user.first_name}! How can I help you with your studies today?"
        
        elif any(word in query_lower for word in ['data structure', 'stack', 'queue']):
            return """
📚 **Data Structures Overview**

**Stack (LIFO - Last In, First Out):**
• Elements added/removed from the top
• Main operations: push(), pop(), peek()
• Use cases: Function calls, expression evaluation

**Queue (FIFO - First In, First Out):**  
• Elements added at rear, removed from front
• Main operations: enqueue(), dequeue(), front()
• Use cases: Task scheduling, breadth-first search

*Note: This is a basic response. Full AI-powered responses coming soon!*
            """.strip()
        
        elif any(word in query_lower for word in ['algorithm', 'sort', 'search']):
            return """
🔍 **Algorithms Basics**

**Common Sorting Algorithms:**
• Bubble Sort: O(n²) - Simple but inefficient
• Quick Sort: O(n log n) - Efficient divide-and-conquer
• Merge Sort: O(n log n) - Stable and consistent

**Search Algorithms:**
• Linear Search: O(n) - Sequential checking
• Binary Search: O(log n) - Requires sorted data

*Full course material analysis coming with RAG implementation!*
            """.strip()
        
        elif any(word in query_lower for word in ['assignment', 'deadline', 'due']):
            return """
📋 **Assignment Tracking**

*Currently showing mock data - LMS integration in progress:*

**Upcoming Deadlines:**
• ICS 201: Data Structures Assignment - Due March 15
• ICS 301: Software Design Document - Due March 20  
• MAT 201: Problem Set 3 - Due March 18

I'll soon be able to automatically track real assignments from your LMS!
            """.strip()
        
        else:
            return f"""
I understand you're asking about: "{query}"

🚧 **AI Processing Coming Soon!**

I'm currently in development mode. Soon I'll be able to:
• Search through your actual course materials
• Provide detailed, contextual answers
• Reference specific lecture notes and textbooks
• Adapt responses to your learning style

For now, try asking about:
• Data structures (stack, queue)
• Algorithms (sorting, searching)  
• Assignments and deadlines

Use /help for available commands!
            """
    
    async def _handle_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str):
        """Handle user feedback on responses"""
        query = update.callback_query
        
        # Parse feedback
        is_helpful = "helpful" in callback_data and "unhelpful" not in callback_data
        
        try:
            user_data = update.effective_user
            with db_manager.get_session() as session:
                # Find the most recent interaction for this user
                user = session.query(User).filter(User.telegram_id == str(user_data.id)).first()
                if user:
                    recent_interaction = session.query(UserInteraction).filter(
                        UserInteraction.user_id == user.id
                    ).order_by(UserInteraction.created_at.desc()).first()
                    
                    if recent_interaction:
                        recent_interaction.was_helpful = is_helpful
                        recent_interaction.user_rating = 5 if is_helpful else 2
                        session.commit()
            
            feedback_text = "👍 Thanks for the feedback! This helps me learn." if is_helpful else "👎 Thanks for the feedback. I'll work on improving my responses."
            await query.edit_message_text(feedback_text)
            
        except Exception as e:
            logger.error(f"Error handling feedback: {e}")
            await query.edit_message_text("Thanks for the feedback!")
    
    async def _handle_setting_change(self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str):
        """Handle settings changes"""
        query = update.callback_query
        setting_type = callback_data.replace("setting_", "")
        
        if setting_type == "learning_style":
            keyboard = [
                [
                    InlineKeyboardButton("👁️ Visual", callback_data="set_visual"),
                    InlineKeyboardButton("👂 Auditory", callback_data="set_auditory")
                ],
                [
                    InlineKeyboardButton("✋ Kinesthetic", callback_data="set_kinesthetic"),
                    InlineKeyboardButton("🧠 Adaptive", callback_data="set_adaptive")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📖 **Choose Your Learning Style:**\n\n"
                "• **Visual**: Learn better with diagrams and visual aids\n"
                "• **Auditory**: Prefer verbal explanations\n" 
                "• **Kinesthetic**: Learn by doing and examples\n"
                "• **Adaptive**: Let me adapt to your preferences automatically",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif setting_type == "difficulty":
            keyboard = [
                [
                    InlineKeyboardButton("😊 Easy", callback_data="set_easy"),
                    InlineKeyboardButton("😐 Medium", callback_data="set_medium"),
                    InlineKeyboardButton("😅 Hard", callback_data="set_hard")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📊 **Choose Difficulty Level:**\n\n"
                "• **Easy**: Simple explanations with basic examples\n"
                "• **Medium**: Balanced detail with practical examples\n"
                "• **Hard**: Comprehensive explanations with advanced concepts",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        else:
            await query.edit_message_text("Setting configuration coming soon!")
    
    def _get_or_create_user(self, session: Session, user_data) -> User:
        """Get existing user or create new one"""
        user = session.query(User).filter(User.telegram_id == str(user_data.id)).first()
        
        if not user:
            # Create new user
            user = User(
                telegram_id=str(user_data.id),
                username=user_data.username,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
                learning_style="adaptive",
                difficulty_preference="medium"
            )
            session.add(user)
            session.flush()  # Get the user ID
            
            # Create personalization profile
            profile = PersonalizationProfile(
                user_id=user.id,
                total_interactions=0,
                successful_interactions=0,
                avg_session_duration=0.0
            )
            session.add(profile)
            session.commit()
            
            logger.info(f"Created new user: {user_data.first_name} ({user_data.id})")
        
        return user
    
    def _log_interaction(self, session: Session, user_id: int, query: str, response: str, interaction_type: str):
        """Log user interaction for analytics and personalization"""
        interaction = UserInteraction(
            user_id=user_id,
            query_text=query,
            response_text=response,
            interaction_type=interaction_type,
            response_time_ms=0  # Will be calculated properly later
        )
        session.add(interaction)
        
        # Update personalization profile
        profile = session.query(PersonalizationProfile).filter(
            PersonalizationProfile.user_id == user_id
        ).first()
        
        if profile:
            profile.total_interactions += 1
            profile.last_interaction = datetime.now()
        
        session.commit()
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        
        if update.message:
            await update.message.reply_text(
                "Sorry, I encountered an unexpected error. Please try again or use /help for assistance."
            )
    
    def run(self):
        """Start the bot"""
        logger.info("Starting Study Helper Agent bot...")
        self.application.run_polling()
    
    def stop(self):
        """Stop the bot"""
        if self.application:
            self.application.stop()
            logger.info("Bot stopped")