import asyncio
import logging
from typing import Any, Dict, Optional, List
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
        """Process pending notifications from scheduler with interactive buttons"""
        try:
            notifications = scheduler_service.get_pending_notifications()

            if not notifications:
                return

            for notification in notifications:
                try:
                    # Check if this is an interactive notification
                    if notification.get('interactive', False):
                        # Create inline keyboard based on material type
                        keyboard = self._create_notification_keyboard(notification)
                        reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None

                        await context.bot.send_message(
                            chat_id=notification['user_telegram_id'],
                            text=notification['message'],
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                        logger.info(f"📤 Sent interactive {notification.get('material_type', 'unknown')} notification to user {notification['user_telegram_id']}")
                    else:
                        # Regular notification without buttons
                        await context.bot.send_message(
                            chat_id=notification['user_telegram_id'],
                            text=notification['message'],
                            parse_mode='Markdown'
                        )
                        logger.info(f"📤 Sent notification to user {notification['user_telegram_id']}")

                    # Clear notification after successful send
                    scheduler_service.notification_service.clear_notification(notification)

                except Exception as e:
                    logger.error(f"❌ Failed to send notification to {notification['user_telegram_id']}: {e}")

        except Exception as e:
            logger.error(f"❌ Error processing notifications: {e}")

    def _create_notification_keyboard(self, notification: dict) -> List[List[InlineKeyboardButton]]:
        """Create interactive keyboard for notifications"""
        material_type = notification.get('material_type', 'reading')
        document_title = notification.get('document_title', '')
        course_name = notification.get('course_name', '')
        document_id = notification.get('document_id', '')

        keyboard = []

        if material_type == 'assignment':
            keyboard = [
                [
                    InlineKeyboardButton("📝 Help with Assignment",
                                       callback_data=f"help_assignment_{document_id}"),
                    InlineKeyboardButton("📋 Break Down Tasks",
                                       callback_data=f"breakdown_assignment_{document_id}")
                ],
                [
                    InlineKeyboardButton("🔍 Explain Requirements",
                                       callback_data=f"explain_assignment_{document_id}")
                ]
            ]
        elif material_type == 'quiz':
            keyboard = [
                [
                    InlineKeyboardButton("🧠 Help Me Study",
                                       callback_data=f"study_quiz_{document_id}"),
                    InlineKeyboardButton("❓ Practice Questions",
                                       callback_data=f"practice_quiz_{document_id}")
                ],
                [
                    InlineKeyboardButton("📝 Key Concepts",
                                       callback_data=f"concepts_quiz_{document_id}")
                ]
            ]
        else:  # reading material
            keyboard = [
                [
                    InlineKeyboardButton("📖 Summarize Document",
                                       callback_data=f"summarize_reading_{document_id}"),
                    InlineKeyboardButton("❓ Ask Questions",
                                       callback_data=f"questions_reading_{document_id}")
                ],
                [
                    InlineKeyboardButton("🔍 Key Points",
                                       callback_data=f"keypoints_reading_{document_id}")
                ]
            ]

        return keyboard
        
    
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
        self.application.add_handler(CommandHandler("process_docs", self.process_documents_command))
                
        # Message handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_query))
        
        # Callback query handlers (for inline keyboards)
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_data = update.effective_user

        try:
            with db_manager.get_session() as session:
                user = self._get_or_create_user(session, user_data)

                connected_platforms = lms_service.get_available_platforms()
                platform_status = ", ".join(connected_platforms) if connected_platforms else "No LMS connected"

                welcome_text = f"""
🎓 Welcome to Study Helper Agent, {user.first_name}!

I'm your AI-powered academic assistant. I can help you with:
• 📚 Answering questions about your course materials
• 📋 Tracking assignments and deadlines
• 🔔 Getting notified about new content
• 📈 Personalizing your learning experience

**Connected LMS:** {platform_status}

To get started:
1. Use /courses to see available courses
2. Ask me questions about your studies
3. Use /help for more commands

Example: "What are the main topics in today's lecture notes?"
                """.strip()

                keyboard = [
                    [
                        InlineKeyboardButton("📚 View Courses", callback_data="view_courses"),
                        InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
                    ],
                    [
                        InlineKeyboardButton("🔄 Sync LMS", callback_data="sync_now"),
                        InlineKeyboardButton("📄 Process Docs", callback_data="process_docs"),
                    ],
                    [
                        InlineKeyboardButton("📊 Status", callback_data="status"),
                        InlineKeyboardButton("❓ Help", callback_data="help"),
                    ],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await update.message.reply_text(welcome_text, reply_markup=reply_markup)

                self._log_interaction(session, user.id, "/start", welcome_text, "command")

        except Exception as e:
            logger.error(f"Error in start_command: {e}")
            await update.message.reply_text(
                "Sorry, I encountered an error. Please try again later."
            )
    
    async def sync_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sync command for manual synchronization"""
        try:
            await update.message.reply_text("🔄 Starting manual sync... This may take a moment.")

            result = scheduler_service.force_sync_now()

            if result['success']:
                stats = result['stats']
                response_text = f"""
✅ **Sync completed successfully!**

📊 **Statistics:**
• Courses synced: {stats['courses_synced']}
• Documents synced: {stats['documents_synced']}
• Timestamp: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

Use the bot to ask questions about your updated materials!
                """.strip()
            else:
                response_text = f"""
❌ **Sync failed**

Error: {result['error']}
Timestamp: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

Please check the LMS connection or try again later.
                """.strip()

            await update.message.reply_text(response_text, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error in sync_command: {e}")
            await update.message.reply_text("❌ Sync failed due to an internal error.")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            status = scheduler_service.get_sync_status()

            status_text = f"""
📊 **System Status**

**Scheduler:** {'🟢 Running' if status['running'] else '🔴 Stopped'}
**Connected LMS:** {', '.join(status['connected_platforms']) if status['connected_platforms'] else 'None'}
**RAG Pipeline:** {'🟢 Available' if status['rag_available'] else '🔴 Not Available'}

**Documents:**
• Total: {status['documents']['total']}
• Processed: {status['documents']['processed']} ✅
• Pending: {status['documents']['pending']} ⏳
• Processing: {status['documents']['processing']} 🔄
• Failed: {status['documents']['failed']} ❌

**Courses:**
• Total: {status['courses']['total']}
• Active: {status['courses']['active']}

**Last Update:** {status['last_update'].strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()

            await update.message.reply_text(status_text, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error in status_command: {e}")
            await update.message.reply_text("❌ Failed to get system status.")

    async def process_documents_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /process_docs command for manual document processing"""
        try:
            await update.message.reply_text("🔄 Starting document processing... This may take a moment.")

            result = scheduler_service.force_process_documents()

            if result['success']:
                initial_stats = result['initial_stats']
                final_stats = result['final_stats']
                processed = result['documents_processed']

                response_text = f"""
✅ **Document Processing Completed!**

📊 **Results:**
• Documents Processed: {processed}
• Success Rate: {final_stats.get('success_rate', 'N/A')}

📈 **Before/After:**
• Processed: {initial_stats.get('processed', 0)} → {final_stats.get('processed', 0)}
• Pending: {initial_stats.get('pending', 0)} → {final_stats.get('pending', 0)}
• Failed: {initial_stats.get('failed', 0)} → {final_stats.get('failed', 0)}

Your documents are now ready for AI-powered queries!
                """.strip()
            else:
                response_text = f"""
❌ **Document Processing Failed**

Error: {result['error']}
Timestamp: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

Please check the system status or try again later.
                """.strip()

            await update.message.reply_text(response_text, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error in process_documents_command: {e}")
            await update.message.reply_text("❌ Document processing failed due to an internal error.")
    

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
🆘 **Study Helper Agent Commands**

**Basic Commands:**
/start - Initialize the bot and see welcome message
/help - Show this help message
/profile - View your learning profile
/courses - View enrolled courses
/settings - Adjust your preferences

**LMS Integration:**
/connect_classroom - Connect your Google Classroom
/disconnect_classroom - Disconnect Google Classroom
/connections - View connection status
/sync - Manually sync with your LMS
/status - View system status
/process_docs - Force document processing

**How to Use:**
📝 **Ask Questions**: Just type your question naturally
   Example: "Explain the concept of inheritance in OOP"

🔍 **Search Content**: Ask about specific topics
   Example: "What did the professor say about databases?"

📊 **Get Summaries**: Request summaries of materials
   Example: "Summarize today's lecture on algorithms"

**Tips for Better Results:**
• Be specific about which course or topic
• Ask follow-up questions for clarification
• Rate my responses to improve personalization
• Use /sync to get the latest materials
• Use /settings to customize your experience

**Getting Started:**
1. Use /connect_classroom to link your Google account
2. Use /courses to see your classes
3. Ask questions about your materials
4. Rate responses to improve personalization

Need more help? Just ask me anything!
        """.strip()

        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def profile_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /profile command"""
        user_data = update.effective_user

        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.telegram_id == str(user_data.id)).first()

                if not user:
                    await update.message.reply_text("Please use /start first to initialize your profile.")
                    return

                profile = session.query(PersonalizationProfile).filter(
                    PersonalizationProfile.user_id == user.id
                ).first()

                total_interactions = session.query(UserInteraction).filter(
                    UserInteraction.user_id == user.id
                ).count()

                member_since = user.created_at.strftime('%B %Y') if user.created_at else 'Date not recorded'
                last_active = user.updated_at.strftime('%B %d, %Y') if user.updated_at else 'Date not recorded'

                avg_session = f"{profile.avg_session_duration:.1f} minutes" if profile and profile.avg_session_duration is not None else 'Not recorded'

                profile_text = f"""
👤 **Your Learning Profile**

**Basic Info:**
• Name: {user.first_name} {user.last_name or ''}
• Learning Style: {user.learning_style.title()}
• Preferred Difficulty: {user.difficulty_preference.title()}

**Activity Stats:**
• Total Interactions: {total_interactions}
• Member Since: {member_since}
• Last Active: {last_active}

**Personalization:**
• Status: {'Active' if profile and profile.total_interactions >= settings.MIN_INTERACTIONS_FOR_PERSONALIZATION else 'Learning your preferences...'}
• Avg Session: {avg_session}

Use /settings to customize your preferences!
                """.strip()

                await update.message.reply_text(profile_text, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error in profile_command: {e}")
            await update.message.reply_text("Sorry, couldn't retrieve your profile. Please try again.")
    
    async def courses_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /courses command"""
        user_data = update.effective_user

        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.telegram_id == str(user_data.id)).first()

                if not user:
                    await update.message.reply_text("Please use /start first to create your profile.")
                    return

                # Check if user has Google Classroom connected
                if not user.google_classroom_connected:
                    await update.message.reply_text(
                        "You need to connect your Google Classroom first.\n"
                        "Use /connect_classroom to get started."
                    )
                    return

                # Try to get courses directly from Google Classroom
                try:
                    from src.services.oauth_manager import UserGoogleClassroomConnector, oauth_manager
                    connector = UserGoogleClassroomConnector(user.id, oauth_manager)

                    if not connector.authenticate():
                        await update.message.reply_text(
                            "Failed to authenticate with Google Classroom. "
                            "Please try /connect_classroom again."
                        )
                        return

                    # Get courses from Google Classroom
                    google_courses = connector.get_user_courses()

                    if not google_courses:
                        courses_text = """
📚 **Your Google Classroom Courses**

No courses found in your Google Classroom. This might mean:
• You're not enrolled in any courses
• Your courses are not accessible with current permissions
• There's a temporary connection issue

Try /connect_classroom again if the issue persists.
                        """.strip()
                    else:
                        course_list = []
                        for course in google_courses:
                            course_list.append(
                                f"📋 **{course.get('name', 'Unnamed Course')}**\n"
                                f"   • ID: {course.get('id', 'N/A')}\n"
                                f"   • Section: {course.get('section', 'N/A')}\n"
                                f"   • Description: {course.get('descriptionHeading', 'N/A')}"
                            )

                        courses_text = f"""
📚 **Your Google Classroom Courses**

{chr(10).join(course_list)}

These are your live Google Classroom courses. Use /sync to download course materials for AI assistance.
                        """.strip()

                except Exception as gc_error:
                    logger.error(f"Error fetching Google Classroom courses: {gc_error}")
                    courses_text = """
📚 **Your Courses**

Failed to fetch courses from Google Classroom. This might be due to:
• Network connectivity issues
• Expired authentication
• Permission problems

Try /connect_classroom again to refresh your connection.
                    """.strip()

                keyboard = [
                    [InlineKeyboardButton("🔄 Refresh Courses", callback_data="sync_courses")],
                    [InlineKeyboardButton("🔗 Reconnect Classroom", callback_data="reconnect_classroom")],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await update.message.reply_text(courses_text, reply_markup=reply_markup, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error in courses_command: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to retrieve courses. Please try again later.")
 
    async def connect_classroom_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /connect_classroom command"""
        user_data = update.effective_user

        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter(
                    User.telegram_id == str(user_data.id)
                ).first()

                if not user:
                    await update.message.reply_text(
                        "Please use /start first to create your profile."
                    )
                    return

                if not getattr(settings, 'GOOGLE_CLASSROOM_CREDENTIALS', None):
                    await update.message.reply_text(
                        "Google Classroom integration is not configured on this server. "
                        "Please contact your administrator."
                    )
                    return

                if user.google_classroom_connected:
                    from src.services.oauth_manager import UserGoogleClassroomConnector, oauth_manager
                    connector = UserGoogleClassroomConnector(user.id, oauth_manager)

                    if connector.is_authenticated():
                        await update.message.reply_text(
                            "Your Google Classroom is already connected and working! "
                            "Use /courses to view your classes or /disconnect_classroom to disconnect."
                        )
                        return

                from src.services.oauth_manager import oauth_manager

                flow_data = oauth_manager.initiate_oauth_flow(user.id)

                keyboard = [
                    [InlineKeyboardButton("🔗 Connect Google Classroom", url=flow_data['auth_url'])]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await update.message.reply_text(
                    "To connect your Google Classroom:\n\n"
                    "1. Click the button below\n"
                    "2. Sign in with your Google account\n"
                    "3. Grant the required permissions\n"
                    "4. You'll be redirected back automatically\n\n"
                    "This link expires in 10 minutes. The connection is secure and only accesses your classroom data.",
                    reply_markup=reply_markup,
                )

        except Exception as e:
            logger.error(f"Error in connect_classroom_command: {e}")
            await update.message.reply_text(
                "Sorry, failed to initiate Google Classroom connection. Please try again later."
            )

    async def disconnect_classroom_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /disconnect_classroom command"""
        user_data = update.effective_user

        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter(
                    User.telegram_id == str(user_data.id)
                ).first()

                if not user:
                    await update.message.reply_text("Please use /start first to create your profile.")
                    return

                if not user.google_classroom_connected:
                    await update.message.reply_text("Your Google Classroom is not currently connected.")
                    return

                keyboard = [
                    [
                        InlineKeyboardButton("Yes, Disconnect", callback_data="confirm_disconnect_gc"),
                        InlineKeyboardButton("Cancel", callback_data="cancel_disconnect"),
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await update.message.reply_text(
                    "Are you sure you want to disconnect your Google Classroom?\n\n"
                    "This will:\n"
                    "• Remove access to your classroom materials\n"
                    "• Stop notifications about new content\n"
                    "• Clear your stored credentials\n\n"
                    "You can reconnect anytime using /connect_classroom",
                    reply_markup=reply_markup,
                )

        except Exception as e:
            logger.error(f"Error in disconnect_classroom_command: {e}")
            await update.message.reply_text("Sorry, an error occurred. Please try again.")

    async def connection_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /connections command to show LMS connection status"""
        user_data = update.effective_user

        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter(
                    User.telegram_id == str(user_data.id)
                ).first()

                if not user:
                    await update.message.reply_text("Please use /start first to create your profile.")
                    return

                status_text = "Your LMS Connections:\n\n"

                gc_status = "Not connected"
                if user.google_classroom_connected:
                    from src.services.oauth_manager import UserGoogleClassroomConnector, oauth_manager
                    connector = UserGoogleClassroomConnector(user.id, oauth_manager)
                    if connector.is_authenticated():
                        gc_status = "Connected and active"
                    else:
                        gc_status = "Connected but needs refresh"

                status_text += f"Google Classroom: {gc_status}\n"

                if getattr(settings, 'MOODLE_BASE_URL', None):
                    moodle_status = "Available but not user-specific"
                    status_text += f"Moodle: {moodle_status}\n"

                status_text += "\nCommands:\n"
                status_text += "• /connect_classroom - Connect Google Classroom\n"
                status_text += "• /disconnect_classroom - Disconnect Google Classroom\n"
                status_text += "• /courses - View your courses\n"
                status_text += "• /sync - Refresh your course data"

                await update.message.reply_text(status_text)

        except Exception as e:
            logger.error(f"Error in connection_status_command: {e}")
            await update.message.reply_text("Sorry, couldn't retrieve connection status.")

    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        keyboard = [
            [
                InlineKeyboardButton("📖 Learning Style", callback_data="setting_learning_style"),
                InlineKeyboardButton("📊 Difficulty", callback_data="setting_difficulty"),
            ],
            [
                InlineKeyboardButton("🔔 Notifications", callback_data="setting_notifications"),
                InlineKeyboardButton("📱 Response Length", callback_data="setting_response_length"),
            ],
            [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        settings_text = """
⚙️ **Settings & Preferences**

Customize your Study Helper Agent experience:

• **Learning Style**: How you prefer to learn (visual, auditory, etc.)
• **Difficulty Level**: Complexity of explanations you prefer
• **Notifications**: When to receive updates about new content
• **Response Length**: How detailed you want my responses

Choose a setting to modify:
        """.strip()

        await update.message.reply_text(settings_text, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def handle_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle general text queries - main AI interaction - with enhanced RAG"""
        user_data = update.effective_user
        query_text = update.message.text
        
        try:
            # Show typing indicator
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            with db_manager.get_session() as session:
                user = self._get_or_create_user(session, user_data)
                
                # Use the enhanced RAG processing
                response_text = await self._process_query_rag_enhanced(query_text, user)
                
                # Send response (split if too long for Telegram)
                # Temporarily disable Markdown to fix parsing errors
                if len(response_text) > 4096:  # Telegram message limit
                    # Split at natural break points
                    parts = self._split_long_message(response_text)
                    for i, part in enumerate(parts):
                        await update.message.reply_text(part)  # No parse_mode
                        if i < len(parts) - 1:  # Small delay between parts
                            await asyncio.sleep(0.5)
                else:
                    await update.message.reply_text(response_text)  # No parse_mode
                
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
                
                # Log interaction with enhanced metadata
                self._log_rag_interaction(session, user.id, query_text, response_text)
                
        except Exception as e:
            logger.error(f"Error in handle_query: {e}")
            await update.message.reply_text(
                "Sorry, I encountered an error processing your question. Please try again."
            )
    def _split_long_message(self, text: str, max_length: int = 4000) -> list:
        """Split long messages at natural break points"""
        if len(text) <= max_length:
            return [text]
        
        parts = []
        current_part = ""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_part + paragraph) > max_length:
                if current_part:
                    parts.append(current_part.strip())
                    current_part = paragraph
                else:
                    # Single paragraph is too long, split by sentences
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if len(current_part + sentence + '. ') > max_length:
                            if current_part:
                                parts.append(current_part.strip())
                                current_part = sentence + '. '
                            else:
                                # Single sentence too long, force split
                                parts.append(sentence[:max_length])
                                current_part = sentence[max_length:]
                        else:
                            current_part += sentence + '. '
            else:
                current_part += paragraph + '\n\n'
        
        if current_part:
            parts.append(current_part.strip())
        
        return parts
    
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

            elif callback_data == "process_docs":
                await query.edit_message_text("🔄 Processing documents...")
                try:
                    result = scheduler_service.force_process_documents()
                    if result.get('success'):
                        processed = result.get('documents_processed', 0)
                        final_stats = result.get('final_stats', {})
                        success_rate = final_stats.get('success_rate', 'N/A')
                        message = f"✅ Document processing completed!\nProcessed: {processed} documents\nSuccess rate: {success_rate}"
                    else:
                        message = f"❌ Processing failed: {result.get('error', 'Unknown error')}"
                    await query.edit_message_text(message)
                except Exception as process_error:
                    logger.error(f"Process docs error: {process_error}")
                    await query.edit_message_text("❌ Document processing failed due to internal error.")
                    
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

            # Handle material assistance callbacks
            elif callback_data.startswith(("help_assignment_", "breakdown_assignment_", "explain_assignment_",
                                          "study_quiz_", "practice_quiz_", "concepts_quiz_",
                                          "summarize_reading_", "questions_reading_", "keypoints_reading_")):
                await self._handle_material_assistance(update, context, callback_data)

            else:
                await query.edit_message_text("Feature coming soon! Though sifai kujibu hivi mkuu. \n\n Kwa handle_callback")
                
        except Exception as e:
            logger.error(f"Error in handle_callback: {e}")
            await query.edit_message_text("Sorry, something went wrong. Please try again. \n\n Kwa handle_callback superior")

    async def _handle_material_assistance(self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str):
        """Handle material assistance button clicks"""
        query = update.callback_query
        user_data = query.from_user

        try:
            # Parse callback data to extract action and document ID
            parts = callback_data.split("_")
            if len(parts) < 3:
                await query.edit_message_text("❌ Invalid request format")
                return

            action = "_".join(parts[:-1])  # e.g., "help_assignment", "summarize_reading"
            document_id = parts[-1]

            # Get document details
            with db_manager.get_session() as session:
                from src.data.models import Document
                document = session.query(Document).filter(Document.id == document_id).first()

                if not document:
                    await query.edit_message_text("❌ Document not found")
                    return

                # Get user
                user = session.query(User).filter(User.telegram_id == str(user_data.id)).first()
                if not user:
                    await query.edit_message_text("❌ Please use /start first to create your profile.")
                    return

            # Update the message to show we're working on it
            await query.edit_message_text(f"🤖 **Working on your request...**\n\n📄 *{document.title}*\n\n⏳ Please wait while I analyze the material...")

            # Generate appropriate response based on the action
            response = await self._generate_material_response(action, document, user)

            # Send the response (split if too long)
            response_parts = self._split_long_message(response)

            # Edit the first message
            await query.edit_message_text(response_parts[0])

            # Send additional parts as new messages if needed
            for part in response_parts[1:]:
                await context.bot.send_message(chat_id=query.message.chat_id, text=part)

            logger.info(f"📤 Handled {action} request for document {document.title} by user {user.telegram_id}")

        except Exception as e:
            logger.error(f"❌ Error handling material assistance: {e}")
            await query.edit_message_text("❌ Sorry, something went wrong while processing your request. Please try again.")

    async def _generate_material_response(self, action: str, document: Document, user: User) -> str:
        """Generate appropriate response based on the requested action"""
        try:
            # Create context-aware queries based on the action
            if action == "help_assignment":
                query = f"Help me understand and complete the assignment '{document.title}'. What are the main requirements and how should I approach it?"
            elif action == "breakdown_assignment":
                query = f"Break down the assignment '{document.title}' into smaller, manageable tasks. What steps should I follow?"
            elif action == "explain_assignment":
                query = f"Explain the requirements and expectations for the assignment '{document.title}'. What exactly am I supposed to do?"
            elif action == "study_quiz":
                query = f"Help me prepare for the quiz/test '{document.title}'. What topics should I focus on studying?"
            elif action == "practice_quiz":
                query = f"Create practice questions based on the material in '{document.title}' to help me prepare."
            elif action == "concepts_quiz":
                query = f"What are the key concepts and topics I should understand for '{document.title}'?"
            elif action == "summarize_reading":
                query = f"Provide a comprehensive summary of the key points in '{document.title}'"
            elif action == "questions_reading":
                query = f"I have questions about the content in '{document.title}'. Can you help me understand it better?"
            elif action == "keypoints_reading":
                query = f"What are the main key points and important concepts in '{document.title}'?"
            else:
                query = f"Help me understand the content in '{document.title}'"

            # Use the RAG pipeline to get a contextual response
            if self.rag_pipeline:
                response = await self._process_query_rag_enhanced(query, user)

                # Add a helpful header
                if action.startswith("help_assignment"):
                    header = "📝 **Assignment Help**\n\n"
                elif action.startswith("study_quiz") or action.startswith("practice_quiz") or action.startswith("concepts_quiz"):
                    header = "🧠 **Study Assistance**\n\n"
                else:
                    header = "📖 **Material Summary**\n\n"

                return header + response
            else:
                return "❌ Sorry, I'm currently unable to process your request. The content analysis system is not available."

        except Exception as e:
            logger.error(f"❌ Error generating material response: {e}")
            return "❌ Sorry, I encountered an error while analyzing the material. Please try asking your question directly in a message."



    async def _process_query_rag_enhanced(self, query: str, user: User) -> str:
        """Enhanced query processing with RAG pipeline and LLM integration"""
        if not self.rag_pipeline:
            logger.error("RAG pipeline not initialized")
            return await self._process_query_basic(query, user)  # Fallback to basic
        
        try:
            # Determine course context from query
            course_id = self._extract_course_context(query, user)
            
            # Get user preferences for personalization
            user_preferences = {
                'learning_style': user.learning_style,
                'difficulty_preference': user.difficulty_preference,
                'response_length': getattr(user, 'preferred_response_length', 'medium')
            }
            
            # Generate RAG response using the enhanced pipeline
            rag_result = self.rag_pipeline.generate_rag_response(
                query, 
                course_id, 
                user_preferences
            )
            
            # Format the response with sources and confidence indicator
            response = self._format_rag_response(rag_result, user)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG-enhanced processing: {e}")
            return await self._process_query_basic(query, user)
        
    def _format_rag_response(self, rag_result: Dict[str, Any], user: User) -> str:
        """Format RAG response with sources and confidence indicators"""
        response_parts = []
        
        # Add the main response
        main_response = rag_result['response']
        if main_response:
            response_parts.append(main_response)
        
        # Add sources if context was used
        if rag_result.get('context_used', False) and rag_result.get('sources'):
            response_parts.append("\n**📚 Sources from your course materials:**")
            
            for i, source in enumerate(rag_result['sources'][:3], 1):  # Show max 3 sources
                confidence_emoji = self._get_confidence_emoji(source['similarity_score'])
                response_parts.append(
                    f"{i}. {source['title']} ({source['course_code']}) {confidence_emoji}"
                )
            
            if len(rag_result['sources']) > 3:
                response_parts.append(f"*...and {len(rag_result['sources']) - 3} more sources*")
        
        # Add confidence and personalization note
        confidence = rag_result.get('confidence', 'medium')
        if confidence == 'high':
            response_parts.append(f"\n*High confidence response tailored for {user.learning_style} learning style.*")
        elif confidence == 'medium':
            response_parts.append(f"\n*Response based on course materials (medium confidence).*")
        else:
            response_parts.append(f"\n*Limited course material found. Consider using /sync to update materials.*")
        
        return "\n".join(response_parts)

    def _get_confidence_emoji(self, similarity_score: float) -> str:
        """Get emoji indicating confidence level"""
        if similarity_score >= 0.8:
            return "🎯"  # High confidence
        elif similarity_score >= 0.6:
            return "✅"  # Medium confidence
        else:
            return "📝"  # Low confidence


    def _extract_course_context(self, query: str, user: User) -> Optional[int]:
        """Extract course context from query"""
        query_lower = query.lower()
        
        # Look for explicit course mentions
        course_keywords = {
            'ics201': {'data structures', 'algorithms', 'programming'},
            'ics301': {'software engineering', 'design patterns', 'uml'},
            'mat201': {'discrete math', 'logic', 'proofs'},
            'ics401': {'machine learning', 'ai', 'neural networks'}
        }
        
        # Try to match based on course codes first
        for course_code, topics in course_keywords.items():
            if course_code in query_lower:
                # Look up course ID from database
                with db_manager.get_session() as session:
                    course = session.query(Course).filter(Course.course_code.ilike(f"%{course_code}%")).first()
                    if course:
                        return course.id
            
            # Try to match based on topic keywords
            if any(topic in query_lower for topic in topics):
                with db_manager.get_session() as session:
                    course = session.query(Course).filter(Course.course_code.ilike(f"%{course_code}%")).first()
                    if course:
                        return course.id
        
        # If no explicit course found, try to get user's most active course
        try:
            with db_manager.get_session() as session:
                # Get user's most recent course enrollment
                recent_enrollment = session.query(CourseEnrollment).filter(
                    CourseEnrollment.user_id == user.id,
                    CourseEnrollment.is_active == True
                ).order_by(CourseEnrollment.enrollment_date.desc()).first()
                
                if recent_enrollment:
                    return recent_enrollment.course_id
        except Exception as e:
            logger.warning(f"Could not determine user's active course: {e}")
        
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
• Try asking about topics from your enrolled courses (ICS 201, ICS 301, MAT 201) HARDCODED!!! :(
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
    
    
    def _log_rag_interaction(self, session, user_id: int, query: str, response: str):
        """Log RAG interaction with enhanced metadata"""
        try:
            interaction = UserInteraction(
                user_id=user_id,
                query_text=query,
                response_text=response,
                interaction_type="rag_enhanced_query",
                response_time_ms=0  # Could track actual response time
            )
            session.add(interaction)
            
            # Update personalization profile
            profile = session.query(PersonalizationProfile).filter(
                PersonalizationProfile.user_id == user_id
            ).first()
            
            if profile:
                profile.total_interactions += 1
                profile.last_interaction = datetime.now()
                
                # Track question complexity (simple heuristic)
                complexity_indicators = len(query.split()) / 10  # Normalize by word count
                if profile.question_complexity_level:
                    profile.question_complexity_level = (profile.question_complexity_level * 0.9) + (complexity_indicators * 0.1)
                else:
                    profile.question_complexity_level = complexity_indicators
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Error logging RAG interaction: {e}")

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
        """Enhanced error handler with detailed logging and user feedback"""
        error = context.error

        # Log detailed error information
        logger.error(f"Bot error occurred:", exc_info=error)

        if update:
            logger.error(f"Update that caused error: {update}")

        # Determine error type and provide appropriate user response
        error_message = "Sorry, I encountered an unexpected error."

        if "timeout" in str(error).lower():
            error_message = "⏱️ Request timed out. Please try again in a moment."
        elif "network" in str(error).lower() or "connection" in str(error).lower():
            error_message = "🌐 Network connection issue. Please check your internet and try again."
        elif "authentication" in str(error).lower() or "unauthorized" in str(error).lower():
            error_message = "🔐 Authentication issue. Please try /connect_classroom to refresh your connection."
        elif "database" in str(error).lower():
            error_message = "💾 Database temporarily unavailable. Please try again later."
        else:
            error_message = "❌ An unexpected error occurred. Please try again or use /help for assistance."

        # Try to send error message to user
        try:
            if update and update.effective_chat:
                if update.message:
                    await update.message.reply_text(error_message)
                elif update.callback_query:
                    await update.callback_query.edit_message_text(error_message)
        except Exception as send_error:
            logger.error(f"Failed to send error message to user: {send_error}")
    
    async def run(self):
        """Start the bot and keep it running"""
        logger.info("Starting Study Helper Agent bot...")

        # Set event loop for scheduler
        from src.services.scheduler import scheduler_service
        import asyncio
        scheduler_service.set_event_loop(asyncio.get_running_loop())

        try:
            # Initialize and start the application
            await self.application.initialize()
            await self.application.start()

            # Start polling for updates
            await self.application.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=False
            )

            logger.info("Bot is running. Press Ctrl+C to stop.")

            # Keep running until interrupted
            stop_event = asyncio.Event()

            # Wait forever (until Ctrl+C)
            await stop_event.wait()

        except asyncio.CancelledError:
            logger.info("Bot run cancelled")
        finally:
            # Cleanup
            logger.info("Stopping bot...")
            if self.application.updater.running:
                await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Bot stopped successfully")