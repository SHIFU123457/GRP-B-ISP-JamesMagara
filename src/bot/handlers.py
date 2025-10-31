import asyncio
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

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
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import and_, desc

from src.data.models import User, UserInteraction, PersonalizationProfile, QuizSession, ConversationSession
from config.database import db_manager
from config.settings import settings
#For RAG functionality
from src.core.rag_pipeline import RAGPipeline
#For LMS integration functionality
from src.services.scheduler import scheduler_service, escape_markdown
from src.services.lms_integration import lms_service
from src.data.models import User, UserInteraction, PersonalizationProfile, Course, Document, CourseEnrollment
#For personalization and session management
from src.services.personalization_engine import personalization_engine, session_manager

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
        """Create type-specific interactive keyboard for notifications"""
        material_type = notification.get('material_type', 'material')
        document_title = notification.get('document_title', '')
        course_name = notification.get('course_name', '')
        document_id = notification.get('document_id', '')

        keyboard = []

        if material_type == 'assignment':
            keyboard = [
                [
                    InlineKeyboardButton("✍️ Help Complete Assignment",
                                       callback_data=f"complete_assignment_{document_id}"),
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
                    InlineKeyboardButton("✍️ Help Answer Quiz",
                                       callback_data=f"complete_quiz_{document_id}"),
                    InlineKeyboardButton("🧠 Study for Quiz",
                                       callback_data=f"study_quiz_{document_id}")
                ],
                [
                    InlineKeyboardButton("❓ Practice Questions",
                                       callback_data=f"practice_quiz_{document_id}")
                ]
            ]
        elif material_type == 'question':
            keyboard = [
                [
                    InlineKeyboardButton("✍️ Help Answer Question",
                                       callback_data=f"complete_question_{document_id}"),
                    InlineKeyboardButton("💡 Hint",
                                       callback_data=f"hint_question_{document_id}")
                ]
            ]
        elif material_type == 'announcement':
            keyboard = [
                [
                    InlineKeyboardButton("📖 Summarize",
                                       callback_data=f"summarize_announcement_{document_id}"),
                    InlineKeyboardButton("❓ Ask About This",
                                       callback_data=f"ask_announcement_{document_id}")
                ]
            ]
        else:  # material (reading/document)
            keyboard = [
                [
                    InlineKeyboardButton("📖 Summarize Document",
                                       callback_data=f"summarize_material_{document_id}"),
                    InlineKeyboardButton("❓ Ask Questions",
                                       callback_data=f"questions_material_{document_id}")
                ],
                [
                    InlineKeyboardButton("🔍 Key Points",
                                       callback_data=f"keypoints_material_{document_id}")
                ]
            ]

        return keyboard
        
    
    def _add_handlers(self):
        """Add all command and message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("new_session", self.new_session_command))
        self.application.add_handler(CommandHandler("my_sessions", self.my_sessions_command))
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

                # Get or create conversation session
                conv_session = session_manager.get_or_create_session(user.id, session)

                # Update session activity for command
                session_manager.update_session_activity(
                    conv_session,
                    session,
                    interaction_type="command"
                )

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
3. Use /new_session to start discussing a new topic
4. Use /help for more commands

Example: "What are the main topics in today's lecture notes?"

💡 **Tip:** I remember our conversation context! Use `/new_session` when switching to a completely different topic.
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

    async def new_session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /new_session command to start a fresh conversation session"""
        user_data = update.effective_user

        try:
            with db_manager.get_session() as session:
                user = self._get_or_create_user(session, user_data)
                user_id = user.id

                # Get current active session
                current_session = session.query(ConversationSession).filter(
                    and_(
                        ConversationSession.user_id == user_id,
                        ConversationSession.is_active == True
                    )
                ).order_by(desc(ConversationSession.last_activity_at)).first()

                # Store info about the old session
                old_session_info = None
                if current_session:
                    old_session_info = {
                        'primary_topic': current_session.primary_topic,
                        'message_count': current_session.message_count,
                        'duration': (datetime.now(timezone.utc) - current_session.started_at.replace(tzinfo=None)).total_seconds() / 60
                    }

                    # Force close the current session
                    session_manager.force_close_session(current_session.session_id)
                    logger.info(f"User {user_id} manually closed session {current_session.session_id}")

                # Create a new session
                new_session = session_manager.get_or_create_session(user_id, session)

                # Prepare response message
                if old_session_info:
                    response_text = f"""🔄 **New Session Started**

📝 Previous session summary:
• Topic: {old_session_info['primary_topic'] or 'General discussion'}
• Messages: {old_session_info['message_count']}
• Duration: {old_session_info['duration']:.1f} minutes

✨ You can now start discussing a new topic with a fresh context. The bot won't reference the previous conversation.

💡 **Tip:** Sessions automatically reset after 30 minutes of inactivity, or you can use `/new_session` anytime to start fresh!
                    """.strip()
                else:
                    response_text = f"""🔄 **New Session Started**

✨ You can now start discussing a new topic with a fresh context.

💡 **Tip:** Sessions automatically reset after 30 minutes of inactivity, or you can use `/new_session` anytime to start fresh!
                    """.strip()

                await update.message.reply_text(response_text, parse_mode='Markdown')

                # Log the interaction
                self._log_interaction(session, user_id, "/new_session", response_text, "command")

        except Exception as e:
            logger.error(f"Error in new_session_command: {e}", exc_info=True)
            await update.message.reply_text(
                "Sorry, I encountered an error while creating a new session. Please try again later."
            )

    async def my_sessions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /my_sessions command to view and switch between previous sessions"""
        user_data = update.effective_user

        try:
            with db_manager.get_session() as session:
                user = self._get_or_create_user(session, user_data)
                user_id = user.id

                # Get all sessions for this user (both active and inactive)
                all_sessions = session.query(ConversationSession).filter(
                    ConversationSession.user_id == user_id
                ).order_by(desc(ConversationSession.last_activity_at)).limit(10).all()

                if not all_sessions:
                    await update.message.reply_text(
                        "📭 You don't have any previous sessions yet.\n\n"
                        "Start chatting with me, and your conversations will be organized into sessions!",
                        parse_mode='Markdown'
                    )
                    return

                # Get current active session
                current_session = session.query(ConversationSession).filter(
                    and_(
                        ConversationSession.user_id == user_id,
                        ConversationSession.is_active == True
                    )
                ).first()

                current_session_id = current_session.session_id if current_session else None

                # Build session list message
                session_list = []
                keyboard_buttons = []

                for idx, sess in enumerate(all_sessions, 1):
                    # Calculate session info
                    is_current = sess.session_id == current_session_id
                    status_emoji = "🟢" if is_current else "⚪"

                    # Format timestamp
                    time_str = sess.last_activity_at.strftime('%b %d, %H:%M') if sess.last_activity_at else 'Unknown'

                    # Calculate duration
                    if sess.ended_at and sess.started_at:
                        duration = (sess.ended_at - sess.started_at).total_seconds() / 60
                    elif is_current and sess.started_at:
                        duration = (datetime.utcnow() - sess.started_at.replace(tzinfo=None)).total_seconds() / 60
                    else:
                        duration = sess.session_duration_minutes or 0

                    # Build session description
                    topic = sess.primary_topic or "General discussion"
                    session_desc = (
                        f"{status_emoji} **Session {idx}** {'(Current)' if is_current else ''}\n"
                        f"   • Topic: {topic}\n"
                        f"   • Messages: {sess.message_count}\n"
                        f"   • Duration: {duration:.0f} min\n"
                        f"   • Last active: {time_str}"
                    )
                    session_list.append(session_desc)

                    # Add button for non-current sessions
                    if not is_current:
                        button_text = f"📂 Session {idx}: {topic[:20]}..."
                        keyboard_buttons.append([
                            InlineKeyboardButton(
                                button_text,
                                callback_data=f"switch_session:{sess.session_id}"
                            )
                        ])

                # Build response message
                response_text = "📚 **Your Conversation Sessions**\n\n" + "\n\n".join(session_list)

                if keyboard_buttons:
                    response_text += "\n\n💡 **Tap a session below to switch to it:**"
                    keyboard = InlineKeyboardMarkup(keyboard_buttons)
                    await update.message.reply_text(response_text, parse_mode='Markdown', reply_markup=keyboard)
                else:
                    response_text += "\n\n✨ All sessions shown are either current or completed."
                    await update.message.reply_text(response_text, parse_mode='Markdown')

                # Log the interaction
                self._log_interaction(session, user_id, "/my_sessions", "Viewed session list", "command")

        except Exception as e:
            logger.error(f"Error in my_sessions_command: {e}", exc_info=True)
            await update.message.reply_text(
                "Sorry, I encountered an error while retrieving your sessions. Please try again later."
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
            # Get user-specific status
            user_telegram_id = str(update.message.from_user.id)
            status = scheduler_service.get_sync_status(user_telegram_id=user_telegram_id)

            status_text = f"""
📊 **Your Study Status**

**Scheduler:** {'🟢 Running' if status['running'] else '🔴 Stopped'}
**Connected LMS:** {', '.join(status['connected_platforms']) if status['connected_platforms'] else 'None'}
**RAG Pipeline:** {'🟢 Available' if status['rag_available'] else '🔴 Not Available'}

**Your Documents:**
• Total: {status['documents']['total']}
• Processed: {status['documents']['processed']} ✅
• Pending: {status['documents']['pending']} ⏳
• Processing: {status['documents']['processing']} 🔄
• Failed: {status['documents']['failed']} ❌

**Your Courses:**
• Enrolled: {status['courses']['total']}
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
/new\\_session - Start a fresh conversation (clear context)
/my\\_sessions - View and switch between previous sessions
/help - Show this help message
/profile - View your learning profile
/courses - View enrolled courses
/settings - Adjust your preferences

**LMS Integration:**
/connect\\_classroom - Connect your Google Classroom
/disconnect\\_classroom - Disconnect Google Classroom
/connections - View connection status
/sync - Manually sync with your LMS
/status - View system status
/process\\_docs - Force document processing

**How to Use:**
📝 **Ask Questions**: Just type your question naturally
   Example: "Explain the concept of inheritance in OOP"

🔍 **Search Content**: Ask about specific topics
   Example: "What did the professor say about databases?"

📊 **Get Summaries**: Request summaries of materials
   Example: "Summarize today's lecture on algorithms"

🔄 **Switch Topics**: Use /new\\_session when changing subjects
   Example: After discussing Python, use /new\\_session before asking about Math

📂 **Resume Sessions**: Use /my\\_sessions to return to previous conversations
   Example: Switch back to a Python discussion from yesterday

**Tips for Better Results:**
• Be specific about which course or topic
• Ask follow-up questions for clarification
• Use /new\\_session when switching to a different topic
• Use /my\\_sessions to resume or review past conversations
• Rate my responses to improve personalization
• Use /sync to get the latest materials
• Use /settings to customize your experience

**Personalization Settings:**
⚙️ Use /settings to manually adjust:
• **Learning Style**: Example-driven, Socratic, Theory-first, etc.
• **Response Length**: Short, Medium, or Long responses
• **Complexity Level**: Beginner, Intermediate, or Advanced
• **Learning Pace**: Casual, Moderate, or Intensive
• **Difficulty**: Easy, Medium, or Hard explanations

💡 The system auto-adjusts these every 12 hours based on your interaction patterns, but you can override them anytime!

**About Context Memory:**
💡 I remember our conversation within a session (30 minutes or until you use /new\\_session). This helps me understand follow-up questions like "explain it with code" or "show me another example." You can always resume past sessions using /my\\_sessions!

**Getting Started:**
1. Use /connect\\_classroom to link your Google account
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
                            # Escape special HTML characters
                            import html
                            course_name = html.escape(course.get('name', 'Unnamed Course'))
                            course_id = html.escape(str(course.get('id', 'N/A')))
                            course_section = html.escape(str(course.get('section', 'N/A')))
                            course_desc = html.escape(str(course.get('descriptionHeading', 'N/A')))

                            course_list.append(
                                f"📋 <b>{course_name}</b>\n"
                                f"   • ID: {course_id}\n"
                                f"   • Section: {course_section}\n"
                                f"   • Description: {course_desc}"
                            )

                        courses_text = f"""
📚 <b>Your Google Classroom Courses</b>

{chr(10).join(course_list)}

These are your live Google Classroom courses. Use /sync to download course materials for AI assistance.
                        """.strip()

                except Exception as gc_error:
                    logger.error(f"Error fetching Google Classroom courses: {gc_error}")
                    courses_text = """
📚 <b>Your Courses</b>

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

                # Log the message for debugging
                logger.debug(f"Courses text to send (length={len(courses_text)}): {courses_text[:300]}")

                # Send with HTML parse mode (more forgiving than Markdown)
                await update.message.reply_text(courses_text, reply_markup=reply_markup, parse_mode='HTML')

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
                InlineKeyboardButton("📱 Response Length", callback_data="setting_response_length"),
                InlineKeyboardButton("🎯 Complexity Level", callback_data="setting_complexity"),
            ],
            [
                InlineKeyboardButton("⚡ Learning Pace", callback_data="setting_pace"),
                InlineKeyboardButton("🔔 Notifications", callback_data="setting_notifications"),
            ],
            [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="main_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        settings_text = """
⚙️ **Settings & Preferences**

Customize your Study Helper Agent experience:

• **Learning Style**: How you prefer to learn (example-driven, socratic, etc.)
• **Difficulty Level**: Complexity of explanations you prefer
• **Response Length**: How detailed you want my responses
• **Complexity Level**: Your question complexity preference
• **Learning Pace**: Your learning intensity level
• **Notifications**: When to receive updates about new content

Choose a setting to modify:
        """.strip()

        await update.message.reply_text(settings_text, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def handle_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle general text queries - main AI interaction - with enhanced RAG and session tracking"""
        user_data = update.effective_user
        query_text = update.message.text
        start_time = datetime.now()

        try:
            # Show typing indicator
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

            # Cache user_id and active_quiz_id within session scope
            with db_manager.get_session() as session:
                user = self._get_or_create_user(session, user_data)
                user_id = user.id

                # Get or create conversation session
                conv_session = session_manager.get_or_create_session(user_id, session)

                # Update session activity
                session_manager.update_session_activity(
                    conv_session,
                    session,
                    interaction_type="question"
                )

                # Check if user has an active quiz session
                active_quiz = session.query(QuizSession).filter(
                    QuizSession.user_id == user_id,
                    QuizSession.is_active == True
                ).first()

                # Cache quiz ID and session ID before session closes
                active_quiz_id = active_quiz.id if active_quiz else None
                current_session_id = conv_session.session_id

            # Detect quiz-related commands
            query_lower = query_text.lower()
            quiz_keywords = [
                'quiz me', 'ask me questions', 'test me', 'generate questions',
                'quiz on', 'questions on', 'questions about', 'test my knowledge',
                'question me on', 'question me about', 'question me regarding',
                'provide a quiz', 'provide quiz', 'format a quiz', 'formate a quiz',
                'create a quiz', 'create quiz', 'give me a quiz', 'make a quiz',
                'another quiz', 'new quiz', 'start a quiz', 'begin a quiz'
            ]

            is_quiz_request = any(keyword in query_lower for keyword in quiz_keywords)

            # If active quiz exists and user types something unrelated
            if active_quiz_id and not is_quiz_request:
                # Pause the quiz and answer the question
                with db_manager.get_session() as session:
                    quiz = session.query(QuizSession).filter(QuizSession.id == active_quiz_id).first()
                    quiz.is_paused = True
                    session.commit()

                # Answer the question
                with db_manager.get_session() as session:
                    user = session.query(User).filter(User.id == user_id).first()
                    response_text = await self._process_query_rag_enhanced(query_text, user)

                await update.message.reply_text(response_text)

                # Ask if they want to continue the quiz
                keyboard = [
                    [
                        InlineKeyboardButton("✅ Resume Quiz", callback_data=f"quiz_resume_{active_quiz_id}"),
                        InlineKeyboardButton("❌ End Quiz", callback_data=f"quiz_end_{active_quiz_id}")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await update.message.reply_text(
                    "Would you like to continue with your quiz?",
                    reply_markup=reply_markup
                )
                return

            # Handle quiz initiation requests
            if is_quiz_request:
                # Extract topic from query
                topic = self._extract_quiz_topic(query_text)

                # Check if topic refers to current conversation
                if topic:
                    conversation_references = [
                        'what we discussed', 'what we have discussed', 'what we talked about',
                        'what we have been discussing', 'our conversation', 'this topic',
                        'what we covered', 'what you explained', 'what we ve been talking about',
                        'the current topic', 'this session', 'what i asked about', 'thus far'
                    ]

                    # Check if the topic is a reference to the conversation
                    topic_lower = topic.lower()
                    is_conversation_reference = any(ref in topic_lower for ref in conversation_references)

                    if is_conversation_reference:
                        # Get conversation history to determine actual topic
                        with db_manager.get_session() as session:
                            conv_session = session_manager.get_or_create_session(user_id, session)
                            conversation_history = session_manager.get_conversation_history(user_id, session, limit=5)

                            # Get topics from session context
                            session_context = conv_session.session_context or {}
                            primary_topic = conv_session.primary_topic
                            all_topics = session_context.get('all_topics', [])

                            if primary_topic:
                                # Use the primary topic from session
                                actual_topic = primary_topic
                                logger.info(f"Quiz request refers to conversation - using session topic: {actual_topic}")
                            elif all_topics:
                                # Use the first topic found
                                actual_topic = all_topics[0]
                                logger.info(f"Quiz request refers to conversation - using first topic: {actual_topic}")
                            elif conversation_history:
                                # Fallback: Use the most recent query as context
                                actual_topic = conversation_history[-1]['query'][:100]  # Last user query
                                logger.info(f"Quiz request refers to conversation - using recent query: {actual_topic}")
                            else:
                                await update.message.reply_text(
                                    "❌ I don't have enough conversation context to generate a quiz.\n"
                                    "Please specify a topic explicitly, e.g., 'Quiz me on machine learning'"
                                )
                                return

                        # Pass conversation history to quiz generation for context
                        await self._initiate_quiz_from_topic(
                            update.effective_chat.id,
                            user_id,
                            actual_topic,
                            conversation_history=conversation_history
                        )
                    else:
                        # Normal topic-based quiz
                        await self._initiate_quiz_from_topic(update.effective_chat.id, user_id, topic)
                else:
                    await update.message.reply_text(
                        "Please specify a topic you'd like to be quizzed on.\n"
                        "For example: 'Quiz me on data structures' or 'Ask me questions about Python'"
                    )
                return

            # Regular query processing
            # === CENTRALIZED TOPIC EXTRACTION (Extract once, use everywhere) ===
            from src.services.adaptive_response_engine import adaptive_response_engine

            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()

                # Get session context FIRST (needed for context-aware topic extraction)
                conv_session = session_manager.get_or_create_session(user_id, session)

                # CONTEXT-AWARE TOPIC EXTRACTION
                # Searches conversation history for "previous", "first", "that" references
                extracted_topics = adaptive_response_engine.topic_analyzer.extract_topics_with_context(
                    query=query_text,
                    user_id=user_id,
                    session_id=current_session_id,
                    db_session=session,
                    conv_session=conv_session
                )

                current_topic = extracted_topics[0] if extracted_topics else None
                # Get previous topic from session context (for confusion tracking)
                previous_topic = conv_session.session_context.get('current_topic') if conv_session.session_context else None

                # === SENTIMENT & CONFUSION DETECTION ===
                # Analyze query for confusion/sentiment
                sentiment_detector = adaptive_response_engine.sentiment_detector
                sentiment_analysis = sentiment_detector.analyze_query(query_text)

                # Log confusion event to database
                confusion_event = sentiment_detector.log_confusion_event(
                    user_id=user_id,
                    query=query_text,
                    analysis=sentiment_analysis,
                    session=session,
                    session_id=current_session_id,
                    topic=current_topic,
                    previous_topic=previous_topic
                )

                # Update struggle topics if confused
                if sentiment_analysis.get('is_confused') and current_topic:
                    adaptive_response_engine.topic_analyzer.update_struggle_topic(
                        user_id=user_id,
                        topic=current_topic,
                        session=session,
                        is_confused=True
                    )

                # Use the enhanced RAG processing (pass pre-extracted topics)
                response_text = await self._process_query_rag_enhanced(
                    query_text, user, pre_extracted_topics=extracted_topics
                )

                # Send response (split if too long for Telegram)
                # Temporarily disable Markdown to fix parsing errors
                if len(response_text) > 8192:  # Telegram message limit
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

            # Calculate response time
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Update session context with current topic and course (reuse extracted topics)
            with db_manager.get_session() as session:
                conv_session = session_manager.get_or_create_session(user_id, session)
                user_obj = session.query(User).filter(User.id == user_id).first()

                # Extract course context
                course_id = self._extract_course_context(query_text, user_obj)

                # Update session context with current topic (use pre-extracted topics)
                context_updates = {}
                if extracted_topics:
                    context_updates['current_topic'] = extracted_topics[0]  # Primary topic
                    context_updates['all_topics'] = extracted_topics
                    # Track primary topic for session
                    if not conv_session.primary_topic:
                        conv_session.primary_topic = extracted_topics[0]

                # Update courses_discussed list
                if course_id:
                    course = session.query(Course).filter(Course.id == course_id).first()
                    if course:
                        context_updates['current_course'] = course.course_name
                        # Add to courses_discussed if not already there
                        courses_discussed = conv_session.courses_discussed or []
                        if course_id not in courses_discussed:
                            courses_discussed.append(course_id)
                            conv_session.courses_discussed = courses_discussed
                            flag_modified(conv_session, 'courses_discussed')

                session.commit()

                # Update session activity with context
                session_manager.update_session_activity(
                    conv_session,
                    session,
                    interaction_type="question",
                    context_updates=context_updates
                )

            # Log interaction with personalization engine
            personalization_engine.record_interaction(
                user_id=user_id,
                query=query_text,
                response=response_text,
                interaction_type="question",
                response_time_ms=response_time_ms
            )

            logger.info(f"Session {current_session_id}: User {user_id} asked question (response time: {response_time_ms}ms)")

        except Exception as e:
            logger.error(f"Error in handle_query: {e}")
            await update.message.reply_text(
                "Sorry, I encountered an error processing your question. Please try again."
            )
    def _split_long_message(self, text: str, max_length: int = 8000) -> list:
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
        callback_data = query.data

        # Try to acknowledge the callback, but don't let it block functionality
        try:
            await asyncio.wait_for(query.answer(), timeout=3.0)
        except (asyncio.TimeoutError, Exception) as e:
            # Log but continue - acknowledgment is just UX, not critical
            logger.warning(f"Failed to acknowledge callback (non-critical): {e}")

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

            # Session switching callback
            elif callback_data.startswith("switch_session:"):
                await self._handle_session_switch(update, callback_data)

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
            elif callback_data.startswith(("complete_assignment_", "breakdown_assignment_", "explain_assignment_",
                                          "complete_quiz_", "study_quiz_", "practice_quiz_",
                                          "complete_question_", "hint_question_",
                                          "summarize_announcement_", "ask_announcement_",
                                          "summarize_material_", "questions_material_", "keypoints_material_")):
                await self._handle_material_assistance(update, context, callback_data)

            # Handle quiz-related callbacks
            elif callback_data.startswith("quiz_answer_"):
                await self._handle_quiz_answer(update, context, callback_data)

            elif callback_data.startswith("quiz_continue_"):
                await self._handle_quiz_continue(update, context, callback_data)

            elif callback_data.startswith("quiz_resume_"):
                await self._handle_quiz_resume(update, context, callback_data)

            elif callback_data.startswith("quiz_end_"):
                await self._handle_quiz_end(update, context, callback_data)

            # Handle learning style preference changes
            elif callback_data.startswith("set_"):
                await self._handle_preference_change(update, context, callback_data)

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

            # Handle quiz initiation separately
            if action == "questions_material":
                await self._initiate_quiz_from_document(update, context, document_id)
                return

            # Get document details (cache values before session closes)
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

                # Cache values before session closes
                doc_title = document.title
                doc_id = document.id
                user_telegram_id = user.telegram_id
                user_id = user.id

            # Update the message to show we're working on it
            await query.edit_message_text(f"🤖 **Working on your request...**\n\n📄 *{escape_markdown(doc_title)}*\n\n⏳ Please wait while I analyze the material...")

            # Generate appropriate response based on the action
            response = await self._generate_material_response(action, doc_id, doc_title, user_id)

            # Send the response (split if too long)
            response_parts = self._split_long_message(response)

            # Edit the first message
            await query.edit_message_text(response_parts[0])

            # Send additional parts as new messages if needed
            for part in response_parts[1:]:
                await context.bot.send_message(chat_id=query.message.chat_id, text=part)

            logger.info(f"📤 Handled {action} request for document {doc_title} by user {user_telegram_id}")

        except Exception as e:
            logger.error(f"❌ Error handling material assistance: {e}")
            await query.edit_message_text("❌ Sorry, something went wrong while processing your request. Please try again.")

    async def _generate_material_response(self, action: str, doc_id: int, doc_title: str, user_id: int) -> str:
        """Generate appropriate response based on the requested action with RAG-powered assistance"""
        try:
            # Get document metadata from database
            with db_manager.get_session() as session:
                from src.data.models import Document
                document = session.query(Document).filter(Document.id == doc_id).first()

                if not document:
                    return "❌ Document not found"

                # Cache document metadata
                material_type = document.material_type
                submission_required = document.submission_required
                due_date = document.due_date
                questions_data = document.questions

            # Create context-aware queries based on the action type
            if action == "complete_assignment":
                # Special handling for assignments - extract and answer all questions
                return await self._complete_assignment_with_questions(doc_id, doc_title, user_id, material_type)
            elif action == "breakdown_assignment":
                query = f"Break down the assignment '{doc_title}' into smaller, manageable tasks. What steps should I follow?"
            elif action == "explain_assignment":
                query = f"Explain the requirements and expectations for the assignment '{doc_title}'. What exactly am I supposed to do?"
            elif action == "complete_quiz":
                query = f"Help me answer the quiz '{doc_title}'. Based on the course material, provide guidance for answering the quiz questions."
            elif action == "study_quiz":
                query = f"Help me prepare for the quiz/test '{doc_title}'. What topics should I focus on studying?"
            elif action == "practice_quiz":
                query = f"Create practice questions based on the material in '{doc_title}' to help me prepare."
            elif action == "complete_question":
                # For question type, use the actual question from metadata if available
                if questions_data:
                    question_text = questions_data.get('question', doc_title)
                    question_type = questions_data.get('type', 'unknown')

                    if question_type == 'multiple_choice':
                        choices = questions_data.get('choices', [])
                        choices_text = '\n'.join([f"- {choice}" for choice in choices])
                        query = f"Help me answer this multiple choice question from my course materials:\n\nQuestion: {question_text}\n\nOptions:\n{choices_text}\n\nBased on the course content, which option is correct and why?"
                    else:
                        query = f"Help me answer this question from my course materials:\n\nQuestion: {question_text}\n\nProvide a comprehensive answer based on the course content."
                else:
                    query = f"Help me answer the question '{doc_title}' using information from my course materials."
            elif action == "hint_question":
                if questions_data:
                    question_text = questions_data.get('question', doc_title)
                    query = f"Give me a hint to help answer this question (don't give the full answer): {question_text}"
                else:
                    query = f"Give me a hint to help answer '{doc_title}' without revealing the full answer."
            elif action == "summarize_announcement":
                query = f"Summarize the announcement '{doc_title}' and highlight any important action items or deadlines."
            elif action == "ask_announcement":
                query = f"I have questions about the announcement '{doc_title}'. Can you help me understand what it means and what I need to do?"
            elif action == "summarize_material":
                query = f"Provide a comprehensive summary of '{doc_title}'"
            elif action == "questions_material":
                query = f"I have questions about the content in '{doc_title}'. Can you help me understand it better?"
            elif action == "keypoints_material":
                query = f"What are the main key points and important concepts in '{doc_title}'?"
            else:
                query = f"Help me understand the content in '{doc_title}'"

            # Load user with fresh session
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return "❌ User not found"

                # Use the RAG pipeline to get a contextual response
                if self.rag_pipeline:
                    response = await self._process_query_rag_enhanced(query, user, document_id=doc_id)

                    # Add a helpful header with due date if applicable
                    if action.startswith("complete_assignment"):
                        header = "✍️ <b>Assignment Completion Assistance</b>\n\n"
                        if due_date:
                            header += f"⏰ Due: {due_date.strftime('%B %d, %Y at %I:%M %p')}\n\n"
                    elif action.startswith("complete_quiz"):
                        header = "✍️ <b>Quiz Answering Assistance</b>\n\n"
                        if due_date:
                            header += f"⏰ Due: {due_date.strftime('%B %d, %Y at %I:%M %p')}\n\n"
                    elif action.startswith("complete_question"):
                        header = "✍️ <b>Question Answering Assistance</b>\n\n"
                        if due_date:
                            header += f"⏰ Due: {due_date.strftime('%B %d, %Y at %I:%M %p')}\n\n"
                    elif action.startswith("study_quiz") or action.startswith("practice_quiz"):
                        header = "🧠 <b>Study Assistance</b>\n\n"
                    elif action.startswith("summarize_"):
                        header = "📖 <b>Summary</b>\n\n"
                    else:
                        header = ""

                    return header + response
                else:
                    return "❌ Sorry, I'm currently unable to process your request. The content analysis system is not available."

        except Exception as e:
            logger.error(f"❌ Error generating material response: {e}")
            return "❌ Sorry, I encountered an error while analyzing the material. Please try asking your question directly in a message."


    async def _complete_assignment_with_questions(self, doc_id: int, doc_title: str, user_id: int, material_type: str) -> str:
        """
        Extract and answer all questions from an exam/assignment document.

        This implements Option 3 (Best approach) for handling multi-question documents:
        1. Retrieve ALL chunks from the document (sorted chronologically)
        2. Extract questions using regex patterns
        3. Answer each question individually using RAG
        4. Format responses chronologically
        """
        import re

        try:
            logger.info(f"Starting comprehensive assignment answering for document {doc_id}")

            # Step 1: Retrieve ALL chunks from document in chronological order
            all_chunks = self.rag_pipeline.retrieve_relevant_chunks(
                query="",  # Empty query - we want all chunks
                user_id=user_id,
                document_id=doc_id,
                top_k=100,  # Get many chunks
                min_similarity=0.0  # Accept ALL chunks from document
            )

            if not all_chunks:
                return "❌ Could not retrieve document content. Please try again."

            # Step 2: Sort chunks by chunk_index to preserve document order
            all_chunks.sort(key=lambda x: x['metadata'].get('chunk_index', 0))
            logger.info(f"Retrieved {len(all_chunks)} chunks from document")

            # Step 3: Build full document text
            full_text = "\n\n".join([chunk['text'] for chunk in all_chunks])
            logger.info(f"Built full document text ({len(full_text)} characters)")

            # Step 4: Extract questions using comprehensive regex patterns
            question_patterns = [
                # "Question ONE", "Question 1", "Q1:", "Q.1"
                r'(?:Question|Q)\.?\s*(\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|COMPULSORY)[:\.\s]+([^?]+\?(?:[^\n]*\n)*?)(?=(?:Question|Q)\.?\s*(?:\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)|$)',
                # Fallback: standalone questions ending with "?"
                r'(\d+)\.\s*([^?]+\?)',
            ]

            questions = []
            for pattern in question_patterns:
                matches = re.findall(pattern, full_text, re.DOTALL | re.IGNORECASE)
                if matches:
                    questions.extend(matches)
                    break  # Use first successful pattern

            if not questions:
                # Fallback: treat entire document as single question
                logger.warning("No questions detected with regex, treating as single question")
                return await self._answer_single_question(
                    full_text[:2000],  # Limit to first 2000 chars
                    doc_title,
                    user_id,
                    material_type
                )

            logger.info(f"Extracted {len(questions)} questions from document")

            # Step 5: Answer each question individually
            responses = []
            responses.append(f"📝 **Assignment: {doc_title}**\n")
            responses.append(f"I'll answer all questions based on the course materials.\n")
            responses.append("="*50 + "\n")

            for i, (q_num, q_text) in enumerate(questions, 1):
                logger.info(f"Answering question {i}/{len(questions)}: {q_num}")

                # Clean question text
                q_text = q_text.strip()

                # Get relevant context for THIS specific question
                # Note: generate_rag_response uses TOP_K_RETRIEVAL from settings
                question_context = self.rag_pipeline.generate_rag_response(
                    query=f"Answer this exam question comprehensively using course materials:\n\n{q_text[:800]}",
                    user_id=user_id,
                    document_id=None  # Search all accessible documents
                )

                # Format the response for this question
                responses.append(f"\n**Question {q_num}**\n")
                responses.append(f"{q_text}\n")
                responses.append(f"\n**Answer:**\n")
                responses.append(f"{question_context['response']}\n")
                responses.append("\n" + "-"*50 + "\n")

            # Step 6: Combine all responses
            final_response = "\n".join(responses)

            logger.info(f"Successfully answered {len(questions)} questions ({len(final_response)} chars)")

            return final_response

        except Exception as e:
            logger.error(f"Error in _complete_assignment_with_questions: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic approach
            return f"❌ Error processing assignment questions: {e}\n\nPlease try asking about specific questions individually."


    async def _answer_single_question(self, question_text: str, doc_title: str, user_id: int, material_type: str) -> str:
        """Fallback for when no questions are detected - treat entire document as one question."""
        try:
            # Load user
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return "❌ User not found"

                # Generate response
                query = f"Help me complete this assignment/exam:\n\n{question_text}"
                response = await self._process_query_rag_enhanced(query, user)

                return f"📝 **{doc_title}**\n\n{response}"
        except Exception as e:
            logger.error(f"Error in _answer_single_question: {e}")
            return f"❌ Error: {e}"


    async def _process_query_rag_enhanced(
        self,
        query: str,
        user: User,
        document_id: Optional[int] = None,
        pre_extracted_topics: Optional[List[str]] = None
    ) -> str:
        """Enhanced query processing with RAG pipeline and LLM integration

        Args:
            query: User's query text
            user: User object
            document_id: Optional specific document to search
            pre_extracted_topics: Optional pre-extracted topics to avoid redundant LLM calls
        """
        if not self.rag_pipeline:
            logger.error("RAG pipeline not initialized")
            return await self._process_query_basic(query, user)  # Fallback to basic

        try:
            # If document_id not provided, try to extract it from query
            searched_doc_title = None
            if not document_id:
                document_id = self._extract_document_from_query(query, user)
                # Check if we searched for a document but didn't find it
                if hasattr(self, '_last_searched_doc_title'):
                    searched_doc_title = self._last_searched_doc_title
                    delattr(self, '_last_searched_doc_title')

            # Determine course context from query
            course_id = self._extract_course_context(query, user)
            if not document_id:
                # If user explicitly mentioned a document that wasn't found,
                # don't filter by course - search all documents for related content
                if searched_doc_title:
                    logger.info(f"Document '{searched_doc_title}' not found - searching ALL user documents without course filter")
                    course_id = None # Override course context to search all
                else:
                    if course_id:
                        logger.info(f"Course context found: {course_id}. Searching within this course.")
                    else:
                        logger.info(f"No specific document or course - searching ALL documents for user {user.id}")

            # Get user preferences for personalization
            user_preferences = {
                'learning_style': user.learning_style,
                'difficulty_preference': user.difficulty_preference,
                'response_length': getattr(user, 'preferred_response_length', 'medium')
            }

            # Generate RAG response using the enhanced pipeline (pass pre-extracted topics)
            rag_result = self.rag_pipeline.generate_rag_response(
                query,
                user_id=user.id, # Pass user_id for data isolation
                course_id=course_id,
                document_id=document_id,
                user_preferences=user_preferences,
                pre_extracted_topics=pre_extracted_topics  # Pass topics to avoid re-extraction
            )
            
            # Format the response with sources and confidence indicator
            response = self._format_rag_response(rag_result, user)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG-enhanced processing: {e}")
            return await self._process_query_basic(query, user)
        
    def _format_rag_response(self, rag_result: Dict[str, Any], user: User) -> str:
        """Format RAG response - sources already added by RAG pipeline, just add confidence emoji"""
        # The response already includes sources formatted by _enhance_response_with_sources()
        # Don't duplicate the sources section, just return the response
        return rag_result['response']

    def _get_confidence_emoji(self, similarity_score: float) -> str:
        """Get emoji indicating confidence level"""
        if similarity_score >= 0.8:
            return "🎯"  # High confidence
        elif similarity_score >= 0.6:
            return "✅"  # Medium confidence
        else:
            return "📝"  # Low confidence


    def _extract_document_from_query(self, query: str, user: User) -> Optional[int]:
        """Extract document ID from query by matching document titles"""
        try:
            # Look for common document reference patterns
            import re

            # Patterns to match document references (with or without quotes)
            patterns = [
                r"(?:document|material|file|pdf|pptx)\s+['\"]([^'\"]+)['\"]",  # "document 'name'"
                r"(?:in|from|about|named|called|regards)\s+(?:the\s+)?(?:document|material|file)?\s*['\"]([^'\"]+)['\"]",  # "about 'name'"
                r"['\"]([^'\"]+\.(?:pdf|pptx|docx|txt))['\"]",  # "'filename.ext'"
                r"['\"]([^'\"]+)['\"]"  # any quoted text
            ]

            potential_titles = []
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                potential_titles.extend(matches)

            # Remove duplicates while preserving order
            potential_titles = list(dict.fromkeys(potential_titles))

            if potential_titles:
                logger.info(f"Extracted potential document titles from query: {potential_titles}")

            # Search for matching documents in database (only those user has access to)
            with db_manager.get_session() as session:
                from src.data.models import DocumentAccess

                # Get list of document IDs user has access to
                accessible_doc_ids = [
                    access.document_id
                    for access in session.query(DocumentAccess).filter(
                        DocumentAccess.user_id == user.id,
                        DocumentAccess.is_active == True
                    ).all()
                ]

                for title in potential_titles:
                    # Skip very short titles (likely false positives)
                    if len(title.strip()) < 3:
                        continue

                    # Try exact match first - but only from accessible documents
                    doc = session.query(Document).filter(
                        Document.title.ilike(f"%{title}%"),
                        Document.id.in_(accessible_doc_ids)  # HYBRID: Check user access
                    ).first()

                    if doc:
                        logger.info(f"✓ Found accessible document '{doc.title}' (ID: {doc.id}, course_id: {doc.course_id})")
                        return doc.id

                # If we searched but found nothing, store the title for fallback handling
                if potential_titles:
                    logger.warning(f"✗ Document not found for title: '{potential_titles[0]}'")
                    logger.info(f"→ Will search ALL documents (no course filter) for related content")
                    # Store in instance variable for the calling method to detect
                    self._last_searched_doc_title = potential_titles[0]

        except Exception as e:
            logger.warning(f"Error extracting document from query: {e}")

        return None

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
        
        # If no explicit course found, return None. The automatic fallback has been removed.
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
                        user_rating = 5 if is_helpful else 2
                        recent_interaction.was_helpful = is_helpful
                        recent_interaction.user_rating = user_rating
                        session.commit()

                        # === UPDATE STRUGGLE TOPICS WITH RATING ===
                        from src.services.adaptive_response_engine import adaptive_response_engine

                        # Extract topic from the query
                        topics = adaptive_response_engine.topic_analyzer.extract_topics(recent_interaction.query_text)
                        if topics:
                            current_topic = topics[0]

                            # Update struggle topic with rating
                            adaptive_response_engine.topic_analyzer.update_struggle_topic(
                                user_id=user.id,
                                topic=current_topic,
                                session=session,
                                user_rating=user_rating
                            )

            feedback_text = "👍 Thanks for the feedback! This helps me learn." if is_helpful else "👎 Thanks for the feedback. I'll work on improving my responses."
            await query.edit_message_text(feedback_text)

        except Exception as e:
            logger.error(f"Error handling feedback: {e}")
            await query.edit_message_text("Thanks for the feedback!")

    async def _handle_session_switch(self, update: Update, callback_data: str):
        """Handle switching to a different conversation session"""
        query = update.callback_query
        user_data = query.from_user

        try:
            # Extract session_id from callback_data (format: "switch_session:session_id")
            session_id = callback_data.replace("switch_session:", "")

            with db_manager.get_session() as session:
                user = self._get_or_create_user(session, user_data)
                user_id = user.id

                # Find the target session
                target_session = session.query(ConversationSession).filter(
                    and_(
                        ConversationSession.user_id == user_id,
                        ConversationSession.session_id == session_id
                    )
                ).first()

                if not target_session:
                    await query.edit_message_text("❌ Session not found. It may have been deleted.")
                    return

                # Close current active session
                current_session = session.query(ConversationSession).filter(
                    and_(
                        ConversationSession.user_id == user_id,
                        ConversationSession.is_active == True
                    )
                ).first()

                if current_session and current_session.session_id != session_id:
                    # Close the current session
                    session_manager.force_close_session(current_session.session_id)
                    logger.info(f"User {user_id} closed session {current_session.session_id} to switch")

                # Activate the target session
                target_session.is_active = True
                target_session.last_activity_at = datetime.now(timezone.utc)
                session.commit()

                # Get session context to show user
                topic = target_session.primary_topic or "General discussion"
                message_count = target_session.message_count

                # Get the last few interactions from this session to provide context
                session_interactions = session.query(UserInteraction).filter(
                    and_(
                        UserInteraction.user_id == user_id,
                        UserInteraction.session_id == session_id
                    )
                ).order_by(desc(UserInteraction.created_at)).limit(3).all()

                # Build context preview
                context_preview = ""
                if session_interactions:
                    recent_questions = [
                        f"   • {interaction.query_text[:60]}..."
                        for interaction in reversed(session_interactions)
                        if interaction.query_text
                    ]
                    if recent_questions:
                        context_preview = "\n\n📝 **Recent questions in this session:**\n" + "\n".join(recent_questions[:3])

                response_text = f"""✅ **Session Switched Successfully!**

📂 **Session Details:**
• Topic: {topic}
• Messages: {message_count}
• Started: {target_session.started_at.strftime('%b %d, %Y at %H:%M') if target_session.started_at else 'Unknown'}
{context_preview}

💬 You can now continue this conversation. I'll use the context from this session when responding to your questions.

💡 Use /new\\_session to start a fresh topic, or /my\\_sessions to switch to another session.
                """.strip()

                await query.edit_message_text(response_text, parse_mode='Markdown')

                # Log the session switch
                self._log_interaction(session, user_id, f"/switch_session:{session_id}", f"Switched to session: {topic}", "command")

                logger.info(f"User {user_id} switched to session {session_id} (topic: {topic})")

        except Exception as e:
            logger.error(f"Error in _handle_session_switch: {e}", exc_info=True)
            await query.edit_message_text(
                "❌ Sorry, I encountered an error while switching sessions. Please try again."
            )

    async def _handle_preference_change(self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str):
        """Handle user preference changes (learning style and difficulty)"""
        query = update.callback_query
        user_data = query.from_user

        try:
            # Parse the preference being changed
            preference_value = callback_data.replace("set_", "")

            # Learning style preferences (using actual engine values)
            learning_styles = {
                "example_driven": "Example-Driven",
                "analogy_driven": "Analogy-Driven",
                "socratic": "Socratic",
                "theory_first": "Theory-First",
                "adaptive": "Adaptive"
            }

            # Difficulty preferences
            difficulty_levels = {
                "easy": "Easy",
                "medium": "Medium",
                "hard": "Hard"
            }

            # Response length preferences
            response_lengths = {
                "short": "Short",
                "medium_length": "Medium",
                "long": "Long"
            }

            # Complexity level preferences
            complexity_levels = {
                "beginner": "Beginner",
                "intermediate": "Intermediate",
                "advanced": "Advanced"
            }

            # Learning pace preferences
            pace_levels = {
                "casual": "Casual",
                "moderate": "Moderate",
                "intensive": "Intensive"
            }

            with db_manager.get_session() as session:
                user = session.query(User).filter(User.telegram_id == str(user_data.id)).first()

                if not user:
                    await query.edit_message_text("❌ User not found. Please use /start first.")
                    return

                # Get or create profile for response length
                profile = session.query(PersonalizationProfile).filter(
                    PersonalizationProfile.user_id == user.id
                ).first()

                if not profile:
                    profile = PersonalizationProfile(
                        user_id=user.id,
                        preferred_response_length="medium"
                    )
                    session.add(profile)
                    session.flush()

                # Handle learning style change
                if preference_value in learning_styles:
                    old_style = user.learning_style
                    user.learning_style = preference_value

                    # Reset the classification timestamp so the 12-hour timer starts from now
                    user.last_style_classification = datetime.now(timezone.utc)

                    session.commit()

                    style_name = learning_styles[preference_value]

                    # Get style description
                    style_descriptions = {
                        "example_driven": "You'll learn through concrete examples first, followed by theory.",
                        "analogy_driven": "You'll learn through metaphors and real-world comparisons.",
                        "socratic": "You'll learn through guided questions and discovery.",
                        "theory_first": "You'll learn formal definitions and theory before examples.",
                        "adaptive": "I'll automatically detect and adapt to your learning style based on your questions."
                    }

                    description = style_descriptions.get(preference_value, "")

                    await query.edit_message_text(
                        f"✅ **Learning Style Updated!**\n\n"
                        f"Your learning style has been set to: **{style_name}**\n\n"
                        f"{description}\n\n"
                        f"Note: The system will automatically re-evaluate your learning style every 12 hours based on your interactions.\n\n"
                        f"You can change this anytime using /settings",
                        parse_mode='Markdown'
                    )

                    logger.info(f"User {user.id} changed learning style from '{old_style}' to '{preference_value}' (timestamp reset)")

                # Handle difficulty change
                elif preference_value in difficulty_levels:
                    old_difficulty = user.difficulty_preference
                    user.difficulty_preference = preference_value
                    session.commit()

                    difficulty_name = difficulty_levels[preference_value]

                    difficulty_descriptions = {
                        "easy": "Simple explanations with basic examples",
                        "medium": "Balanced detail with practical examples",
                        "hard": "Comprehensive explanations with advanced concepts"
                    }

                    description = difficulty_descriptions.get(preference_value, "")

                    await query.edit_message_text(
                        f"✅ **Difficulty Level Updated!**\n\n"
                        f"Your difficulty level has been set to: **{difficulty_name}**\n\n"
                        f"{description}\n\n"
                        f"You can change this anytime using /settings",
                        parse_mode='Markdown'
                    )

                    logger.info(f"User {user.id} changed difficulty from '{old_difficulty}' to '{preference_value}'")

                # Handle response length change
                elif preference_value in response_lengths:
                    # Convert 'medium_length' back to 'medium' for DB
                    actual_value = 'medium' if preference_value == 'medium_length' else preference_value
                    old_length = profile.preferred_response_length
                    profile.preferred_response_length = actual_value
                    session.commit()

                    length_name = response_lengths[preference_value]

                    length_descriptions = {
                        "short": "You'll get concise responses (3-5 paragraphs) focusing on core concepts.",
                        "medium_length": "You'll get balanced explanations (5-8 paragraphs) with examples and details.",
                        "long": "You'll get comprehensive, detailed responses (8-15 paragraphs) with multiple examples and in-depth coverage."
                    }

                    description = length_descriptions.get(preference_value, "")

                    await query.edit_message_text(
                        f"✅ **Response Length Updated!**\n\n"
                        f"Your response length preference has been set to: **{length_name}**\n\n"
                        f"{description}\n\n"
                        f"💡 Note: The system will automatically re-evaluate this preference every 12 hours based on which response lengths you rate most highly.\n\n"
                        f"You can change this anytime using /settings",
                        parse_mode='Markdown'
                    )

                    logger.info(f"User {user.id} changed response length from '{old_length}' to '{actual_value}'")

                # Handle complexity level change
                elif preference_value in complexity_levels:
                    # Map user-friendly names to numeric values (0.0-1.0)
                    complexity_mapping = {
                        "beginner": 0.2,      # Low complexity
                        "intermediate": 0.5,  # Medium complexity
                        "advanced": 0.8       # High complexity
                    }

                    old_complexity = profile.question_complexity_level
                    new_complexity_value = complexity_mapping[preference_value]
                    profile.question_complexity_level = new_complexity_value
                    session.commit()

                    complexity_name = complexity_levels[preference_value]

                    complexity_descriptions = {
                        "beginner": "Responses will start with absolute basics, define all terms clearly, and avoid assuming prior knowledge.",
                        "intermediate": "Responses will provide balanced explanations with some prior knowledge assumed.",
                        "advanced": "Responses will skip basic definitions, include edge cases and nuances, and reference advanced concepts freely."
                    }

                    description = complexity_descriptions.get(preference_value, "")

                    await query.edit_message_text(
                        f"✅ **Question Complexity Level Updated!**\n\n"
                        f"Your complexity level has been set to: **{complexity_name}**\n\n"
                        f"{description}\n\n"
                        f"💡 Note: The system will automatically re-evaluate this level every 12 hours based on your question patterns.\n\n"
                        f"You can change this anytime using /settings",
                        parse_mode='Markdown'
                    )

                    logger.info(f"User {user.id} changed complexity level from '{old_complexity}' to '{new_complexity_value}' ({preference_value})")

                # Handle learning pace change
                elif preference_value in pace_levels:
                    # Map user-friendly names to system values
                    pace_mapping = {
                        "casual": "slow",
                        "moderate": "medium",
                        "intensive": "fast"
                    }

                    old_pace = profile.learning_pace
                    new_pace_value = pace_mapping[preference_value]
                    profile.learning_pace = new_pace_value
                    session.commit()

                    pace_name = pace_levels[preference_value]

                    pace_descriptions = {
                        "casual": "Responses will be focused and digestible, emphasizing retention and clarity over breadth.",
                        "moderate": "Responses will have standard information density with a good mix of depth and breadth.",
                        "intensive": "Responses will pack more information, suggest related topics, and assume quick comprehension."
                    }

                    description = pace_descriptions.get(preference_value, "")

                    await query.edit_message_text(
                        f"✅ **Learning Pace Updated!**\n\n"
                        f"Your learning pace has been set to: **{pace_name}**\n\n"
                        f"{description}\n\n"
                        f"💡 Note: The system will automatically re-evaluate this pace every 12 hours based on your interaction patterns.\n\n"
                        f"You can change this anytime using /settings",
                        parse_mode='Markdown'
                    )

                    logger.info(f"User {user.id} changed learning pace from '{old_pace}' to '{new_pace_value}' ({preference_value})")

                else:
                    await query.edit_message_text("❌ Invalid preference option.")

        except Exception as e:
            logger.error(f"Error handling preference change: {e}", exc_info=True)
            await query.edit_message_text("❌ Failed to update preference. Please try again.")

    async def _handle_setting_change(self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str):
        """Handle settings changes"""
        query = update.callback_query
        setting_type = callback_data.replace("setting_", "")

        if setting_type == "learning_style":
            keyboard = [
                [
                    InlineKeyboardButton("📝 Example-Driven", callback_data="set_example_driven"),
                    InlineKeyboardButton("🔍 Analogy-Driven", callback_data="set_analogy_driven")
                ],
                [
                    InlineKeyboardButton("❓ Socratic", callback_data="set_socratic"),
                    InlineKeyboardButton("📚 Theory-First", callback_data="set_theory_first")
                ],
                [
                    InlineKeyboardButton("🧠 Adaptive", callback_data="set_adaptive")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📖 **Choose Your Learning Style:**\n\n"
                "• **Example-Driven**: Learn with concrete examples first, then theory\n"
                "• **Analogy-Driven**: Learn through metaphors and real-world comparisons\n"
                "• **Socratic**: Learn through guided questions and discovery\n"
                "• **Theory-First**: Learn formal definitions before examples\n"
                "• **Adaptive**: Let me automatically adapt to your preferences",
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

        elif setting_type == "response_length":
            keyboard = [
                [
                    InlineKeyboardButton("📏 Short", callback_data="set_short"),
                    InlineKeyboardButton("📐 Medium", callback_data="set_medium_length"),
                    InlineKeyboardButton("📊 Long", callback_data="set_long")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "📱 **Choose Response Length:**\n\n"
                "• **Short**: Concise responses (3-5 paragraphs)\n"
                "• **Medium**: Balanced explanations (5-8 paragraphs)\n"
                "• **Long**: Comprehensive, detailed responses (8-15 paragraphs)\n\n"
                "💡 Note: The system will automatically adjust this based on your ratings every 12 hours.",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )

        elif setting_type == "complexity":
            keyboard = [
                [
                    InlineKeyboardButton("🌱 Beginner", callback_data="set_beginner"),
                    InlineKeyboardButton("🌿 Intermediate", callback_data="set_intermediate"),
                    InlineKeyboardButton("🌳 Advanced", callback_data="set_advanced")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "🎯 **Choose Your Question Complexity Level:**\n\n"
                "• **Beginner**: Simple, foundational questions\n"
                "  → Responses start with absolute basics\n"
                "  → All terms clearly defined\n"
                "  → No prior knowledge assumed\n\n"
                "• **Intermediate**: Moderate complexity questions\n"
                "  → Balanced explanations\n"
                "  → Some prior knowledge assumed\n\n"
                "• **Advanced**: Complex, technical questions\n"
                "  → Skip basic definitions\n"
                "  → Include edge cases and nuances\n"
                "  → Advanced concepts referenced freely\n\n"
                "💡 Note: The system will automatically adjust this based on your question patterns every 12 hours.",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )

        elif setting_type == "pace":
            keyboard = [
                [
                    InlineKeyboardButton("🐢 Casual", callback_data="set_casual"),
                    InlineKeyboardButton("🚶 Moderate", callback_data="set_moderate"),
                    InlineKeyboardButton("🏃 Intensive", callback_data="set_intensive")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "⚡ **Choose Your Learning Pace:**\n\n"
                "• **Casual**: Relaxed, steady learning\n"
                "  → Focused, digestible information\n"
                "  → Emphasis on retention and clarity\n"
                "  → Patient with repetition\n\n"
                "• **Moderate**: Balanced learning pace\n"
                "  → Standard information density\n"
                "  → Good mix of depth and breadth\n\n"
                "• **Intensive**: Fast-paced, intensive learning\n"
                "  → More information per response\n"
                "  → Related topics suggested\n"
                "  → Assumes quick comprehension\n\n"
                "💡 Note: The system will automatically adjust this based on your interaction patterns every 12 hours.",
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

    def _log_interaction(self, session: Session, user_id: int, query: str, response: str, interaction_type: str, session_id: Optional[str] = None):
        """Log user interaction for analytics and personalization

        Args:
            session: Database session
            user_id: User ID
            query: User's query text
            response: Bot's response text
            interaction_type: Type of interaction (question, command, etc.)
            session_id: Optional conversation session ID. If not provided, will use active session.
        """
        # Get session_id if not provided
        if not session_id:
            active_session = session.query(ConversationSession).filter(
                and_(
                    ConversationSession.user_id == user_id,
                    ConversationSession.is_active == True
                )
            ).order_by(desc(ConversationSession.last_activity_at)).first()

            if active_session:
                session_id = active_session.session_id

        interaction = UserInteraction(
            user_id=user_id,
            session_id=session_id,  # Link to conversation session
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
            profile.last_interaction = datetime.now(timezone.utc)

        session.commit()

    # ============= QUIZ FUNCTIONALITY =============

    def _extract_quiz_topic(self, query_text: str) -> Optional[str]:
        """Extract topic from quiz request query"""
        import re

        query_lower = query_text.lower()

        # Patterns to extract topic - ordered from most specific to least specific
        patterns = [
            # "quiz me on/about X"
            r'quiz me (?:on|about|regarding)\s+(.+)',
            # "ask me questions on/about X"
            r'ask me questions (?:on|about|regarding)\s+(.+)',
            # "test me on/about X"
            r'test me (?:on|about|regarding)\s+(.+)',
            # "generate questions on/about X"
            r'generate questions (?:on|about|regarding)\s+(.+)',
            # "question me on/about X"
            r'question me (?:on|about|regarding)\s+(.+)',
            # "provide a quiz on/about X"
            r'provide (?:a )?quiz (?:on|about|regarding)\s+(.+)',
            # "format/formate a quiz on X"
            r'forma?te? (?:a |another )?quiz (?:on|about|regarding)\s+(.+)',
            # "create a quiz on X"
            r'create (?:a |another )?quiz (?:on|about|regarding)\s+(.+)',
            # "give me a quiz on X"
            r'give me (?:a )?quiz (?:on|about|regarding)\s+(.+)',
            # "another quiz on X" or "new quiz on X"
            r'(?:another|new) quiz (?:on|about|regarding)\s+(.+)',
            # "start a quiz on X"
            r'(?:start|begin) (?:a )?quiz (?:on|about|regarding)\s+(.+)',
            # Fallback: "quiz on X" (no "me")
            r'quiz (?:on|about|regarding)\s+(.+)',
            # Fallback: "questions on X"
            r'questions (?:on|about|regarding)\s+(.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                topic = match.group(1).strip()
                # Remove trailing punctuation
                topic = topic.rstrip('.,!?')
                return topic

        return None

    async def _initiate_quiz_from_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE, document_id: str):
        """Initiate a quiz session from a document"""
        query = update.callback_query
        user_data = query.from_user

        try:
            await query.edit_message_text("🧠 Generating quiz questions from the material... Please wait.")

            # Get user
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.telegram_id == str(user_data.id)).first()
                if not user:
                    await query.edit_message_text("❌ Please use /start first.")
                    return

                # Check if user already has an active quiz
                active_quiz = session.query(QuizSession).filter(
                    QuizSession.user_id == user.id,
                    QuizSession.is_active == True
                ).first()

                if active_quiz:
                    await query.edit_message_text(
                        "⚠️ You already have an active quiz session. Please complete it first or type /cancel_quiz to cancel it."
                    )
                    return

                user_id = user.id

            # Generate questions using RAG pipeline
            questions = self.rag_pipeline.generate_quiz_questions(user_id=user_id, document_id=int(document_id), num_questions=5)

            if not questions:
                await query.edit_message_text(
                    "❌ Sorry, I couldn't generate questions from this material. The content might not be suitable for quiz generation."
                )
                return

            # Create quiz session
            with db_manager.get_session() as session:
                quiz_session = QuizSession(
                    user_id=user_id,
                    document_id=int(document_id),
                    questions=questions,
                    current_question_index=0,
                    total_questions=len(questions),
                    is_active=True,
                    is_paused=False
                )
                session.add(quiz_session)
                session.commit()
                quiz_id = quiz_session.id

            # Send the first question
            await self._send_quiz_question(context, query.message.chat_id, quiz_id)
            await query.message.delete()  # Delete the "generating..." message

            logger.info(f"✅ Started quiz session {quiz_id} for user {user_data.id} from document {document_id}")

        except Exception as e:
            logger.error(f"❌ Error initiating quiz from document: {e}")
            await query.edit_message_text("❌ Sorry, failed to start the quiz. Please try again.")

    async def _initiate_quiz_from_topic(
        self,
        chat_id: int,
        user_id: int,
        topic: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """Initiate a quiz session from a topic search

        Args:
            chat_id: Telegram chat ID
            user_id: User database ID
            topic: Topic to generate quiz about
            conversation_history: Optional conversation history for context-based quizzes
        """
        try:
            # Send initial message
            if conversation_history:
                msg = await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=f"🧠 Generating quiz questions about our conversation on '{topic}'... Please wait."
                )
            else:
                msg = await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=f"🧠 Generating quiz questions about '{topic}'... Please wait."
                )

            # Check if user already has an active quiz
            with db_manager.get_session() as session:
                active_quiz = session.query(QuizSession).filter(
                    QuizSession.user_id == user_id,
                    QuizSession.is_active == True
                ).first()

                if active_quiz:
                    await msg.edit_text(
                        "⚠️ You already have an active quiz session. Please complete it first or type /cancel_quiz to cancel it."
                    )
                    return

            # Generate questions using RAG pipeline
            # If conversation_history is provided, use it as additional context
            questions = self.rag_pipeline.generate_quiz_questions(
                user_id=user_id,
                topic=topic,
                num_questions=5,
                conversation_history=conversation_history
            )

            if not questions:
                await msg.edit_text(
                    f"❌ Sorry, I couldn't find enough material about '{topic}' to generate quiz questions."
                )
                return

            # Create quiz session
            with db_manager.get_session() as session:
                quiz_session = QuizSession(
                    user_id=user_id,
                    topic=topic,
                    questions=questions,
                    current_question_index=0,
                    total_questions=len(questions),
                    is_active=True,
                    is_paused=False
                )
                session.add(quiz_session)
                session.commit()
                quiz_id = quiz_session.id

            # Send the first question
            await self._send_quiz_question(self.application, chat_id, quiz_id)
            await msg.delete()  # Delete the "generating..." message

            logger.info(f"✅ Started quiz session {quiz_id} for user {user_id} about topic '{topic}'")

        except Exception as e:
            logger.error(f"❌ Error initiating quiz from topic: {e}")
            await self.application.bot.send_message(
                chat_id=chat_id,
                text="❌ Sorry, failed to start the quiz. Please try again."
            )

    async def _send_quiz_question(self, context_or_app, chat_id: int, quiz_id: int):
        """Send the current quiz question to the user"""
        try:
            with db_manager.get_session() as session:
                quiz = session.query(QuizSession).filter(QuizSession.id == quiz_id).first()

                if not quiz or not quiz.is_active:
                    return

                current_idx = quiz.current_question_index
                question_data = quiz.questions[current_idx]

                # Build question text
                question_text = f"❓ **Question {current_idx + 1}/{quiz.total_questions}**\n\n"
                question_text += f"{question_data['question']}\n\n"

                # Create inline keyboard with options
                keyboard = []
                for i, option in enumerate(question_data['options']):
                    keyboard.append([
                        InlineKeyboardButton(
                            f"{chr(65 + i)}) {option}",
                            callback_data=f"quiz_answer_{quiz_id}_{i}"
                        )
                    ])

                reply_markup = InlineKeyboardMarkup(keyboard)

                # Send question as new message
                if hasattr(context_or_app, 'bot'):
                    await context_or_app.bot.send_message(
                        chat_id=chat_id,
                        text=question_text,
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )
                else:
                    await context_or_app.send_message(
                        chat_id=chat_id,
                        text=question_text,
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )

        except Exception as e:
            logger.error(f"❌ Error sending quiz question: {e}")

    async def _handle_quiz_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str):
        """Handle user's answer selection"""
        query = update.callback_query
        await query.answer()

        try:
            # Parse callback data: quiz_answer_{quiz_id}_{option_index}
            parts = callback_data.split("_")
            quiz_id = int(parts[2])
            selected_option = int(parts[3])

            # Cache all needed data INSIDE the session scope
            with db_manager.get_session() as session:
                quiz = session.query(QuizSession).filter(QuizSession.id == quiz_id).first()

                if not quiz or not quiz.is_active:
                    await query.edit_message_text("❌ Quiz session not found or expired.")
                    return

                # Cache data before session closes
                current_idx = quiz.current_question_index
                total_questions = quiz.total_questions
                question_data = quiz.questions[current_idx].copy()  # Make a copy
                correct_answer = question_data['correct_answer_index']
                is_correct = (selected_option == correct_answer)

                # Update score
                if is_correct:
                    quiz.correct_answers += 1
                else:
                    quiz.wrong_answers += 1

                session.commit()
                # Session closes here - all data is now cached in local variables

            # Edit message to show answer feedback with colored indicators (remove buttons)
            # Use cached variables instead of quiz object
            feedback_text = f"❓ **Question {current_idx + 1}/{total_questions}**\n\n"
            feedback_text += f"{question_data['question']}\n\n"

            # Show options with indicators
            for i, option in enumerate(question_data['options']):
                if i == correct_answer:
                    feedback_text += f"✅ {chr(65 + i)}) {option}\n"
                elif i == selected_option and not is_correct:
                    feedback_text += f"❌ {chr(65 + i)}) {option}\n"
                else:
                    feedback_text += f"   {chr(65 + i)}) {option}\n"

            feedback_text += f"\n{'🎉 Correct!' if is_correct else '❌ Incorrect!'}\n"

            # Edit the question message to show feedback (no buttons)
            await query.edit_message_text(
                text=feedback_text,
                parse_mode='Markdown'
            )

            # Automatically send explanation as a NEW message with Continue buttons
            explanation = question_data.get('explanation', 'No explanation available.')
            explanation_text = f"💡 **Explanation:**\n\n{explanation}\n\n"
            explanation_text += "Would you like to continue with the quiz?"

            keyboard = [
                [
                    InlineKeyboardButton("✅ Yes, Continue", callback_data=f"quiz_continue_{quiz_id}"),
                    InlineKeyboardButton("❌ No, End Quiz", callback_data=f"quiz_end_{quiz_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=explanation_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )

        except Exception as e:
            logger.error(f"❌ Error handling quiz answer: {e}")
            await query.edit_message_text("❌ Error processing your answer.")

    async def _handle_quiz_continue(self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str):
        """Continue to next question or show results"""
        query = update.callback_query
        await query.answer()

        try:
            # Parse callback data
            parts = callback_data.split("_")
            quiz_id = int(parts[2])

            with db_manager.get_session() as session:
                quiz = session.query(QuizSession).filter(QuizSession.id == quiz_id).first()

                if not quiz:
                    await query.edit_message_text("❌ Quiz session not found.")
                    return

                # Cache explanation text before modifying quiz
                current_idx = quiz.current_question_index
                question_data = quiz.questions[current_idx]
                explanation = question_data.get('explanation', 'No explanation available.')

                # Move to next question
                quiz.current_question_index += 1
                quiz.last_interaction_at = datetime.now()

                # Check if quiz is complete
                if quiz.current_question_index >= quiz.total_questions:
                    # Quiz complete
                    quiz.is_active = False
                    quiz.completed_at = datetime.now()

                    # Cache quiz results data before session closes
                    correct_answers = quiz.correct_answers
                    wrong_answers = quiz.wrong_answers

                    session.commit()

                    # Remove buttons from explanation message (keep explanation visible)
                    explanation_text = f"💡 **Explanation:**\n\n{explanation}"
                    await query.edit_message_text(explanation_text, parse_mode='Markdown')

                    # Show final results as NEW message (don't edit explanation)
                    await self._show_quiz_results_as_new_message(context, query.message.chat_id, correct_answers, wrong_answers)
                else:
                    # More questions remaining
                    session.commit()

                    # Remove buttons from explanation message (keep the explanation visible)
                    explanation_text = f"💡 **Explanation:**\n\n{explanation}"
                    await query.edit_message_text(explanation_text, parse_mode='Markdown')

                    # Send next question as NEW message (don't edit explanation)
                    await self._send_quiz_question(context, query.message.chat_id, quiz_id)

        except Exception as e:
            logger.error(f"❌ Error continuing quiz: {e}")

    async def _handle_quiz_no_continue(self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str):
        """Handle user declining to continue without explanation"""
        query = update.callback_query
        await query.answer()

        try:
            # Parse callback data
            parts = callback_data.split("_")
            quiz_id = int(parts[2])

            with db_manager.get_session() as session:
                quiz = session.query(QuizSession).filter(QuizSession.id == quiz_id).first()

                if not quiz:
                    await query.edit_message_text("❌ Quiz session not found.")
                    return

                # Move to next question
                quiz.current_question_index += 1
                quiz.last_interaction_at = datetime.now()

                # Check if quiz is complete
                if quiz.current_question_index >= quiz.total_questions:
                    # Quiz complete
                    quiz.is_active = False
                    quiz.completed_at = datetime.now()
                    session.commit()

                    # Show final results
                    await self._show_quiz_results(query, quiz)
                else:
                    # More questions remaining
                    session.commit()
                    await self._send_quiz_question(context, query.message.chat_id, quiz_id)

        except Exception as e:
            logger.error(f"❌ Error in quiz no-continue: {e}")

    async def _handle_quiz_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str):
        """Resume a paused quiz"""
        query = update.callback_query
        await query.answer()

        try:
            quiz_id = int(callback_data.split("_")[2])

            with db_manager.get_session() as session:
                quiz = session.query(QuizSession).filter(QuizSession.id == quiz_id).first()

                if not quiz:
                    await query.edit_message_text("❌ Quiz session not found.")
                    return

                # Resume quiz
                quiz.is_paused = False
                quiz.last_interaction_at = datetime.now()
                session.commit()

            await query.edit_message_text("Resuming quiz...")
            await self._send_quiz_question(context, query.message.chat_id, quiz_id)

        except Exception as e:
            logger.error(f"❌ Error resuming quiz: {e}")

    async def _handle_quiz_end(self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str):
        """End quiz early"""
        query = update.callback_query
        await query.answer()

        try:
            quiz_id = int(callback_data.split("_")[2])

            with db_manager.get_session() as session:
                quiz = session.query(QuizSession).filter(QuizSession.id == quiz_id).first()

                if not quiz:
                    await query.edit_message_text("❌ Quiz session not found.")
                    return

                # End quiz
                quiz.is_active = False
                quiz.completed_at = datetime.now()
                session.commit()

                # Show results
                await self._show_quiz_results(query, quiz)

        except Exception as e:
            logger.error(f"❌ Error ending quiz: {e}")

    async def _show_quiz_results(self, query, quiz: QuizSession):
        """Show final quiz results (legacy method that edits message)"""
        try:
            total = quiz.correct_answers + quiz.wrong_answers
            percentage = (quiz.correct_answers / total * 100) if total > 0 else 0

            results_text = f"🎯 **Quiz Complete!**\n\n"
            results_text += f"📊 **Your Results:**\n"
            results_text += f"✅ Correct: {quiz.correct_answers}\n"
            results_text += f"❌ Wrong: {quiz.wrong_answers}\n"
            results_text += f"📈 Score: {percentage:.1f}%\n\n"

            if percentage >= 80:
                results_text += "🌟 Excellent work! You've mastered this material!"
            elif percentage >= 60:
                results_text += "👍 Good job! Keep studying to improve further."
            else:
                results_text += "📚 Keep practicing! Review the material and try again."

            results_text += "\n\nIs there any other topic you'd like me to help you with?"

            await query.edit_message_text(results_text, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"❌ Error showing quiz results: {e}")

    async def _show_quiz_results_as_new_message(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, correct_answers: int, wrong_answers: int):
        """Show final quiz results as a NEW message (preserves explanation)"""
        try:
            total = correct_answers + wrong_answers
            percentage = (correct_answers / total * 100) if total > 0 else 0

            results_text = f"🎯 **Quiz Complete!**\n\n"
            results_text += f"📊 **Your Results:**\n"
            results_text += f"✅ Correct: {correct_answers}\n"
            results_text += f"❌ Wrong: {wrong_answers}\n"
            results_text += f"📈 Score: {percentage:.1f}%\n\n"

            if percentage >= 80:
                results_text += "🌟 Excellent work! You've mastered this material!"
            elif percentage >= 60:
                results_text += "👍 Good job! Keep studying to improve further."
            else:
                results_text += "📚 Keep practicing! Review the material and try again."

            results_text += "\n\nIs there any other topic you'd like me to help you with?"

            await context.bot.send_message(
                chat_id=chat_id,
                text=results_text,
                parse_mode='Markdown'
            )

        except Exception as e:
            logger.error(f"❌ Error showing quiz results as new message: {e}")

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