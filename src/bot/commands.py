import logging
from typing import Any, Dict, Optional, Callable

from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from sqlalchemy.orm import Session

from config.database import db_manager
from config.settings import settings
from src.data.models import User, UserInteraction, PersonalizationProfile
from src.services.scheduler import scheduler_service
from src.services.lms_integration import lms_service

logger = logging.getLogger(__name__)


# Basic command implementations extracted from handlers

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_data = update.effective_user
    
    try:
        with db_manager.get_session() as session:
            # Handle user creation directly here
            user = session.query(User).filter(User.telegram_id == str(user_data.id)).first()
            
            if not user:
                user = User(
                    telegram_id=str(user_data.id),
                    username=user_data.username,
                    first_name=user_data.first_name,
                    last_name=user_data.last_name,
                    learning_style="adaptive",
                    difficulty_preference="medium"
                )
                session.add(user)
                session.flush()
                
                # Create personalization profile
                profile = PersonalizationProfile(
                    user_id=user.id,
                    total_interactions=0,
                    successful_interactions=0,
                    avg_session_duration=0.0
                )
                session.add(profile)
                session.commit()

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
                    InlineKeyboardButton("🔄 Sync Now", callback_data="sync_now"),
                    InlineKeyboardButton("📊 Status", callback_data="status"),
                ],
                [
                    InlineKeyboardButton("❓ Help", callback_data="help"),
                    InlineKeyboardButton("👤 Profile", callback_data="profile"),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(welcome_text, reply_markup=reply_markup)

            ##log_interaction(session, user.id, "/start", welcome_text, "command")

    except Exception as e:
        logger.error(f"Error in start_command: {e}")
        await update.message.reply_text(
            "Sorry, I encountered an error. Please try again later."
        )


async def sync_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        status = scheduler_service.get_sync_status()

        status_text = f"""
📊 **System Status**

**Scheduler:** {'🟢 Running' if status['running'] else '🔴 Stopped'}
**Connected LMS:** {', '.join(status['connected_platforms']) if status['connected_platforms'] else 'None'}
**RAG Pipeline:** {'🟢 Available' if status['rag_available'] else '🔴 Not Available'}

**Documents:**
• Total: {status['documents']['total']}
• Processed: {status['documents']['processed']}
• Pending: {status['documents']['pending']}
• Processing: {status['documents']['processing']}
• Failed: {status['documents']['failed']}

**Courses:**
• Total: {status['courses']['total']}
• Active: {status['courses']['active']}

**Last Update:** {status['last_update'].strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()

        await update.message.reply_text(status_text, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Error in status_command: {e}")
        await update.message.reply_text("❌ Failed to get system status.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_data = update.effective_user

    try:
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.telegram_id == str(user_data.id)).first()

            if not user:
                if update.message:
                    await update.message.reply_text("Please use /start first to initialize your profile.")
                elif update.callback_query:
                    await update.callback_query.edit_message_text("Please use /start first to initialize your profile.")
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

            if update.message:
                await update.message.reply_text(profile_text, parse_mode='Markdown')
            elif update.callback_query:
                await update.callback_query.edit_message_text(profile_text, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Error in profile_command: {e}")
        if update.message:
            await update.message.reply_text("Sorry, couldn't retrieve your profile. Please try again.")
        elif update.callback_query:
            await update.callback_query.edit_message_text("Sorry, couldn't retrieve your profile. Please try again.")


async def courses_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def connect_classroom_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def disconnect_classroom_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def connection_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

• **Learning Style**: How you prefer to learn (example-driven, socratic, etc.)
• **Difficulty Level**: Complexity of explanations you prefer  
• **Notifications**: When to receive updates about new content
• **Response Length**: How detailed you want my responses

Choose a setting to modify:
    """.strip()

    await update.message.reply_text(settings_text, reply_markup=reply_markup, parse_mode='Markdown')


