import logging
from typing import Optional, Dict, Any
import json
from datetime import datetime, timedelta
import secrets
import base64
import hmac
import hashlib
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.database import db_manager
from config.settings import settings
from src.data.models import User

logger = logging.getLogger(__name__)

class UserOAuthManager:
    """Manages per-user OAuth flows for Google Classroom"""
    
    SCOPES = [
        'https://www.googleapis.com/auth/classroom.courses.readonly',
        'https://www.googleapis.com/auth/classroom.student-submissions.students.readonly',
        'https://www.googleapis.com/auth/drive.readonly'
    ]
    
    def __init__(self, credentials_file_path: str, redirect_uri: str):
        self.credentials_file = credentials_file_path
        self.redirect_uri = redirect_uri
        # Stateless state signing; avoid in-memory flow coupling
    
    def initiate_oauth_flow(self, user_id: int) -> Dict[str, str]:
        """Initiate OAuth flow for a specific user"""
        try:
            # Create flow
            flow = Flow.from_client_secrets_file(
                self.credentials_file,
                scopes=self.SCOPES,
                redirect_uri=self.redirect_uri
            )
            
            # Generate signed state parameter embedding user_id and expiry
            state = self._generate_signed_state(user_id=user_id, expires_in_seconds=600)
            
            # Generate authorization URL
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent',
                state=state
            )
            
            return {
                'auth_url': auth_url,
                'state': state,
                'expires_in': 600  # 10 minutes
            }
            
        except Exception as e:
            logger.error(f"Failed to initiate OAuth flow: {e}")
            raise
    
    def complete_oauth_flow(self, state: str, authorization_code: str) -> bool:
        """Complete OAuth flow with authorization code"""
        try:
            logger.info(f"Starting OAuth flow completion for state: {state[:20]}...")
            
            # Validate state and extract user id
            parsed = self._parse_and_verify_state(state)
            if not parsed:
                logger.error("Invalid or expired OAuth state")
                return False
            user_id = parsed['user_id']
            logger.info(f"Extracted user_id: {user_id} from state")

            # Recreate flow for token exchange
            logger.info(f"Creating flow with credentials file: {self.credentials_file}")
            flow = Flow.from_client_secrets_file(
                self.credentials_file,
                scopes=self.SCOPES,
                redirect_uri=self.redirect_uri
            )
            
            # Exchange authorization code for credentials
            logger.info("Exchanging authorization code for credentials...")
            flow.fetch_token(code=authorization_code)
            credentials = flow.credentials
            logger.info("Successfully obtained credentials")
            
            # Store credentials for user
            logger.info(f"Storing credentials for user {user_id}")
            success = self._store_user_credentials(user_id, credentials)
            if success:
                logger.info(f"Credentials stored successfully for user {user_id}")
                # Mark user as connected
                try:
                    with db_manager.get_session() as session:
                        user = session.query(User).filter(User.id == user_id).first()
                        if user:
                            user.google_classroom_connected = True
                            user.last_oauth_refresh = datetime.now()
                            session.commit()
                            logger.info(f"User {user_id} marked as connected")
                        else:
                            logger.error(f"User {user_id} not found in database")
                except Exception as mark_err:
                    logger.error(f"Failed to mark user {user_id} as connected: {mark_err}")
            else:
                logger.error(f"Failed to store credentials for user {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to complete OAuth flow: {e}", exc_info=True)
            return False
    
    def _store_user_credentials(self, user_id: int, credentials: Credentials) -> bool:
        """Store Google credentials for a user"""
        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return False
                
                # Serialize credentials
                creds_data = {
                    'token': credentials.token,
                    'refresh_token': credentials.refresh_token,
                    'token_uri': credentials.token_uri,
                    'client_id': credentials.client_id,
                    'client_secret': credentials.client_secret,
                    'scopes': credentials.scopes,
                    'expiry': credentials.expiry.isoformat() if credentials.expiry else None
                }
                
                # Store as encrypted JSON in user record
                # Note: In production, encrypt this data
                user.google_credentials = json.dumps(creds_data)
                session.commit()
                
                logger.info(f"Stored Google credentials for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store user credentials: {e}")
            return False
    
    def get_user_credentials(self, user_id: int) -> Optional[Credentials]:
        """Retrieve and refresh user credentials"""
        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user or not user.google_credentials:
                    logger.info(f"No credentials found for user {user_id}")
                    return None
                
                # Deserialize credentials with better error handling
                try:
                    creds_data = json.loads(user.google_credentials)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Invalid JSON in stored credentials for user {user_id}: {json_err}")
                    # Clear the malformed credentials
                    user.google_credentials = None
                    user.google_classroom_connected = False
                    session.commit()
                    return None
                
                credentials = Credentials(
                    token=creds_data.get('token'),
                    refresh_token=creds_data.get('refresh_token'),
                    token_uri=creds_data.get('token_uri'),
                    client_id=creds_data.get('client_id'),
                    client_secret=creds_data.get('client_secret'),
                    scopes=creds_data.get('scopes')
                )
                
                if creds_data.get('expiry'):
                    try:
                        credentials.expiry = datetime.fromisoformat(creds_data['expiry'])
                    except ValueError as date_err:
                        logger.error(f"Invalid expiry date in credentials for user {user_id}: {date_err}")
                        credentials.expiry = None
                
                # Refresh if needed
                if credentials.expired and credentials.refresh_token:
                    try:
                        credentials.refresh(Request())
                        # Update stored credentials
                        self._store_user_credentials(user_id, credentials)
                    except Exception as refresh_err:
                        logger.error(f"Failed to refresh credentials for user {user_id}: {refresh_err}")
                        return None
                
                return credentials
                
        except Exception as e:
            logger.error(f"Failed to get user credentials for user {user_id}: {e}", exc_info=True)
            return None
    
    def _generate_signed_state(self, user_id: int, expires_in_seconds: int) -> str:
        """Generate a signed state token embedding user_id and expiry."""
        expiry = int(time.time()) + int(expires_in_seconds)
        nonce = secrets.token_urlsafe(16)
        payload = f"{user_id}:{expiry}:{nonce}"
        signature = hmac.new(
            key=settings.SECRET_KEY.encode("utf-8"),
            msg=payload.encode("utf-8"),
            digestmod=hashlib.sha256
        ).hexdigest()
        token = f"{payload}:{signature}"
        return base64.urlsafe_b64encode(token.encode("utf-8")).decode("utf-8")

    def _parse_and_verify_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Verify signed state and return its contents if valid."""
        try:
            decoded = base64.urlsafe_b64decode(state.encode("utf-8")).decode("utf-8")
            parts = decoded.split(":")
            if len(parts) != 4:
                return None
            user_id_str, expiry_str, nonce, signature = parts
            payload = f"{user_id_str}:{expiry_str}:{nonce}"
            expected_sig = hmac.new(
                key=settings.SECRET_KEY.encode("utf-8"),
                msg=payload.encode("utf-8"),
                digestmod=hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(signature, expected_sig):
                return None
            if int(expiry_str) < int(time.time()):
                return None
            return {"user_id": int(user_id_str), "expiry": int(expiry_str), "nonce": nonce}
        except Exception:
            return None

class UserGoogleClassroomConnector:
    """Per-user Google Classroom connector"""
    
    def __init__(self, user_id: int, oauth_manager: UserOAuthManager):
        self.user_id = user_id
        self.oauth_manager = oauth_manager
        self.credentials = None
        self.service = None
        self.drive_service = None
    
    def authenticate(self) -> bool:
        """Authenticate this user's Google Classroom access"""
        try:
            self.credentials = self.oauth_manager.get_user_credentials(self.user_id)
            
            if not self.credentials:
                return False
            
            # Build services
            self.service = build('classroom', 'v1', credentials=self.credentials)
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to authenticate user {self.user_id}: {e}")
            return False
    
    def get_user_courses(self) -> list:
        """Get courses for this specific user"""
        if not self.service:
            return []
        
        try:
            # Get courses where user is a student
            student_results = self.service.courses().list(
                studentId='me',
                pageSize=50
            ).execute()
            
            courses = student_results.get('courses', [])
            
            # Also try to get courses where user is a teacher
            try:
                teacher_results = self.service.courses().list(
                    teacherId='me',
                    pageSize=50
                ).execute()
                
                teacher_courses = teacher_results.get('courses', [])
                
                # Combine and deduplicate
                all_course_ids = {course['id'] for course in courses}
                for course in teacher_courses:
                    if course['id'] not in all_course_ids:
                        courses.append(course)
            except HttpError:
                # User might not be a teacher in any courses
                pass
            
            return courses
        
        except HttpError as e:
            logger.error(f"Failed to get courses for user {self.user_id}: {e}")
            return []
    
    def is_authenticated(self) -> bool:
        """Check if user has valid authentication"""
        credentials = self.oauth_manager.get_user_credentials(self.user_id)
        return credentials is not None and not credentials.expired

# Updated bot handlers to support per-user OAuth
class GoogleClassroomOAuthHandlers:
    """OAuth handlers for bot commands"""
    
    def __init__(self, oauth_manager: UserOAuthManager):
        self.oauth_manager = oauth_manager
    
    async def handle_connect_classroom(self, update, context):
        """Handle /connect_classroom command"""
        user_data = update.effective_user
        
        with db_manager.get_session() as session:
            user = session.query(User).filter(
                User.telegram_id == str(user_data.id)
            ).first()
            
            if not user:
                await update.message.reply_text(
                    "Please use /start first to create your profile."
                )
                return
            
            try:
                # Check if already connected
                connector = UserGoogleClassroomConnector(user.id, self.oauth_manager)
                if connector.is_authenticated():
                    await update.message.reply_text(
                        "Your Google Classroom is already connected! Use /courses to view your classes."
                    )
                    return
                
                # Initiate OAuth flow
                flow_data = self.oauth_manager.initiate_oauth_flow(user.id)
                
                await update.message.reply_text(
                    f"To connect your Google Classroom:\n\n"
                    f"1. Click this link: {flow_data['auth_url']}\n"
                    f"2. Sign in with your Google account\n"
                    f"3. Grant the required permissions\n"
                    f"4. You'll be redirected back automatically\n\n"
                    f"This link expires in 10 minutes.",
                    disable_web_page_preview=True
                )
                
            except Exception as e:
                logger.error(f"Error initiating OAuth: {e}")
                await update.message.reply_text(
                    "Sorry, failed to initiate Google Classroom connection. Please try again."
                )
    
    async def handle_oauth_callback(self, request):
        """Handle OAuth callback (web endpoint)"""
        try:
            state = request.query_params.get('state')
            code = request.query_params.get('code')
            error = request.query_params.get('error')
            
            if error:
                return {"status": "error", "message": f"OAuth error: {error}"}
            
            if not state or not code:
                return {"status": "error", "message": "Missing OAuth parameters"}
            
            success = self.oauth_manager.complete_oauth_flow(state, code)
            
            if success:
                return {
                    "status": "success", 
                    "message": "Google Classroom connected successfully! Return to Telegram."
                }
            else:
                return {
                    "status": "error", 
                    "message": "Failed to complete OAuth flow"
                }
                
        except Exception as e:
            logger.error(f"OAuth callback error: {e}")
            return {"status": "error", "message": "Internal error"}

# Provide a shared OAuth manager instance configured from settings
oauth_manager = UserOAuthManager(
    credentials_file_path=settings.GOOGLE_CLASSROOM_CREDENTIALS,
    redirect_uri=settings.OAUTH_REDIRECT_URI,
)