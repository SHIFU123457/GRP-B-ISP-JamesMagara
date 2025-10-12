import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.database import db_manager
from src.data.models import User

logger = logging.getLogger(__name__)


class GoogleClassroomEmailParser:
    """Parse Google Classroom notification emails to extract material info"""

    # Email subject patterns for different material types
    # Order matters: check more specific patterns first (quiz before assignment)
    # Each type has patterns for both "New X:" and "New X posted:" formats
    PATTERNS = {
        'quiz': [
            r'New quiz assignment:?\s*(.+)',  # "New quiz assignment: Title"
            r'New quiz assignment posted:?\s*(.+)',
            r'Quiz assignment:?\s*(.+)',
            r'Quiz assignment posted:?\s*(.+)',
            r'New quiz:?\s*(.+)',  # "New quiz: Title"
            r'New quiz posted:?\s*(.+)',
            r'Quiz posted:?\s*(.+)',
            r'(.+)\s+posted a quiz assignment',
            r'(.+)\s+posted a quiz',
        ],
        'question': [
            r'New question:?\s*(.+)',  # "New question: Title"
            r'New question posted:?\s*(.+)',
            r'Question:?\s*(.+)',
            r'Question posted:?\s*(.+)',
            r'(.+)\s+posted a question',
        ],
        'assignment': [
            r'New assignment:?\s*(.+)',  # "New assignment: Title"
            r'New assignment posted:?\s*(.+)',
            r'Assignment:?\s*(.+)',
            r'Assignment posted:?\s*(.+)',
            r'(.+)\s+posted an assignment',
        ],
        'announcement': [
            r'New announcement:?\s*(.+)',  # "New announcement: Title"
            r'New announcement posted:?\s*(.+)',
            r'Announcement:?\s*(.+)',
            r'Announcement posted:?\s*(.+)',
            r'(.+)\s+posted an announcement',
            r'(.+)\s+made an announcement',
        ],
        'material': [
            r'New material:?\s*(.+)',  # "New material: Title"
            r'New material posted:?\s*(.+)',
            r'Material:?\s*(.+)',
            r'Material posted:?\s*(.+)',
            r'(.+)\s+posted material',
            r'(.+)\s+posted a material',
            r'(.+)\s+shared a file',
            r'(.+)\s+shared new material',
        ]
    }

    @classmethod
    def parse_subject(cls, subject: str) -> Optional[Dict[str, str]]:
        """
        Parse email subject to extract material type and title

        Returns:
            {
                'material_type': 'assignment'|'quiz'|'question'|'material'|'announcement',
                'title': 'Material title'
            }
        """
        if not subject:
            return None

        # Check patterns in order (quiz before assignment is important!)
        for material_type, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, subject, re.IGNORECASE)
                if match:
                    title = match.group(1).strip() if match.groups() else subject
                    return {
                        'material_type': material_type,
                        'title': title
                    }

        # Default to material if no specific type matched
        return {
            'material_type': 'material',
            'title': subject
        }

    @classmethod
    def extract_course_from_body(cls, email_body: str) -> Optional[str]:
        """Extract course name from email body"""
        # Google Classroom emails have a specific format:
        # Notification settings
        # [Course Name]
        # New assignment/material/quiz/etc.

        # Try patterns in order of reliability
        patterns = [
            # Pattern 1: Course name between URL and URL (most reliable for plain text emails)
            # Format: <URL>\nCourse Name\n<URL>
            r'<https://[^>]+>\s*[\r\n]+\s*([A-Z][^\r\n<]{3,80}?)\s*[\r\n]+\s*<https://',

            # Pattern 2: Course name after "Notification settings" line, URL, then course name
            # Format: Notification settings\n<URL>\nCourse Name\n
            r'Notification\s+settings\s*[\r\n]+\s*<[^>]+>\s*[\r\n]+\s*([A-Z][^\r\n<]{3,80}?)\s*[\r\n]',

            # Pattern 3: Course name directly before "New [type]" with URL separator
            r'([A-Z][^\r\n<]{3,60}?)\s*[\r\n]+\s*<[^>]+>\s*[\r\n]+\s*New\s+(?:assignment|material|quiz|announcement|question)',

            # Pattern 4: Traditional "posted in" format (fallback)
            r'posted\s+in\s+([^\n\r.<]+?)(?:\s*\n|\s*\.|\s*$|\s*<)',

            # Pattern 5: "in [Course]" format (fallback)
            r'in\s+([A-Z][^\n\r.<]+?)(?:\s*posted|\s*shared|\s*\n|\s*\.|\s*<)',

            # Pattern 6: "class: [Course]" format (fallback)
            r'class:\s*([^\n\r<]+)',
        ]

        for i, pattern in enumerate(patterns, 1):
            match = re.search(pattern, email_body, re.IGNORECASE | re.MULTILINE)
            if match:
                course_name = match.group(1).strip()

                # Clean up course name
                course_name = cls._clean_course_name(course_name)

                # Validate course name
                if cls._is_valid_course_name(course_name):
                    logger.debug(f"Extracted course name using pattern {i}: '{course_name}'")
                    return course_name

        logger.debug("Could not extract course name from email body")
        return None

    @classmethod
    def _clean_course_name(cls, course_name: str) -> str:
        """Clean extracted course name"""
        # Remove common unwanted text
        unwanted = ['Classroom Logo', 'Notification settings', 'See details', 'Posted on']
        for text in unwanted:
            course_name = course_name.replace(text, '')

        # Strip whitespace and tabs
        course_name = course_name.strip()

        return course_name

    @classmethod
    def _is_valid_course_name(cls, course_name: str) -> bool:
        """Validate if extracted text is a valid course name"""
        if not course_name or len(course_name) < 3:
            return False

        # Filter out common false matches
        invalid_patterns = [
            r'^\d{5}',  # ZIP codes
            r'\d{5}\s+(USA|CA|United States)',  # US/CA addresses (ZIP + country)
            r'^(settings|notification|logo|details|see details)$',  # UI text
            r'^http',  # URLs
            r'^\d{1,2}:\d{2}',  # Times
            r'@',  # Email addresses (contains @)
            r'&Email=',  # URL parameters with emails
            r'^\s*$',  # Empty or whitespace only
            r'Posted on',  # Timestamp lines
            r'\(EAT\)|\(GMT\)|\(UTC\)',  # Timezone indicators
            r'^[\w\.-]+@[\w\.-]+',  # Email format (starts with email)
            r',\s*(CA|NY|TX|FL|WA)\s+\d{5}',  # US addresses (City, State ZIP)
            r'Mountain View|Menlo Park|Palo Alto|San Francisco',  # Google office locations
            r'USA$|Canada$',  # Ends with country name
            r'^\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)',  # Street addresses
            r'^[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}',  # City, State ZIP format
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, course_name, re.IGNORECASE):
                logger.debug(f"Rejected course name '{course_name}' - matched invalid pattern: {pattern}")
                return False

        return True

    @classmethod
    def extract_classroom_link(cls, email_body: str) -> Optional[str]:
        """Extract Google Classroom link from email"""
        # Pattern for Classroom URLs
        link_pattern = r'https://classroom\.google\.com/[^\s<>\'"]*'
        match = re.search(link_pattern, email_body)
        if match:
            return match.group(0)
        return None


class GmailMonitorService:
    """Monitor Gmail for Google Classroom notifications"""

    CLASSROOM_SENDER = 'classroom.google.com'
    CLASSROOM_LABELS = ['Category_Updates', 'CATEGORY_UPDATES']

    def __init__(self, oauth_manager):
        """
        Initialize Gmail monitor

        Args:
            oauth_manager: UserOAuthManager instance for credential management
        """
        self.oauth_manager = oauth_manager
        self.parser = GoogleClassroomEmailParser()
        self.last_check = {}  # user_id -> last_check_timestamp

    def get_gmail_service(self, user_id: int):
        """Get Gmail API service for a user"""
        try:
            # Get credentials from oauth_manager (returns Credentials object, not dict)
            creds = self.oauth_manager.get_user_credentials(user_id)

            if not creds:
                logger.warning(f"No Google credentials found for user {user_id}")
                return None

            # Refresh if expired
            if creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    # Update stored credentials
                    self.oauth_manager.store_user_credentials(user_id, creds)
                except Exception as refresh_error:
                    logger.error(f"Failed to refresh credentials for user {user_id}: {refresh_error}")
                    return None

            # Build Gmail service (uses same OAuth token as Classroom)
            service = build('gmail', 'v1', credentials=creds)
            return service

        except Exception as e:
            logger.error(f"Failed to get Gmail service for user {user_id}: {e}")
            return None

    def check_new_classroom_emails(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Check for new Google Classroom notification emails

        Returns:
            List of {
                'material_type': str,
                'title': str,
                'course_name': str,
                'classroom_link': str,
                'email_id': str,
                'timestamp': datetime
            }
        """
        try:
            service = self.get_gmail_service(user_id)
            if not service:
                return []

            # Determine time range for search
            if user_id in self.last_check:
                # Check emails since last check
                last_check = self.last_check[user_id]
            else:
                # First check - look back 1 hour
                last_check = datetime.now() - timedelta(hours=1)

            # Convert to Gmail query format
            after_timestamp = int(last_check.timestamp())

            # Build Gmail search query
            query = f'from:({self.CLASSROOM_SENDER}) after:{after_timestamp}'

            # Search for emails
            results = service.users().messages().list(
                userId='me',
                q=query,
                maxResults=50
            ).execute()

            messages = results.get('messages', [])

            if not messages:
                logger.debug(f"No new Classroom emails for user {user_id}")
                self.last_check[user_id] = datetime.now()
                return []

            logger.info(f"Found {len(messages)} new Classroom emails for user {user_id}")

            # Parse each email
            parsed_notifications = []

            for msg in messages:
                try:
                    notification = self._parse_email(service, msg['id'], user_id)
                    if notification:
                        parsed_notifications.append(notification)
                except Exception as e:
                    logger.error(f"Failed to parse email {msg['id']}: {e}")
                    continue

            # Update last check timestamp
            self.last_check[user_id] = datetime.now()

            return parsed_notifications

        except HttpError as e:
            logger.error(f"Gmail API error for user {user_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error checking Classroom emails for user {user_id}: {e}")
            return []

    def _parse_email(self, service, email_id: str, user_id: int) -> Optional[Dict[str, Any]]:
        """Parse a single email message"""
        try:
            # Get full email message
            msg = service.users().messages().get(
                userId='me',
                id=email_id,
                format='full'
            ).execute()

            # Extract headers
            headers = {h['name']: h['value'] for h in msg['payload']['headers']}
            subject = headers.get('Subject', '')
            from_addr = headers.get('From', '')

            # Verify it's from Classroom
            if self.CLASSROOM_SENDER not in from_addr.lower():
                return None

            # Parse subject for material type and title
            parsed_subject = self.parser.parse_subject(subject)
            if not parsed_subject:
                logger.warning(f"Could not parse Classroom email subject: {subject}")
                return None

            # Extract email body
            body = self._extract_email_body(msg['payload'])

            # Extract course name and link
            course_name = self.parser.extract_course_from_body(body)
            classroom_link = self.parser.extract_classroom_link(body)

            # Enhanced debug logging for course extraction
            if course_name:
                logger.info(f"✅ Extracted course name: '{course_name}' from email")
            else:
                logger.warning(f"❌ Could not extract course name from email. Subject: '{subject}'")
                # Only show email body when extraction fails (for debugging)
                logger.debug(f"Email body preview:\n{body[:500]}")

            logger.debug(f"Extracted classroom link: '{classroom_link}'")

            # Get email timestamp
            timestamp = datetime.fromtimestamp(int(msg['internalDate']) / 1000)

            notification = {
                'material_type': parsed_subject['material_type'],
                'title': parsed_subject['title'],
                'course_name': course_name,
                'classroom_link': classroom_link,
                'email_id': email_id,
                'timestamp': timestamp,
                'subject': subject
            }

            logger.info(f"Parsed Classroom email: {parsed_subject['material_type']} - {parsed_subject['title']}")

            return notification

        except Exception as e:
            logger.error(f"Failed to parse email {email_id}: {e}")
            return None

    def _extract_email_body(self, payload: Dict) -> str:
        """Extract text from email payload"""
        try:
            body = ""

            if 'parts' in payload:
                # Multipart email
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        if 'data' in part['body']:
                            import base64
                            body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                            break
            elif 'body' in payload and 'data' in payload['body']:
                # Simple email
                import base64
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')

            return body

        except Exception as e:
            logger.error(f"Failed to extract email body: {e}")
            return ""

    def check_all_connected_users(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Check Gmail for all users with Google Classroom connected

        Returns:
            {user_id: [notifications]}
        """
        results = {}

        try:
            with db_manager.get_session() as session:
                # Get all users with Google Classroom connected
                connected_users = session.query(User).filter(
                    User.google_classroom_connected == True,
                    User.is_active == True
                ).all()

                logger.info(f"Checking Gmail for {len(connected_users)} connected users")

                for user in connected_users:
                    try:
                        notifications = self.check_new_classroom_emails(user.id)
                        if notifications:
                            results[user.id] = notifications
                            logger.info(f"User {user.id} has {len(notifications)} new Classroom notifications")
                    except Exception as e:
                        logger.error(f"Failed to check emails for user {user.id}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error checking all users: {e}")

        return results
