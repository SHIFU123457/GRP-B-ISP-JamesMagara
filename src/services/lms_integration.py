import logging
import requests
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod
from pathlib import Path
import time
import hashlib
from functools import wraps

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.settings import settings
from src.services.oauth_manager import UserOAuthManager, UserGoogleClassroomConnector, oauth_manager as shared_oauth_manager
from config.database import db_manager
from src.data.models import Course, Document
from src.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function calls on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator

class BaseLMSConnector(ABC):
    """Abstract base class for LMS connectors"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the LMS platform"""
        pass
    
    @abstractmethod
    def get_courses(self) -> List[Dict[str, Any]]:
        """Retrieve list of courses"""
        pass
    
    @abstractmethod
    def get_course_materials(self, course_id: str) -> List[Dict[str, Any]]:
        """Retrieve materials for a specific course"""
        pass
    
    @abstractmethod
    def download_document(self, document_info: Dict[str, Any]) -> Optional[str]:
        """Download a document and return local file path"""
        pass

class MoodleConnector(BaseLMSConnector):
    """Moodle LMS connector using Web Service API"""

    def __init__(self, base_url: str, token: str):
        super().__init__()
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.session = requests.Session()
        self.authenticated = False

    @retry_on_failure(max_retries=2, delay=1.0)
    def authenticate(self) -> bool:
        """Test authentication with Moodle"""
        try:
            response = self._make_request('core_webservice_get_site_info')
            if response and 'sitename' in response:
                self.authenticated = True
                logger.info(f"Successfully authenticated with Moodle: {response['sitename']}")
                return True
        except Exception as e:
            logger.error(f"Moodle authentication failed: {e}")
        
        return False
    
    def _make_request(self, function: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Make API request to Moodle"""
        if params is None:
            params = {}
        
        url = f"{self.base_url}/webservice/rest/server.php"
        data = {
            'wstoken': self.token,
            'wsfunction': function,
            'moodlewsrestformat': 'json',
            **params
        }
        
        try:
            response = self.session.post(url, data=data, timeout=30)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Moodle API request failed: {e}")
            return None
    
    def get_courses(self) -> List[Dict[str, Any]]:
        """Get all courses the user is enrolled in"""
        try:
            # First get user info
            user_info = self._make_request('core_webservice_get_site_info')
            if not user_info or 'userid' not in user_info:
                logger.error("Failed to get user information")
                return []
            
            user_id = user_info['userid']
            
            # Get enrolled courses
            courses_data = self._make_request('core_enrol_get_users_courses', {'userid': user_id})
            
            if not courses_data:
                return []
            
            courses = []
            for course in courses_data:
                courses.append({
                    'id': str(course['id']),
                    'name': course['fullname'],
                    'short_name': course['shortname'],
                    'category': course.get('categoryname', ''),
                    'start_date': datetime.fromtimestamp(course.get('startdate', 0)) if course.get('startdate') else None,
                    'platform': 'moodle'
                })
            
            logger.info(f"Retrieved {len(courses)} courses from Moodle")
            return courses
        
        except Exception as e:
            logger.error(f"Error retrieving Moodle courses: {e}")
            return []
    
    def get_course_materials(self, course_id: str) -> List[Dict[str, Any]]:
        """Get materials for a specific course"""
        try:
            # Get course contents
            contents = self._make_request('core_course_get_contents', {'courseid': int(course_id)})
            
            if not contents:
                return []
            
            materials = []
            
            for section in contents:
                section_name = section.get('name', 'General')
                
                for module in section.get('modules', []):
                    # Process files in the module
                    for content_file in module.get('contents', []):
                        if content_file['type'] == 'file':
                            # Check if it's a supported file type
                            filename = content_file['filename']
                            file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
                            
                            if file_ext in settings.SUPPORTED_FILE_TYPES:
                                materials.append({
                                    'id': str(content_file['fileurl']),  # Use file URL as ID
                                    'title': filename,
                                    'description': module.get('description', ''),
                                    'section': section_name,
                                    'file_url': content_file['fileurl'],
                                    'file_size': content_file.get('filesize', 0),
                                    'file_type': file_ext,
                                    'modified_date': datetime.fromtimestamp(content_file.get('timemodified', 0)) if content_file.get('timemodified') else datetime.now(),
                                    'course_id': course_id,
                                    'platform': 'moodle'
                                })
            
            logger.info(f"Retrieved {len(materials)} materials from course {course_id}")
            return materials
        
        except Exception as e:
            logger.error(f"Error retrieving course materials: {e}")
            return []
    
    def download_document(self, document_info: Dict[str, Any]) -> Optional[str]:
        """Download a document from Moodle"""
        try:
            file_url = document_info['file_url']
            filename = document_info['title']
            
            # Add token to URL
            if '?' in file_url:
                download_url = f"{file_url}&token={self.token}"
            else:
                download_url = f"{file_url}?token={self.token}"
            
            # Create local directory
            local_dir = Path("./data/documents")
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Create safe filename
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
            local_path = local_dir / f"{document_info['course_id']}_{safe_filename}"
            
            # Download file
            response = self.session.get(download_url, timeout=60)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded document: {local_path}")
            return str(local_path)
        
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            return None

class GoogleClassroomConnector(BaseLMSConnector):
    """Google Classroom connector using Google Classroom API"""
    
    SCOPES = [
        'https://www.googleapis.com/auth/classroom.courses.readonly',
        'https://www.googleapis.com/auth/classroom.coursework.students.readonly',
        'https://www.googleapis.com/auth/classroom.coursework.me.readonly',
        'https://www.googleapis.com/auth/classroom.student-submissions.students.readonly',
        'https://www.googleapis.com/auth/drive.readonly'
    ]
    
    def __init__(self, credentials_path: str):
        super().__init__()
        self.credentials_path = credentials_path
        self.service = None
        self.drive_service = None
        self.authenticated = False
    
    def authenticate(self) -> bool:
        """Authenticate with Google Classroom"""
        try:
            creds = None
            token_path = "token.json"
            
            # Load existing token
            if Path(token_path).exists():
                creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
            
            # If there are no (valid) credentials available, only refresh; do NOT start local server
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    logger.warning("No valid Google credentials found and interactive auth is disabled. Skipping Google Classroom connector initialization.")
                    return False
                
                # Save refreshed credentials for next run
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            
            # Build services
            self.service = build('classroom', 'v1', credentials=creds)
            self.drive_service = build('drive', 'v3', credentials=creds)
            self.authenticated = True
            
            logger.info("Successfully authenticated with Google Classroom")
            return True
        
        except Exception as e:
            logger.error(f"Google Classroom authentication failed: {e}")
            return False
    
    def get_courses(self) -> List[Dict[str, Any]]:
        """Get all courses from Google Classroom"""
        try:
            if not self.authenticated:
                logger.error("Not authenticated with Google Classroom")
                return []
            
            results = self.service.courses().list(pageSize=50).execute()
            courses_data = results.get('courses', [])
            
            courses = []
            for course in courses_data:
                courses.append({
                    'id': course['id'],
                    'name': course['name'],
                    'short_name': course.get('section', course['name'][:20]),
                    'category': course.get('descriptionHeading', ''),
                    'start_date': None,  # Google Classroom doesn't provide start dates directly
                    'platform': 'google_classroom'
                })
            
            logger.info(f"Retrieved {len(courses)} courses from Google Classroom")
            return courses
        
        except HttpError as e:
            logger.error(f"Error retrieving Google Classroom courses: {e}")
            return []
    
    def get_course_materials(self, course_id: str) -> List[Dict[str, Any]]:
        """Get materials for a specific course"""
        try:
            materials = []
            
            # Get course work (assignments, materials)
            coursework = self.service.courses().courseWork().list(courseId=course_id).execute()
            
            for work in coursework.get('courseWork', []):
                # Process attachments
                for material in work.get('materials', []):
                    if 'driveFile' in material:
                        drive_file = material['driveFile']['driveFile']
                        
                        # Check if it's a supported file type
                        filename = drive_file['title']
                        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
                        
                        if file_ext in settings.SUPPORTED_FILE_TYPES:
                            materials.append({
                                'id': drive_file['id'],
                                'title': filename,
                                'description': work.get('description', ''),
                                'section': work.get('topicId', 'General'),
                                'file_url': drive_file.get('alternateLink', ''),
                                'file_size': 0,  # Google Drive API requires separate call for size
                                'file_type': file_ext,
                                'modified_date': datetime.now(),  # Would need separate API call for actual date
                                'course_id': course_id,
                                'platform': 'google_classroom',
                                'drive_file_id': drive_file['id']
                            })
            
            logger.info(f"Retrieved {len(materials)} materials from course {course_id}")
            return materials
        
        except HttpError as e:
            logger.error(f"Error retrieving course materials: {e}")
            return []
    
    def download_document(self, document_info: Dict[str, Any]) -> Optional[str]:
        """Download a document from Google Drive"""
        try:
            file_id = document_info['drive_file_id']
            filename = document_info['title']
            
            # Create local directory
            local_dir = Path("./data/documents")
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Create safe filename
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
            local_path = local_dir / f"{document_info['course_id']}_{safe_filename}"
            
            # Download file
            request = self.drive_service.files().get_media(fileId=file_id)
            content = request.execute()
            
            with open(local_path, 'wb') as f:
                f.write(content)
            
            logger.info(f"Downloaded document: {local_path}")
            return str(local_path)
        
        except HttpError as e:
            logger.error(f"Error downloading document: {e}")
            return None

class LMSIntegrationService:
    """Main service for managing LMS integrations"""
    
    def __init__(self):
        self.connectors: Dict[str, BaseLMSConnector] = {}
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        """Initialize LMS connectors based on settings"""
        
        # Initialize Moodle connector if configured
        moodle_url = getattr(settings, 'MOODLE_BASE_URL', None)
        moodle_token = getattr(settings, 'MOODLE_API_TOKEN', None)
        
        if moodle_url and moodle_token:
            moodle_connector = MoodleConnector(moodle_url, moodle_token)
            if moodle_connector.authenticate():
                self.connectors['moodle'] = moodle_connector
                logger.info("Moodle connector initialized successfully")
            else:
                logger.error("Failed to initialize Moodle connector")
        
        # Initialize Google Classroom connector (server-wide) only if per-user OAuth is disabled
        gc_credentials_path = getattr(settings, 'GOOGLE_CLASSROOM_CREDENTIALS', None)
        if not getattr(settings, 'PER_USER_GOOGLE_OAUTH', True):
            if gc_credentials_path and Path(gc_credentials_path).exists():
                gc_connector = GoogleClassroomConnector(gc_credentials_path)
                if gc_connector.authenticate():
                    self.connectors['google_classroom'] = gc_connector
                    logger.info("Google Classroom connector initialized successfully (server-wide credentials)")
                else:
                    logger.error("Failed to initialize Google Classroom connector with server-wide credentials")
            else:
                logger.warning("Google Classroom credentials not found; skipping server-wide connector")
    
    def sync_courses(self) -> List[Course]:
        """Sync courses from all connected LMS platforms"""
        synced_courses = []
        
        with db_manager.get_session() as session:
            for platform, connector in self.connectors.items():
                try:
                    courses_data = connector.get_courses()
                    
                    for course_data in courses_data:
                        # Check if course already exists
                        existing_course = session.query(Course).filter(
                            Course.lms_course_id == course_data['id'],
                            Course.lms_platform == platform
                        ).first()
                        
                        if existing_course:
                            # Update existing course
                            existing_course.course_name = course_data['name']
                            existing_course.updated_at = datetime.now()
                            synced_courses.append(existing_course)
                        else:
                            # Create new course
                            new_course = Course(
                                course_code=course_data['short_name'],
                                course_name=course_data['name'],
                                description=course_data.get('category', ''),
                                lms_course_id=course_data['id'],
                                lms_platform=platform,
                                year=datetime.now().year,
                                semester=self._determine_semester(),
                                is_active=True
                            )
                            session.add(new_course)
                            session.flush()  # Get the ID
                            synced_courses.append(new_course)
                    
                    logger.info(f"Synced {len(courses_data)} courses from {platform}")
                
                except Exception as e:
                    logger.error(f"Error syncing courses from {platform}: {e}")
            
            session.commit()
        
        return synced_courses
    
    def sync_course_materials(self, course: Course) -> List[Document]:
        """Sync materials for a specific course"""
        if course.lms_platform not in self.connectors:
            logger.warning(f"No connector available for platform: {course.lms_platform}")
            return []
        
        connector = self.connectors[course.lms_platform]
        synced_documents = []
        
        with db_manager.get_session() as session:
            try:
                materials_data = connector.get_course_materials(course.lms_course_id)
                
                for material in materials_data:
                    # Check if document already exists
                    existing_doc = session.query(Document).filter(
                        Document.lms_document_id == material['id'],
                        Document.course_id == course.id
                    ).first()
                    
                    if existing_doc:
                        # Check if document was modified
                        if material['modified_date'] > existing_doc.lms_last_modified:
                            # Re-download and reprocess
                            file_path = connector.download_document(material)
                            if file_path:
                                existing_doc.file_path = file_path
                                existing_doc.lms_last_modified = material['modified_date']
                                existing_doc.is_processed = False
                                existing_doc.processing_status = "pending"
                                synced_documents.append(existing_doc)
                    else:
                        # Download new document
                        file_path = connector.download_document(material)
                        if file_path:
                            new_document = Document(
                                course_id=course.id,
                                title=material['title'],
                                file_path=file_path,
                                file_type=material['file_type'],
                                file_size=material['file_size'],
                                lms_document_id=material['id'],
                                lms_last_modified=material['modified_date'],
                                is_processed=False,
                                processing_status="pending"
                            )
                            session.add(new_document)
                            session.flush()
                            synced_documents.append(new_document)
                
                logger.info(f"Synced {len(synced_documents)} documents for course: {course.course_name}")
                session.commit()
            
            except Exception as e:
                logger.error(f"Error syncing materials for course {course.course_name}: {e}")
                session.rollback()
        
        return synced_documents
    
    def sync_all_materials(self) -> Dict[str, int]:
        """Sync materials for all courses"""
        stats = {"courses_synced": 0, "documents_synced": 0}

        # First sync courses from server-wide connectors
        courses = self.sync_courses()
        stats["courses_synced"] = len(courses)

        # Then sync materials for each course
        for course in courses:
            documents = self.sync_course_materials(course)
            stats["documents_synced"] += len(documents)

            # Add small delay to avoid overwhelming the API
            time.sleep(1)

        # Additionally sync per-user Google Classroom if enabled
        if getattr(settings, 'PER_USER_GOOGLE_OAUTH', True):
            per_user_stats = self.sync_per_user_google_classroom()
            stats["courses_synced"] += per_user_stats["courses_synced"]
            stats["documents_synced"] += per_user_stats["documents_synced"]

        logger.info(f"Sync completed - {stats}")
        return stats

    def sync_per_user_google_classroom(self) -> Dict[str, int]:
        """Sync Google Classroom for each connected user"""
        stats = {"courses_synced": 0, "documents_synced": 0}

        try:
            from src.data.models import User, CourseEnrollment

            with db_manager.get_session() as session:
                # Get all users with Google Classroom connected
                connected_users = session.query(User).filter(
                    User.google_classroom_connected == True
                ).all()

                logger.info(f"Found {len(connected_users)} users with Google Classroom connected")

                for user in connected_users:
                    try:
                        # Create connector for this user
                        connector = UserGoogleClassroomConnector(user.id, shared_oauth_manager)

                        if not connector.authenticate():
                            logger.warning(f"Failed to authenticate user {user.id}")
                            continue

                        # Get user's courses
                        google_courses = connector.get_user_courses()

                        for course_data in google_courses:
                            # Check if course already exists
                            existing_course = session.query(Course).filter(
                                Course.lms_course_id == course_data['id'],
                                Course.lms_platform == 'google_classroom'
                            ).first()

                            course = None
                            if existing_course:
                                # Update existing course
                                existing_course.course_name = course_data['name']
                                existing_course.updated_at = datetime.now()
                                course = existing_course
                            else:
                                # Create new course
                                course = Course(
                                    course_code=course_data.get('section', course_data['name'][:20]),
                                    course_name=course_data['name'],
                                    description=course_data.get('descriptionHeading', ''),
                                    lms_course_id=course_data['id'],
                                    lms_platform='google_classroom',
                                    year=datetime.now().year,
                                    semester=self._determine_semester(),
                                    is_active=True
                                )
                                session.add(course)
                                session.flush()
                                stats["courses_synced"] += 1

                            # Ensure user is enrolled in this course
                            enrollment = session.query(CourseEnrollment).filter(
                                CourseEnrollment.user_id == user.id,
                                CourseEnrollment.course_id == course.id
                            ).first()

                            if not enrollment:
                                enrollment = CourseEnrollment(
                                    user_id=user.id,
                                    course_id=course.id,
                                    enrollment_date=datetime.now(),
                                    is_active=True
                                )
                                session.add(enrollment)

                            # Sync course materials (scheduled sync - don't re-notify)
                            documents = self._sync_google_classroom_materials(
                                connector, course, session, include_unnotified=False
                            )
                            stats["documents_synced"] += len(documents)

                        session.commit()
                        logger.info(f"Synced {len(google_courses)} courses for user {user.id}")

                        # Small delay between users
                        time.sleep(1)

                    except Exception as e:
                        logger.error(f"Error syncing Google Classroom for user {user.id}: {e}")
                        session.rollback()
                        continue

                logger.info(f"Per-user Google Classroom sync completed: {stats}")

        except Exception as e:
            logger.error(f"Error in per-user Google Classroom sync: {e}")

        return stats

    def _sync_google_classroom_materials(self, connector: UserGoogleClassroomConnector,
                                       course: Course, session, include_unnotified: bool = True) -> List[Document]:
        """
        Sync materials for a Google Classroom course

        Args:
            connector: Google Classroom API connector
            course: Course to sync
            session: Database session
            include_unnotified: If True, include existing documents that haven't been notified to this user yet
        """
        from src.data.models import UserNotification

        synced_documents = []
        user_id = connector.user_id  # Get the user ID from connector

        try:
            # PART 1: Get course work (assignments, quizzes, questions, materials posted as coursework)
            coursework = connector.service.courses().courseWork().list(
                courseId=course.lms_course_id
            ).execute()

            for work in coursework.get('courseWork', []):
                # Determine material type from Google Classroom API
                work_type = work.get('workType', 'MATERIAL')  # ASSIGNMENT, SHORT_ANSWER_QUESTION, MULTIPLE_CHOICE_QUESTION, MATERIAL
                work_title = work.get('title', 'Untitled')
                work_description = work.get('description', '')

                logger.debug(f"Processing courseWork: '{work_title}' with workType='{work_type}'")

                # Map Google Classroom workType to our material_type
                if work_type == 'ASSIGNMENT':
                    # Check if it's a quiz based on title or description
                    title_lower = work_title.lower()
                    description_lower = work_description.lower()

                    # Keywords that indicate this is a quiz/test
                    quiz_keywords = ['quiz', 'test', 'exam']

                    if any(keyword in title_lower for keyword in quiz_keywords) or \
                       any(keyword in description_lower for keyword in quiz_keywords):
                        material_type = 'quiz'
                        logger.debug(f"Detected QUIZ assignment: '{work_title}'")
                    else:
                        material_type = 'assignment'
                        logger.debug(f"Detected regular assignment: '{work_title}'")
                elif work_type in ['SHORT_ANSWER_QUESTION', 'MULTIPLE_CHOICE_QUESTION']:
                    material_type = 'question'
                    logger.debug(f"Detected QUESTION: '{work_title}' ({work_type})")
                else:  # MATERIAL
                    material_type = 'material'
                    logger.debug(f"Detected MATERIAL: '{work_title}'")
                due_date = work.get('dueDate')  # {year, month, day}
                due_time = work.get('dueTime')  # {hours, minutes}
                submission_required = work_type in ['ASSIGNMENT', 'SHORT_ANSWER_QUESTION', 'MULTIPLE_CHOICE_QUESTION']

                # Parse due date if exists
                due_datetime = None
                if due_date:
                    try:
                        due_datetime = datetime(
                            year=due_date.get('year'),
                            month=due_date.get('month'),
                            day=due_date.get('day'),
                            hour=due_time.get('hours', 23) if due_time else 23,
                            minute=due_time.get('minutes', 59) if due_time else 59
                        )
                    except Exception as e:
                        logger.warning(f"Failed to parse due date for {work_title}: {e}")

                # Extract questions for question-type materials
                questions_data = None
                if material_type == 'question':
                    if work_type == 'SHORT_ANSWER_QUESTION':
                        questions_data = {
                            'type': 'short_answer',
                            'question': work_description or work_title
                        }
                    elif work_type == 'MULTIPLE_CHOICE_QUESTION':
                        choices = work.get('multipleChoiceQuestion', {}).get('choices', [])
                        questions_data = {
                            'type': 'multiple_choice',
                            'question': work_description or work_title,
                            'choices': choices
                        }

                # Process attachments
                for material in work.get('materials', []):
                    if 'driveFile' in material:
                        drive_file = material['driveFile']['driveFile']

                        # Check if it's a supported file type
                        filename = drive_file.get('title') or drive_file.get('name', 'Unknown')
                        if not filename or filename == 'Unknown':
                            logger.warning(f"Skipping file with no title in course {course.course_name}")
                            continue

                        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''

                        if file_ext in settings.SUPPORTED_FILE_TYPES:
                            # Check if document already exists FOR THIS USER
                            # IMPORTANT: Include user_id to prevent duplicate documents across users
                            existing_doc = session.query(Document).filter(
                                Document.lms_document_id == drive_file['id'],
                                Document.course_id == course.id,
                                Document.user_id == user_id  # FIX: Added user_id check for proper deduplication
                            ).first()

                            if not existing_doc:
                                # Download new document
                                try:
                                    local_path = self._download_google_drive_file(
                                        connector, drive_file, course, filename
                                    )

                                    if local_path:
                                        new_document = Document(
                                            user_id=user_id, # FIX: Added user_id for data isolation
                                            course_id=course.id,
                                            title=filename,
                                            file_path=local_path,
                                            file_type=file_ext,
                                            file_size=0,  # Size would need separate API call
                                            lms_document_id=drive_file['id'],
                                            lms_last_modified=datetime.now(),
                                            is_processed=False,
                                            processing_status="pending",
                                            material_type=material_type,
                                            submission_required=submission_required,
                                            due_date=due_datetime,
                                            questions=questions_data
                                        )
                                        session.add(new_document)
                                        session.flush()
                                        synced_documents.append(new_document)
                                        logger.info(f"Added new document: {filename}")

                                except Exception as e:
                                    logger.error(f"Failed to download document {filename}: {e}")
                                    continue
                            elif include_unnotified:
                                # Gmail-triggered sync: check if this user has been notified for this document
                                user_notification = session.query(UserNotification).filter(
                                    UserNotification.user_id == user_id,
                                    UserNotification.document_id == existing_doc.id
                                ).first()

                                if not user_notification or not user_notification.notification_sent:
                                    # User hasn't been notified yet - re-queue the document
                                    synced_documents.append(existing_doc)
                                    logger.info(f"Re-queuing existing un-notified document for user {user_id}: {filename}")
                        else:
                            logger.debug(f"Skipping unsupported file type '{file_ext}': {filename}")

            # ALSO sync announcements (where teachers post materials)
            try:
                announcements = connector.service.courses().announcements().list(
                    courseId=course.lms_course_id
                ).execute()

                for announcement in announcements.get('announcements', []):
                    announcement_id = announcement.get('id')
                    announcement_text = announcement.get('text', '')
                    announcement_title = announcement_text[:100] if announcement_text else 'Announcement'  # First 100 chars as title

                    # Check if we already have this announcement text stored
                    has_attachments = bool(announcement.get('materials'))

                    # If announcement has no attachments, create a text-only document
                    if not has_attachments and announcement_text:
                        existing_announcement = session.query(Document).filter(
                            Document.lms_document_id == announcement_id,
                            Document.course_id == course.id
                        ).first()

                        if not existing_announcement:
                            # Create text file for announcement
                            announcement_dir = Path("./data/documents") / course.course_code / "announcements"
                            announcement_dir.mkdir(parents=True, exist_ok=True)

                            # Sanitize filename
                            safe_title = "".join(c for c in announcement_title if c.isalnum() or c in (' ', '-', '_')).strip()
                            text_filename = f"{safe_title[:50]}.txt"
                            text_path = announcement_dir / text_filename

                            # Write announcement text to file
                            with open(text_path, 'w', encoding='utf-8') as f:
                                f.write(announcement_text)

                            new_announcement_doc = Document(
                                course_id=course.id,
                                title=announcement_title,
                                file_path=str(text_path),
                                file_type='txt',
                                file_size=len(announcement_text.encode('utf-8')),
                                lms_document_id=announcement_id,
                                lms_last_modified=datetime.now(),
                                is_processed=False,
                                processing_status="pending",
                                material_type='announcement',
                                content_text=announcement_text  # Store text directly
                            )
                            session.add(new_announcement_doc)
                            session.flush()
                            synced_documents.append(new_announcement_doc)
                            logger.info(f"Added new text-only announcement: {announcement_title}")

                        elif include_unnotified:
                            # Check if this user has been notified for this announcement
                            user_notification = session.query(UserNotification).filter(
                                UserNotification.user_id == user_id,
                                UserNotification.document_id == existing_announcement.id
                            ).first()

                            if not user_notification or not user_notification.notification_sent:
                                # User hasn't been notified yet - re-queue the announcement
                                synced_documents.append(existing_announcement)
                                logger.info(f"Re-queuing existing un-notified announcement for user {user_id}: {announcement_title}")

                    # Process attachments in announcements
                    for material in announcement.get('materials', []):
                        if 'driveFile' in material:
                            drive_file = material['driveFile']['driveFile']

                            # Check if it's a supported file type
                            filename = drive_file.get('title') or drive_file.get('name', 'Unknown')
                            if not filename or filename == 'Unknown':
                                logger.warning(f"Skipping announcement file with no title in course {course.course_name}")
                                continue

                            file_ext = filename.split('.')[-1].lower() if '.' in filename else ''

                            if file_ext in settings.SUPPORTED_FILE_TYPES:
                                # Check if document already exists FOR THIS USER
                                # IMPORTANT: Include user_id to prevent duplicate documents across users
                                existing_doc = session.query(Document).filter(
                                    Document.lms_document_id == drive_file['id'],
                                    Document.course_id == course.id,
                                    Document.user_id == user_id  # FIX: Added user_id check for proper deduplication
                                ).first()

                                if not existing_doc:
                                    # Download new document
                                    try:
                                        local_path = self._download_google_drive_file(
                                            connector, drive_file, course, filename
                                        )

                                        if local_path:
                                            new_document = Document(
                                                user_id=user_id, # FIX: Added user_id for data isolation
                                                course_id=course.id,
                                                title=filename,
                                                file_path=local_path,
                                                file_type=file_ext,
                                                file_size=0,
                                                lms_document_id=drive_file['id'],
                                                lms_last_modified=datetime.now(),
                                                is_processed=False,
                                                processing_status="pending",
                                                material_type='announcement'  # Mark as announcement
                                            )
                                            session.add(new_document)
                                            session.flush()
                                            synced_documents.append(new_document)
                                            logger.info(f"Added new announcement document: {filename}")

                                    except Exception as e:
                                        logger.error(f"Failed to download announcement document {filename}: {e}")
                                        continue
                                elif include_unnotified:
                                    # Gmail-triggered sync: check if this user has been notified
                                    user_notification = session.query(UserNotification).filter(
                                        UserNotification.user_id == user_id,
                                        UserNotification.document_id == existing_doc.id
                                    ).first()

                                    if not user_notification or not user_notification.notification_sent:
                                        # User hasn't been notified yet - re-queue
                                        synced_documents.append(existing_doc)
                                        logger.info(f"Re-queuing existing un-notified announcement for user {user_id}: {filename}")

            except Exception as announce_error:
                logger.error(f"Error syncing announcements for course {course.course_name}: {announce_error}")

            # PART 3: Sync courseMaterials (materials posted directly, NOT as coursework/assignment)
            # Note: These are materials teachers post directly to "Classwork" without making them assignments
            try:
                course_materials = connector.service.courses().courseWorkMaterials().list(
                    courseId=course.lms_course_id
                ).execute()

                material_count = len(course_materials.get('courseWorkMaterial', []))
                logger.info(f"Found {material_count} courseMaterials in {course.course_name}")

                for material_item in course_materials.get('courseWorkMaterial', []):
                    material_id = material_item.get('id')
                    material_title = material_item.get('title', 'Untitled Material')
                    material_description = material_item.get('description', '')

                    logger.debug(f"Processing courseMaterial: '{material_title}'")

                    # Process attachments in course materials
                    for material in material_item.get('materials', []):
                        if 'driveFile' in material:
                            drive_file = material['driveFile']['driveFile']

                            # Check if it's a supported file type
                            filename = drive_file.get('title') or drive_file.get('name', 'Unknown')
                            if not filename or filename == 'Unknown':
                                logger.warning(f"Skipping course material file with no title in course {course.course_name}")
                                continue

                            file_ext = filename.split('.')[-1].lower() if '.' in filename else ''

                            if file_ext in settings.SUPPORTED_FILE_TYPES:
                                # Check if document already exists FOR THIS USER
                                # IMPORTANT: Include user_id to prevent duplicate documents across users
                                existing_doc = session.query(Document).filter(
                                    Document.lms_document_id == drive_file['id'],
                                    Document.course_id == course.id,
                                    Document.user_id == user_id  # FIX: Added user_id check for proper deduplication
                                ).first()

                                if not existing_doc:
                                    # Download new document
                                    try:
                                        local_path = self._download_google_drive_file(
                                            connector, drive_file, course, filename
                                        )

                                        if local_path:
                                            new_document = Document(
                                                user_id=user_id, # FIX: Added user_id for data isolation
                                                course_id=course.id,
                                                title=filename,
                                                file_path=local_path,
                                                file_type=file_ext,
                                                file_size=0,
                                                lms_document_id=drive_file['id'],
                                                lms_last_modified=datetime.now(),
                                                is_processed=False,
                                                processing_status="pending",
                                                material_type='material'  # Materials are always type 'material'
                                            )
                                            session.add(new_document)
                                            session.flush()
                                            synced_documents.append(new_document)
                                            logger.info(f"Added new course material: {filename}")

                                    except Exception as e:
                                        logger.error(f"Failed to download course material {filename}: {e}")
                                        continue
                                elif include_unnotified:
                                    # Gmail-triggered sync: check if this user has been notified
                                    user_notification = session.query(UserNotification).filter(
                                        UserNotification.user_id == user_id,
                                        UserNotification.document_id == existing_doc.id
                                    ).first()

                                    if not user_notification or not user_notification.notification_sent:
                                        # User hasn't been notified yet - re-queue
                                        synced_documents.append(existing_doc)
                                        logger.info(f"Re-queuing existing un-notified course material for user {user_id}: {filename}")
                            else:
                                logger.debug(f"Skipping unsupported file type '{file_ext}' in course material: {filename}")

            except Exception as course_material_error:
                logger.error(f"Error syncing course materials for course {course.course_name}: {course_material_error}")

        except Exception as e:
            logger.error(f"Error syncing materials for course {course.course_name}: {e}")

        return synced_documents

    def _download_google_drive_file(self, connector: UserGoogleClassroomConnector,
                                   drive_file: Dict, course: Course, filename: str) -> Optional[str]:
        """Download a file from Google Drive with size limit check"""
        try:
            file_id = drive_file['id']

            # Check file size before downloading (get metadata)
            file_metadata = connector.drive_service.files().get(
                fileId=file_id,
                fields='size,name'
            ).execute()

            file_size_bytes = int(file_metadata.get('size', 0))
            file_size_mb = file_size_bytes / (1024 * 1024)

            # Enforce file size limit (default: 25MB from settings)
            max_size_mb = settings.MAX_FILE_SIZE_MB
            if file_size_mb > max_size_mb:
                logger.warning(
                    f"Skipping file {filename}: {file_size_mb:.1f}MB exceeds limit of {max_size_mb}MB. "
                    f"Large textbooks can cause connection timeouts."
                )
                return None

            # Create local directory
            local_dir = Path("./data/documents")
            local_dir.mkdir(parents=True, exist_ok=True)

            # Create safe filename
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
            local_path = local_dir / f"{course.lms_course_id}_{safe_filename}"

            # Download file using the Drive service
            request = connector.drive_service.files().get_media(fileId=file_id)
            content = request.execute()

            with open(local_path, 'wb') as f:
                f.write(content)

            logger.info(f"Downloaded Google Drive file: {local_path} ({file_size_mb:.1f}MB)")
            return str(local_path)

        except Exception as e:
            logger.error(f"Error downloading Google Drive file {filename}: {e}")
            return None

    def _determine_semester(self) -> str:
        """Determine current semester based on date"""
        month = datetime.now().month
        year = datetime.now().year
        
        if 1 <= month <= 4:
            return f"{year}.1"  # First semester
        elif 5 <= month <= 8:
            return f"{year}.2"  # Second semester
        else:
            return f"{year}.3"  # Third semester
    
    def get_available_platforms(self) -> List[str]:
        """Get list of successfully connected platforms"""
        return list(self.connectors.keys())
    
    def is_platform_connected(self, platform: str) -> bool:
        """Check if a platform is connected and authenticated"""
        return platform in self.connectors

    def sync_specific_user_course(self, user_id: int, course_name_hint: str = None,
                                  material_type: str = None) -> Dict[str, Any]:
        """
        Sync a specific course for a user (triggered by Gmail notification)

        Args:
            user_id: User database ID
            course_name_hint: Course name extracted from email (optional)
            material_type: Type of material (assignment, quiz, reading, announcement)

        Returns:
            {
                'success': bool,
                'new_documents': List[Document],
                'course_name': str
            }
        """
        stats = {
            'success': False,
            'new_documents': [],
            'course_name': None
        }

        try:
            from src.data.models import CourseEnrollment

            # Get user's Google Classroom connector
            connector = UserGoogleClassroomConnector(user_id, shared_oauth_manager)

            if not connector.authenticate():
                logger.error(f"Failed to authenticate user {user_id} for triggered sync")
                return stats

            with db_manager.get_session() as session:
                # Get user's enrolled courses
                enrollments = session.query(CourseEnrollment).filter(
                    CourseEnrollment.user_id == user_id,
                    CourseEnrollment.is_active == True
                ).join(Course).filter(
                    Course.lms_platform == 'google_classroom'
                ).all()

                if not enrollments:
                    logger.warning(f"No Google Classroom courses found for user {user_id}")
                    return stats

                # Try to match course by name hint if provided
                target_courses = []
                if course_name_hint:
                    for enrollment in enrollments:
                        course = enrollment.course
                        # Fuzzy match on course name
                        if (course_name_hint.lower() in course.course_name.lower() or
                            course.course_name.lower() in course_name_hint.lower()):
                            target_courses.append(course)
                            logger.debug(f"Matched course '{course.course_name}' with hint '{course_name_hint}'")

                    if not target_courses:
                        logger.warning(f"Could not match course hint '{course_name_hint}' for user {user_id}")
                        # Fall back to syncing all courses
                        target_courses = [e.course for e in enrollments]
                else:
                    # No hint provided - sync all courses
                    logger.info(f"No course hint provided - syncing all {len(enrollments)} courses")
                    target_courses = [e.course for e in enrollments]

                logger.info(f"Triggering sync for {len(target_courses)} course(s) for user {user_id}")

                # Sync materials for matched courses
                all_new_docs = []
                for course in target_courses:
                    try:
                        logger.debug(f"Syncing course: {course.course_name}")
                        new_docs = self._sync_google_classroom_materials(
                            connector, course, session
                        )

                        # Set material type if provided
                        if material_type and new_docs:
                            for doc in new_docs:
                                if not doc.material_type:
                                    doc.material_type = material_type

                        all_new_docs.extend(new_docs)

                        if new_docs:
                            logger.info(f"Found {len(new_docs)} new documents in {course.course_name}")
                            stats['course_name'] = course.course_name
                        else:
                            logger.debug(f"No new documents in {course.course_name}")

                    except Exception as e:
                        logger.error(f"Error syncing course {course.course_name}: {e}")
                        continue

                session.commit()

                # Extract document IDs before session closes (objects will be detached)
                doc_ids = [doc.id for doc in all_new_docs]

                stats['success'] = True
                stats['new_documents'] = all_new_docs
                stats['new_document_ids'] = doc_ids

                logger.info(f"Triggered sync completed: {len(all_new_docs)} new documents")

        except Exception as e:
            logger.error(f"Error in triggered sync for user {user_id}: {e}")

        return stats

# Global service instance
lms_service = LMSIntegrationService()