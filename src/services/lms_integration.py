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

                            # Sync course materials
                            documents = self._sync_google_classroom_materials(
                                connector, course, session
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
                                       course: Course, session) -> List[Document]:
        """Sync materials for a Google Classroom course"""
        synced_documents = []

        try:
            # Get course work and materials
            coursework = connector.service.courses().courseWork().list(
                courseId=course.lms_course_id
            ).execute()

            for work in coursework.get('courseWork', []):
                # Process attachments
                for material in work.get('materials', []):
                    if 'driveFile' in material:
                        drive_file = material['driveFile']['driveFile']

                        # Check if it's a supported file type
                        filename = drive_file['title']
                        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''

                        if file_ext in settings.SUPPORTED_FILE_TYPES:
                            # Check if document already exists
                            existing_doc = session.query(Document).filter(
                                Document.lms_document_id == drive_file['id'],
                                Document.course_id == course.id
                            ).first()

                            if not existing_doc:
                                # Download new document
                                try:
                                    local_path = self._download_google_drive_file(
                                        connector, drive_file, course, filename
                                    )

                                    if local_path:
                                        new_document = Document(
                                            course_id=course.id,
                                            title=filename,
                                            file_path=local_path,
                                            file_type=file_ext,
                                            file_size=0,  # Size would need separate API call
                                            lms_document_id=drive_file['id'],
                                            lms_last_modified=datetime.now(),
                                            is_processed=False,
                                            processing_status="pending"
                                        )
                                        session.add(new_document)
                                        session.flush()
                                        synced_documents.append(new_document)
                                        logger.info(f"Added new document: {filename}")

                                except Exception as e:
                                    logger.error(f"Failed to download document {filename}: {e}")
                                    continue

        except Exception as e:
            logger.error(f"Error syncing materials for course {course.course_name}: {e}")

        return synced_documents

    def _download_google_drive_file(self, connector: UserGoogleClassroomConnector,
                                   drive_file: Dict, course: Course, filename: str) -> Optional[str]:
        """Download a file from Google Drive"""
        try:
            file_id = drive_file['id']

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

            logger.info(f"Downloaded Google Drive file: {local_path}")
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

# Global service instance
lms_service = LMSIntegrationService()