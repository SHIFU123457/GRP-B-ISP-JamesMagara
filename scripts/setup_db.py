#script to setup the database with sample data for testing
#This script has already been run and the database is setup with sample data
#Thus, it is for reference only
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add src directory to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from config.database import db_manager
from config.settings import settings
from src.data.models import User, Course, Document, CourseEnrollment, PersonalizationProfile

def create_sample_data():
    """Create sample data for testing"""
    print("Creating sample data...")
    
    with db_manager.get_session() as session:
        # Check if courses already exist
        existing_courses = session.query(Course).count()
        if existing_courses > 0:
            print(f"Found {existing_courses} existing courses. Skipping sample data creation.")
            return
        
        # Create sample courses
        courses = [
            Course(
                course_code="ICS201",
                course_name="Data Structures and Algorithms",
                description="Introduction to data structures, algorithms, and their analysis",
                semester="1.2",
                year=2025,
                is_public=False,
                lms_platform="moodle"
            ),
            Course(
                course_code="ICS301", 
                course_name="Software Engineering",
                description="Software development methodologies, design patterns, and project management",
                semester="3.1",
                year=2025,
                is_public=False,
                lms_platform="moodle"
            ),
            Course(
                course_code="MAT201",
                course_name="Discrete Mathematics", 
                description="Mathematical foundations for computer science",
                semester="1.1",
                year=2025,
                is_public=False,
                lms_platform="google_classroom"
            )
        ]
        
        for course in courses:
            session.add(course)
        session.flush()  # Get course IDs
        
        # Create a sample user
        user = User(
            telegram_id="123456789",
            username="testuser",
            first_name="Test",
            last_name="User",
            email="test@example.com"
        )
        session.add(user)
        session.flush() # Get user ID

        # Enroll user in courses
        for course in courses:
            enrollment = CourseEnrollment(user_id=user.id, course_id=course.id)
            session.add(enrollment)
        
        # Create sample documents
        documents = [
            Document(
                user_id=user.id, # Assign to the new user
                course_id=courses[0].id,
                title="Week 1 - Introduction to Data Structures",
                file_path="./data/mock_lms/courses/ics201/week1_intro.pdf",
                file_type="pdf",
                content_text="Introduction to arrays, linked lists, and basic operations...",
                is_processed=True,
                processing_status="completed"
            ),
            Document(
                user_id=user.id, # Assign to the new user
                course_id=courses[0].id,
                title="Week 2 - Stacks and Queues",
                file_path="./data/mock_lms/courses/ics201/week2_stacks_queues.pdf", 
                file_type="pdf",
                content_text="Stack operations: push, pop, peek. Queue operations: enqueue, dequeue...",
                is_processed=True,
                processing_status="completed"
            ),
            Document(
                user_id=user.id, # Assign to the new user
                course_id=courses[1].id,
                title="Software Development Life Cycle",
                file_path="./data/mock_lms/courses/ics301/sdlc_overview.pdf",
                file_type="pdf", 
                content_text="Overview of waterfall, agile, and other SDLC methodologies...",
                is_processed=True,
                processing_status="completed"
            )
        ]
        
        for doc in documents:
            session.add(doc)
        
        session.commit()
        print(f"Created {len(courses)} courses and {len(documents)} documents")

def setup_database():
    """Setup database with tables and sample data"""
    print("Setting up Study Helper Agent database...")
    print(f"Database URL: {settings.DATABASE_URL}")
    
    try:
        # Create tables
        print("Creating database tables...")
        db_manager.create_tables()
        print("✅ Database tables created successfully")
        
        # Migrate schema (add missing columns and create new tables)
        print("Migrating database schema...")
        db_manager.migrate_schema()
        print("✅ Database schema migrated successfully")
        
        # Create sample data
        create_sample_data()
        print("✅ Sample data created successfully")
        
        print("\n🎉 Database setup completed!")
        print("\nYou can now:")
        print("1. Create your .env file based on .env.example")
        print("2. Add your Telegram bot token to .env")
        print("3. Run: python src/main.py")
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_database()