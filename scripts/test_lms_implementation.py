#!/usr/bin/env python3
"""
Setup and test script for LMS integration
Run this to configure and test your LMS connections
"""

import sys
import os
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.database import db_manager
from src.services.lms_integration import lms_service
from config.settings import settings
from src.services.scheduler import scheduler_service

def test_database_connection():
    """Test database connection"""
    print("🔄 Testing database connection...")
    
    try:
        with db_manager.get_session() as session:
            # Simple query to test connection
            result = session.execute("SELECT 1").fetchone()
            if result:
                print("✅ Database connection successful")
                return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_moodle_connection():
    """Test Moodle connection"""
    print("\n🔄 Testing Moodle connection...")
    
    moodle_url = os.getenv("MOODLE_BASE_URL")
    moodle_token = os.getenv("MOODLE_API_TOKEN")
    
    if not moodle_url or not moodle_token:
        print("⚠️ Moodle credentials not configured")
        return False
    
    try:
        from src.services.lms_integration import MoodleConnector
        connector = MoodleConnector(moodle_url, moodle_token)
        
        if connector.authenticate():
            print("✅ Moodle authentication successful")
            
            # Test getting courses
            courses = connector.get_courses()
            print(f"📚 Found {len(courses)} Moodle courses")
            
            if courses:
                print("Sample courses:")
                for course in courses[:3]:  # Show first 3
                    print(f"  • {course['short_name']}: {course['name']}")
            
            return True
        else:
            print("❌ Moodle authentication failed")
            return False
            
    except Exception as e:
        print(f"❌ Moodle connection error: {e}")
        return False

def test_google_classroom_connection():
    """Test Google Classroom connection"""
    print("\n🔄 Testing Google Classroom connection...")
    
    if getattr(settings, 'PER_USER_GOOGLE_OAUTH', True):
        print("ℹ️ Per-user Google OAuth is enabled; skipping server-side Google Classroom authentication test.")
        return True
    
    credentials_path = os.getenv("GOOGLE_CLASSROOM_CREDENTIALS")
    
    if not credentials_path or not Path(credentials_path).exists():
        print("⚠️ Google Classroom credentials not found")
        return False
    
    try:
        from src.services.lms_integration import GoogleClassroomConnector
        connector = GoogleClassroomConnector(credentials_path)
        
        if connector.authenticate():
            print("✅ Google Classroom authentication successful")
            
            # Test getting courses
            courses = connector.get_courses()
            print(f"📚 Found {len(courses)} Google Classroom courses")
            
            if courses:
                print("Sample courses:")
                for course in courses[:3]:  # Show first 3
                    print(f"  • {course['short_name']}: {course['name']}")
            
            return True
        else:
            print("❌ Google Classroom authentication failed")
            return False
            
    except Exception as e:
        print(f"❌ Google Classroom connection error: {e}")
        return False

def test_lms_service():
    """Test the main LMS service"""
    print("\n🔄 Testing LMS Integration Service...")
    
    try:
        platforms = lms_service.get_available_platforms()
        
        if platforms:
            print(f"✅ LMS Service initialized with platforms: {platforms}")
            
            # Test syncing courses
            print("🔄 Testing course synchronization...")
            courses = lms_service.sync_courses()
            print(f"📚 Synchronized {len(courses)} courses")
            
            if courses:
                print("Synchronized courses:")
                for course in courses[:5]:  # Show first 5
                    print(f"  • {course.course_code}: {course.course_name} ({course.lms_platform})")
                
                # Test syncing materials for the first course
                if courses:
                    print(f"\n🔄 Testing material sync for: {courses[0].course_name}")
                    materials = lms_service.sync_course_materials(courses[0])
                    print(f"📄 Found {len(materials)} materials")
                    
                    if materials:
                        print("Sample materials:")
                        for material in materials[:3]:
                            print(f"  • {material.title} ({material.file_type})")
            
            return True
        else:
            print("❌ No LMS platforms available")
            return False
            
    except Exception as e:
        print(f"❌ LMS Service error: {e}")
        return False

def test_scheduler_service():
    """Test the scheduler service"""
    print("\n🔄 Testing Scheduler Service...")
    
    try:
        # Get status without starting the scheduler
        status = scheduler_service.get_sync_status()
        
        print("📊 Scheduler Status:")
        print(f"  • Connected platforms: {status.get('connected_platforms', [])}")
        print(f"  • Total documents: {status.get('documents', {}).get('total', 0)}")
        print(f"  • Processed documents: {status.get('documents', {}).get('processed', 0)}")
        print(f"  • Pending documents: {status.get('documents', {}).get('pending', 0)}")
        print(f"  • Total courses: {status.get('courses', {}).get('total', 0)}")
        print(f"  • RAG available: {status.get('rag_available', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scheduler Service error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Study Helper Agent - LMS Integration Setup & Test")
    print("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n📋 Running connection tests...\n")
    
    # Track test results
    results = {
        'database': test_database_connection(),
        'moodle': test_moodle_connection(),
        'google_classroom': test_google_classroom_connection(),
        'lms_service': False,
        'scheduler': False
    }
    
    # Only test higher-level services if basic connections work
    if results['database'] and (results['moodle'] or results['google_classroom']):
        results['lms_service'] = test_lms_service()
        results['scheduler'] = test_scheduler_service()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test.replace('_', ' ').title()}: {status}")
    
    # Provide recommendations
    print("\n💡 Recommendations:")
    
    if not results['database']:
        print("• Fix database connection before proceeding")
        print("• Check your DATABASE_URL in the .env file")
    
    if not results['moodle'] and not results['google_classroom']:
        print("• Configure at least one LMS platform")
        print("• Add MOODLE_BASE_URL and MOODLE_API_TOKEN for Moodle")
        print("• Add GOOGLE_CLASSROOM_CREDENTIALS for Google Classroom")
    
    if results['database'] and (results['moodle'] or results['google_classroom']):
        if results['lms_service'] and results['scheduler']:
            print("• 🎉 All tests passed! Your LMS integration is ready")
            print("• Run 'python src/main.py' to start the bot")
        else:
            print("• Basic connections work, but service integration needs attention")
    
    print("\n📚 Next Steps:")
    print("2. Run this test script again to verify connections")
    print("3. Start the bot with 'python src/main.py'")
    print("4. Use /sync command in Telegram to test live integration")

if __name__ == "__main__":
    main()