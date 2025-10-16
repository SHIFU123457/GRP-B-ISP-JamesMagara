"""
Test Script for Personalization Engine and Session Management

This script tests:
1. Session creation and timeout
2. Session activity tracking
3. Personalization profile updates
4. Interaction recording
5. Analytics calculation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import time
from datetime import datetime, timedelta

from config.database import db_manager
from src.data.models import User, ConversationSession, UserInteraction, PersonalizationProfile
from src.services.personalization_engine import personalization_engine, session_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_user():
    """Create a test user for testing"""
    try:
        with db_manager.get_session() as session:
            # Check if test user exists
            test_user = session.query(User).filter(
                User.telegram_id == "test_user_12345"
            ).first()

            if test_user:
                logger.info(f"Test user already exists: ID {test_user.id}")
                return test_user.id

            # Create new test user
            test_user = User(
                telegram_id="test_user_12345",
                username="test_user",
                first_name="Test",
                last_name="User",
                learning_style="adaptive",
                difficulty_preference="medium"
            )
            session.add(test_user)
            session.commit()

            logger.info(f"✅ Created test user with ID: {test_user.id}")
            return test_user.id

    except Exception as e:
        logger.error(f"❌ Error creating test user: {e}")
        return None


def test_session_creation(user_id):
    """Test session creation and retrieval"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Session Creation")
    logger.info("=" * 60)

    try:
        with db_manager.get_session() as session:
            # Create first session
            conv_session = session_manager.get_or_create_session(user_id, session)

            assert conv_session is not None, "Session should not be None"
            assert conv_session.user_id == user_id, "User ID should match"
            assert conv_session.is_active == True, "Session should be active"
            assert conv_session.message_count == 0, "Initial message count should be 0"

            session_id_1 = conv_session.session_id
            logger.info(f"✅ Created session: {session_id_1}")

            # Get same session again (should return existing)
            conv_session_2 = session_manager.get_or_create_session(user_id, session)

            assert conv_session_2.session_id == session_id_1, "Should return same session"
            logger.info(f"✅ Retrieved same session: {conv_session_2.session_id}")

        logger.info("✅ TEST 1 PASSED: Session creation works correctly")
        return True

    except AssertionError as e:
        logger.error(f"❌ TEST 1 FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ TEST 1 ERROR: {e}", exc_info=True)
        return False


def test_session_activity_tracking(user_id):
    """Test session activity tracking"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Session Activity Tracking")
    logger.info("=" * 60)

    try:
        with db_manager.get_session() as session:
            conv_session = session_manager.get_or_create_session(user_id, session)

            initial_count = conv_session.message_count

            # Update activity multiple times
            for i in range(3):
                session_manager.update_session_activity(
                    conv_session,
                    session,
                    interaction_type="question" if i % 2 == 0 else "command",
                    context_updates={'test_key': f'value_{i}'}
                )

            session.refresh(conv_session)

            assert conv_session.message_count == initial_count + 3, "Message count should increase by 3"
            assert conv_session.questions_asked >= 2, "Should have at least 2 questions"
            assert 'test_key' in conv_session.session_context, "Context should be updated"

            logger.info(f"✅ Message count: {conv_session.message_count}")
            logger.info(f"✅ Questions asked: {conv_session.questions_asked}")
            logger.info(f"✅ Commands used: {conv_session.commands_used}")
            logger.info(f"✅ Context: {conv_session.session_context}")

        logger.info("✅ TEST 2 PASSED: Activity tracking works correctly")
        return True

    except AssertionError as e:
        logger.error(f"❌ TEST 2 FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ TEST 2 ERROR: {e}", exc_info=True)
        return False


def test_session_timeout(user_id):
    """Test session timeout functionality"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Session Timeout")
    logger.info("=" * 60)

    try:
        with db_manager.get_session() as session:
            # Get current active session
            conv_session = session_manager.get_or_create_session(user_id, session)
            session_id_1 = conv_session.session_id

            # Manually set last_activity_at to 31 minutes ago (past timeout)
            old_time = datetime.utcnow() - timedelta(minutes=31)
            conv_session.last_activity_at = old_time
            session.commit()

            logger.info(f"Set session last activity to {old_time}")

            # Try to get session again - should create new one due to timeout
            new_session = session_manager.get_or_create_session(user_id, session)

            assert new_session.session_id != session_id_1, "Should create new session after timeout"
            assert new_session.is_active == True, "New session should be active"

            # Check old session is closed
            old_session = session.query(ConversationSession).filter(
                ConversationSession.session_id == session_id_1
            ).first()

            assert old_session.is_active == False, "Old session should be inactive"
            assert old_session.ended_at is not None, "Old session should have end time"

            logger.info(f"✅ Old session {session_id_1} closed")
            logger.info(f"✅ New session {new_session.session_id} created")

        logger.info("✅ TEST 3 PASSED: Session timeout works correctly")
        return True

    except AssertionError as e:
        logger.error(f"❌ TEST 3 FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ TEST 3 ERROR: {e}", exc_info=True)
        return False


def test_interaction_recording(user_id):
    """Test interaction recording"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Interaction Recording")
    logger.info("=" * 60)

    try:
        # Record some test interactions
        test_queries = [
            ("What is Python?", "Python is a programming language..."),
            ("Explain data structures", "Data structures are ways to organize data..."),
            ("How does sorting work?", "Sorting algorithms arrange elements...")
        ]

        for query, response in test_queries:
            success = personalization_engine.record_interaction(
                user_id=user_id,
                query=query,
                response=response,
                interaction_type="question",
                course_context="Computer Science",
                response_time_ms=1500
            )
            assert success, f"Failed to record interaction: {query}"

        # Verify interactions were recorded
        with db_manager.get_session() as session:
            interaction_count = session.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).count()

            assert interaction_count >= 3, f"Should have at least 3 interactions, got {interaction_count}"

            logger.info(f"✅ Recorded {interaction_count} interactions")

        logger.info("✅ TEST 4 PASSED: Interaction recording works correctly")
        return True

    except AssertionError as e:
        logger.error(f"❌ TEST 4 FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ TEST 4 ERROR: {e}", exc_info=True)
        return False


def test_personalization_profile_update(user_id):
    """Test personalization profile updates"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Personalization Profile Update")
    logger.info("=" * 60)

    try:
        # Update profile
        profile = personalization_engine.update_personalization_profile(user_id)

        assert profile is not None, "Profile should not be None"

        # Read profile data within a new session to avoid DetachedInstanceError
        with db_manager.get_session() as session:
            profile_check = session.query(PersonalizationProfile).filter(
                PersonalizationProfile.user_id == user_id
            ).first()

            assert profile_check is not None, "Profile should exist in database"
            assert profile_check.user_id == user_id, "User ID should match"
            assert profile_check.total_interactions > 0, "Should have interactions"

            logger.info(f"✅ Total interactions: {profile_check.total_interactions}")
            logger.info(f"✅ Complexity level: {profile_check.question_complexity_level}")
            logger.info(f"✅ Learning pace: {profile_check.learning_pace}")
            logger.info(f"✅ Active hours: {profile_check.most_active_hours}")
            logger.info(f"✅ Preferred subjects: {profile_check.preferred_subjects}")

        # Get personalized settings
        settings = personalization_engine.get_personalized_settings(user_id)

        assert settings is not None, "Settings should not be None"
        assert 'learning_style' in settings, "Settings should have learning_style"
        assert 'difficulty_preference' in settings, "Settings should have difficulty_preference"

        logger.info(f"✅ Personalized settings: {settings}")

        logger.info("✅ TEST 5 PASSED: Profile updates work correctly")
        return True

    except AssertionError as e:
        logger.error(f"❌ TEST 5 FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ TEST 5 ERROR: {e}", exc_info=True)
        return False


def test_analytics_calculation(user_id):
    """Test analytics calculation"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Analytics Calculation")
    logger.info("=" * 60)

    try:
        with db_manager.get_session() as session:
            analytics = personalization_engine.analyze_user_interactions(user_id, session)

            assert analytics is not None, "Analytics should not be None"
            assert 'total_interactions' in analytics, "Should have total_interactions"
            assert 'preferred_topics' in analytics, "Should have preferred_topics"
            assert 'question_complexity' in analytics, "Should have question_complexity"

            logger.info(f"✅ Analytics computed:")
            logger.info(f"   - Total interactions: {analytics['total_interactions']}")
            logger.info(f"   - Avg response time: {analytics['avg_response_time_ms']}ms")
            logger.info(f"   - Question complexity: {analytics['question_complexity']}")
            logger.info(f"   - Preferred topics: {analytics['preferred_topics']}")
            logger.info(f"   - Interaction types: {analytics['interaction_types']}")

        logger.info("✅ TEST 6 PASSED: Analytics calculation works correctly")
        return True

    except AssertionError as e:
        logger.error(f"❌ TEST 6 FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ TEST 6 ERROR: {e}", exc_info=True)
        return False


def cleanup_test_data(user_id):
    """Clean up test data"""
    logger.info("\n" + "=" * 60)
    logger.info("CLEANUP: Removing test data")
    logger.info("=" * 60)

    try:
        with db_manager.get_session() as session:
            # Delete interactions
            session.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).delete()

            # Delete sessions
            session.query(ConversationSession).filter(
                ConversationSession.user_id == user_id
            ).delete()

            # Delete profile
            session.query(PersonalizationProfile).filter(
                PersonalizationProfile.user_id == user_id
            ).delete()

            # Delete user
            session.query(User).filter(User.id == user_id).delete()

            session.commit()

            logger.info("✅ Cleaned up test data")

    except Exception as e:
        logger.error(f"⚠️  Error during cleanup: {e}")


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("PERSONALIZATION ENGINE TEST SUITE")
    logger.info("=" * 60)

    # Create test user
    user_id = create_test_user()
    if not user_id:
        logger.error("❌ Failed to create test user. Exiting.")
        return False

    # Run tests
    results = {
        "Session Creation": test_session_creation(user_id),
        "Activity Tracking": test_session_activity_tracking(user_id),
        "Session Timeout": test_session_timeout(user_id),
        "Interaction Recording": test_interaction_recording(user_id),
        "Profile Update": test_personalization_profile_update(user_id),
        "Analytics Calculation": test_analytics_calculation(user_id)
    }

    # Cleanup
    cleanup_test_data(user_id)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info("=" * 60)
    logger.info(f"TOTAL: {passed}/{total} tests passed")
    logger.info("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
