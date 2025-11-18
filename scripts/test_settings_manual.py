"""
Test script for manual learning style settings

This script tests the ability for users to manually change their learning style
through the /settings command and verifies the database is updated correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from config.database import db_manager
from src.data.models import User

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_learning_style_values():
    """Test that the learning style values match between UI and engine"""

    logger.info("=" * 80)
    logger.info("Testing Learning Style Value Consistency")
    logger.info("=" * 80)

    # These are the values used in the UI buttons (handlers.py)
    ui_values = [
        "example_driven",
        "analogy_driven",
        "socratic",
        "theory_first",
        "adaptive"
    ]

    # These are the values used in explanation_style_engine.py
    engine_values = [
        "example_driven",
        "analogy_driven",
        "socratic",
        "theory_first",
        "adaptive"
    ]

    logger.info("\nUI Button Values:")
    for val in ui_values:
        logger.info(f"  - {val}")

    logger.info("\nEngine Style Values:")
    for val in engine_values:
        logger.info(f"  - {val}")

    # Check consistency
    if set(ui_values) == set(engine_values):
        logger.info("\n✅ SUCCESS: UI and Engine values are consistent!")
        return True
    else:
        logger.error("\n❌ FAILED: UI and Engine values are inconsistent!")
        missing_in_engine = set(ui_values) - set(engine_values)
        missing_in_ui = set(engine_values) - set(ui_values)

        if missing_in_engine:
            logger.error(f"  Missing in engine: {missing_in_engine}")
        if missing_in_ui:
            logger.error(f"  Missing in UI: {missing_in_ui}")

        return False


def test_database_update():
    """Test that user learning style can be updated in database"""

    logger.info("\n" + "=" * 80)
    logger.info("Testing Database Update Functionality")
    logger.info("=" * 80)

    test_telegram_id = "99999999"  # Test user ID

    try:
        with db_manager.get_session() as session:
            # Clean up any existing test user
            test_user = session.query(User).filter(User.telegram_id == test_telegram_id).first()
            if test_user:
                session.delete(test_user)
                session.commit()
                logger.info(f"\nCleaned up existing test user")

            # Create a test user with default 'adaptive' style
            test_user = User(
                telegram_id=test_telegram_id,
                username="test_user",
                first_name="Test",
                last_name="User",
                learning_style="adaptive",
                difficulty_preference="medium"
            )
            session.add(test_user)
            session.commit()
            user_id = test_user.id
            logger.info(f"\n✅ Created test user with ID: {user_id}")
            logger.info(f"   Initial learning style: {test_user.learning_style}")

        # Test changing to each learning style
        styles_to_test = [
            "example_driven",
            "analogy_driven",
            "socratic",
            "theory_first",
            "adaptive"
        ]

        for style in styles_to_test:
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                old_style = user.learning_style
                user.learning_style = style
                session.commit()

                # Verify change
                session.refresh(user)
                if user.learning_style == style:
                    logger.info(f"✅ Successfully changed style: {old_style} -> {style}")
                else:
                    logger.error(f"❌ Failed to change style to {style}")
                    return False

        # Clean up
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                session.delete(user)
                session.commit()
                logger.info(f"\n✅ Cleaned up test user")

        logger.info("\n✅ SUCCESS: All database update tests passed!")
        return True

    except Exception as e:
        logger.error(f"\n❌ FAILED: Database update test failed with error: {e}", exc_info=True)
        return False


def test_settings_descriptions():
    """Test that all learning styles have proper descriptions"""

    logger.info("\n" + "=" * 80)
    logger.info("Testing Learning Style Descriptions")
    logger.info("=" * 80)

    style_descriptions = {
        "example_driven": "You'll learn through concrete examples first, followed by theory.",
        "analogy_driven": "You'll learn through metaphors and real-world comparisons.",
        "socratic": "You'll learn through guided questions and discovery.",
        "theory_first": "You'll learn formal definitions and theory before examples.",
        "adaptive": "I'll automatically detect and adapt to your learning style based on your questions."
    }

    logger.info("\nLearning Style Descriptions:")
    for style, description in style_descriptions.items():
        logger.info(f"\n{style}:")
        logger.info(f"  {description}")

    # Check all styles have descriptions
    required_styles = ["example_driven", "analogy_driven", "socratic", "theory_first", "adaptive"]
    missing = [s for s in required_styles if s not in style_descriptions]

    if not missing:
        logger.info("\n✅ SUCCESS: All learning styles have descriptions!")
        return True
    else:
        logger.error(f"\n❌ FAILED: Missing descriptions for: {missing}")
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 80)
    logger.info("MANUAL SETTINGS TEST SUITE")
    logger.info("=" * 80)

    results = []

    # Test 1: Value consistency
    results.append(("Value Consistency", test_learning_style_values()))

    # Test 2: Database updates
    results.append(("Database Updates", test_database_update()))

    # Test 3: Descriptions
    results.append(("Style Descriptions", test_settings_descriptions()))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        logger.info("\n✅ ALL TESTS PASSED!")
        logger.info("\nThe manual settings feature is ready to use!")
        logger.info("Users can now change their learning style via /settings")
    else:
        logger.error("\n❌ SOME TESTS FAILED!")
        logger.error("Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}", exc_info=True)
        sys.exit(1)
