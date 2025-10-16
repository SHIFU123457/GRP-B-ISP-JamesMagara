"""
Test script for Explanation Style Engine

Tests the explanation style customization feature (Phase 2):
- Learning style classification
- Query pattern matching
- Template generation
- Style-based prompt customization
- Integration with adaptive response engine

Run: python scripts/test_explanation_style_engine.py
"""

import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.database import db_manager
from src.data.models import User, PersonalizationProfile, UserInteraction
from src.services.explanation_style_engine import (
    LearningStyleClassifier,
    StyleBasedPromptGenerator,
    explanation_style_engine
)
from src.services.adaptive_response_engine import adaptive_response_engine


def print_test_header(test_name: str):
    """Print formatted test header"""
    print("\n" + "=" * 70)
    print(f"TEST: {test_name}")
    print("=" * 70)


def print_success(message: str):
    """Print success message"""
    print(f"[PASS] {message}")


def print_info(message: str):
    """Print info message"""
    print(f"[INFO] {message}")


def test_query_pattern_matching():
    """Test 1: Query pattern matching for all learning styles"""
    print_test_header("Query Pattern Matching")

    classifier = LearningStyleClassifier()

    # Test example-driven patterns
    test_cases = [
        # Example-driven queries
        ("Can you show me an example of recursion?", "example_driven", "example"),
        ("Give me a code example for binary search", "example_driven", "example"),
        ("Demonstrate how merge sort works", "example_driven", "demonstrate"),

        # Analogy-driven queries
        ("What is recursion like in real life?", "analogy_driven", "like"),
        ("Can you use an analogy to explain stacks?", "analogy_driven", "analogy"),
        ("Compare linked lists to something familiar", "analogy_driven", "compare"),

        # Socratic queries
        ("Why does quicksort have O(n log n) complexity?", "socratic", "why"),
        ("What if I use recursion instead of iteration?", "socratic", "what if"),
        ("How would you approach solving this problem?", "socratic", "how would"),

        # Theory-first queries
        ("Define what a binary tree is", "theory_first", "define"),
        ("What's the formal definition of Big O notation?", "theory_first", "formal"),
        ("Explain the principle behind dynamic programming", "theory_first", "principle"),
    ]

    passed = 0
    failed = 0

    for query, expected_style, keyword in test_cases:
        result = classifier.analyze_query_style(query)

        # Check if expected style has highest score
        max_style = max(result, key=result.get)

        if max_style == expected_style and result[expected_style] > 0:
            print_success(f"'{query[:50]}...' -> {expected_style} (score: {result[expected_style]:.2f})")
            passed += 1
        else:
            print(f"[FAIL] '{query[:50]}...' -> Expected {expected_style}, got {max_style}")
            print(f"   Scores: {result}")
            failed += 1

    print(f"\n[RESULT] Pattern Matching: {passed}/{len(test_cases)} passed")
    return failed == 0


def test_style_classification_with_history():
    """Test 2: Style classification based on interaction history"""
    print_test_header("Style Classification with History")

    with db_manager.get_session() as session:
        # Create test user
        test_user = User(
            telegram_id=99999998,
            username="style_test_user",
            first_name="Style",
            last_name="Tester",
            learning_style="adaptive"  # Initially adaptive
        )
        session.add(test_user)
        session.flush()

        user_id = test_user.id

        # Create personalization profile
        profile = PersonalizationProfile(
            user_id=user_id,
            total_interactions=0
        )
        session.add(profile)
        session.commit()

        # Simulate interaction history - primarily example-driven
        example_queries = [
            "Show me an example of binary search",
            "Can you demonstrate how stacks work?",
            "Give me a code example for recursion",
            "Show me how to implement a linked list",
            "Demonstrate quicksort with an example",
        ]

        analogy_queries = [
            "What's recursion like in real life?",
            "Compare arrays to something familiar",
        ]

        # Add example-driven interactions
        for i, query in enumerate(example_queries):
            interaction = UserInteraction(
                user_id=user_id,
                query_text=query,
                response_text=f"Here's an example: {query}",
                interaction_type="question",
                created_at=datetime.now() - timedelta(days=10-i)
            )
            session.add(interaction)

        # Add some analogy-driven interactions
        for i, query in enumerate(analogy_queries):
            interaction = UserInteraction(
                user_id=user_id,
                query_text=query,
                response_text=f"Analogy: {query}",
                interaction_type="question",
                created_at=datetime.now() - timedelta(days=5-i)
            )
            session.add(interaction)

        session.commit()

        # Test classification
        classifier = LearningStyleClassifier()
        classified_style, scores = classifier.classify_user_style(user_id, session, lookback_interactions=30)

        print_info(f"User ID: {user_id}")
        print_info(f"Total interactions: {len(example_queries) + len(analogy_queries)}")
        print_info(f"Style scores: {scores}")
        print_info(f"Classified style: {classified_style}")

        # Should classify as example_driven (5 queries vs 2 analogy)
        if classified_style == "example_driven":
            print_success(f"Correctly classified as example_driven (score: {scores['example_driven']:.2f})")
            success = True
        else:
            print(f"❌ FAILED: Expected example_driven, got {classified_style}")
            success = False

        # Cleanup
        session.query(UserInteraction).filter(UserInteraction.user_id == user_id).delete()
        session.query(PersonalizationProfile).filter(PersonalizationProfile.user_id == user_id).delete()
        session.query(User).filter(User.id == user_id).delete()
        session.commit()

        return success


def test_template_generation():
    """Test 3: Template generation for each learning style"""
    print_test_header("Template Generation for All Styles")

    generator = StyleBasedPromptGenerator()
    test_topic = "recursion"
    test_level = "beginner"

    styles = ["example_driven", "analogy_driven", "socratic", "theory_first"]

    all_passed = True

    for style in styles:
        template = generator.generate_prompt_template(style, test_topic, test_level)

        print_info(f"\nStyle: {style.upper()}")
        print_info(f"Template length: {len(template)} chars")
        print_info(f"First 150 chars: {template[:150]}...")

        # Validate template characteristics
        if style == "example_driven":
            if "example" in template.lower() and "demonstrate" in template.lower():
                print_success("Example-driven template: Contains 'example' and 'demonstrate'")
            else:
                print("[FAIL] Example-driven template missing key terms")
                all_passed = False

        elif style == "analogy_driven":
            if "analogy" in template.lower() or "metaphor" in template.lower() or "like" in template.lower():
                print_success("Analogy-driven template: Contains analogy/metaphor keywords")
            else:
                print("[FAIL] Analogy-driven template missing key terms")
                all_passed = False

        elif style == "socratic":
            if "question" in template.lower() or "guide" in template.lower() or "discover" in template.lower():
                print_success("Socratic template: Contains questioning/guiding keywords")
            else:
                print("[FAIL] Socratic template missing key terms")
                all_passed = False

        elif style == "theory_first":
            if "definition" in template.lower() or "formal" in template.lower() or "principle" in template.lower():
                print_success("Theory-first template: Contains formal/definition keywords")
            else:
                print("[FAIL] Theory-first template missing key terms")
                all_passed = False

    return all_passed


def test_behavioral_heuristics():
    """Test 4: Behavioral heuristics (query length, why/how ratio)"""
    print_test_header("Behavioral Heuristics")

    classifier = LearningStyleClassifier()

    with db_manager.get_session() as session:
        # Create test user
        test_user = User(
            telegram_id=99999997,
            username="heuristic_test",
            first_name="Heuristic",
            last_name="Test"
        )
        session.add(test_user)
        session.flush()
        user_id = test_user.id

        # Create profile
        profile = PersonalizationProfile(user_id=user_id, total_interactions=0)
        session.add(profile)
        session.commit()

        # Simulate Socratic learner behavior: lots of "why" and "how" questions
        socratic_queries = [
            "Why does recursion work?",
            "How does the base case prevent infinite loops?",
            "Why is merge sort better than bubble sort?",
            "How would I optimize this algorithm?",
            "Why does dynamic programming use memoization?",
            "How can I determine the time complexity?",
        ]

        for i, query in enumerate(socratic_queries):
            interaction = UserInteraction(
                user_id=user_id,
                query_text=query,
                response_text="Answer",
                interaction_type="question",
                created_at=datetime.now() - timedelta(days=6-i)
            )
            session.add(interaction)

        session.commit()

        # Classify
        classified_style, scores = classifier.classify_user_style(user_id, session)

        print_info(f"Queries: {socratic_queries}")
        print_info(f"Style scores: {scores}")
        print_info(f"Classified: {classified_style}")

        # Should have high socratic score due to why/how questions
        if scores.get('socratic', 0) > 0.3:  # Threshold for socratic behavior
            print_success(f"Behavioral heuristics detected Socratic pattern (score: {scores['socratic']:.2f})")
            success = True
        else:
            print(f"[FAIL] Socratic score too low ({scores.get('socratic', 0):.2f})")
            success = False

        # Cleanup
        session.query(UserInteraction).filter(UserInteraction.user_id == user_id).delete()
        session.query(PersonalizationProfile).filter(PersonalizationProfile.user_id == user_id).delete()
        session.query(User).filter(User.id == user_id).delete()
        session.commit()

        return success


def test_style_override():
    """Test 5: Query-specific style override"""
    print_test_header("Query-Specific Style Override")

    with db_manager.get_session() as session:
        # Create user with theory_first preference
        test_user = User(
            telegram_id=99999996,
            username="override_test",
            first_name="Override",
            last_name="Test",
            learning_style="theory_first"  # Default preference
        )
        session.add(test_user)
        session.flush()
        user_id = test_user.id

        # Create profile
        profile = PersonalizationProfile(user_id=user_id, total_interactions=0)
        session.add(profile)
        session.commit()

        # Query that STRONGLY indicates example preference (should override)
        override_query = "Can you show me a concrete example with step-by-step code demonstration?"

        # Get style for this query
        style_instructions, detected_style = explanation_style_engine.generate_style_based_instructions(
            user_id=user_id,
            query=override_query,
            session=session,
            user_level="beginner"
        )

        print_info(f"User's default style: theory_first")
        print_info(f"Query: '{override_query}'")
        print_info(f"Detected style: {detected_style}")
        print_info(f"Instructions (first 200 chars): {style_instructions[:200]}...")

        # Should override to example_driven
        if detected_style == "example_driven":
            print_success("Successfully overrode theory_first to example_driven based on query")
            success = True
        else:
            print(f"[FAIL] Expected override to example_driven, got {detected_style}")
            success = False

        # Cleanup
        session.query(PersonalizationProfile).filter(PersonalizationProfile.user_id == user_id).delete()
        session.query(User).filter(User.id == user_id).delete()
        session.commit()

        return success


def test_integration_with_adaptive_engine():
    """Test 6: Integration with adaptive response engine"""
    print_test_header("Integration with Adaptive Response Engine")

    with db_manager.get_session() as session:
        # Create test user
        test_user = User(
            telegram_id=99999995,
            username="integration_test",
            first_name="Integration",
            last_name="Test",
            learning_style="analogy_driven"
        )
        session.add(test_user)
        session.flush()
        user_id = test_user.id

        # Create profile
        profile = PersonalizationProfile(
            user_id=user_id,
            total_interactions=10,
            user_level="intermediate"
        )
        session.add(profile)
        session.commit()

        # Test query
        query = "Explain how dynamic programming works"
        base_prompt = "You are a study assistant. Explain dynamic programming."

        # Call adaptive engine (which should invoke explanation style engine)
        enhanced_prompt, metadata = adaptive_response_engine.analyze_and_enhance_prompt(
            user_id=user_id,
            query=query,
            base_prompt=base_prompt,
            session=session
        )

        print_info(f"Query: {query}")
        print_info(f"User's learning style: analogy_driven")
        print_info(f"Metadata keys: {list(metadata.keys())}")
        print_info(f"Explanation style in metadata: {metadata.get('explanation_style')}")
        print_info(f"Enhanced prompt length: {len(enhanced_prompt)} chars")

        # Check that explanation_style is in metadata
        if 'explanation_style' in metadata:
            print_success(f"Explanation style included in metadata: {metadata['explanation_style']}")

            # Check that enhanced prompt is different from base prompt
            if enhanced_prompt != base_prompt and len(enhanced_prompt) > len(base_prompt):
                print_success("Enhanced prompt differs from base prompt (style instructions added)")
                success = True
            else:
                print("[FAIL] Enhanced prompt not properly modified")
                success = False
        else:
            print("[FAIL] explanation_style not in metadata")
            success = False

        # Cleanup
        session.query(PersonalizationProfile).filter(PersonalizationProfile.user_id == user_id).delete()
        session.query(User).filter(User.id == user_id).delete()
        session.commit()

        return success


def test_style_caching():
    """Test 7: Style caching in User.learning_style"""
    print_test_header("Style Caching in User Model")

    with db_manager.get_session() as session:
        # Create user with no style preference
        test_user = User(
            telegram_id=99999994,
            username="cache_test",
            first_name="Cache",
            last_name="Test",
            learning_style="adaptive"  # Default
        )
        session.add(test_user)
        session.flush()
        user_id = test_user.id

        # Create profile
        profile = PersonalizationProfile(user_id=user_id, total_interactions=0)
        session.add(profile)
        session.commit()

        # Add theory-first interactions
        theory_queries = [
            "Define what a binary tree is",
            "What's the formal definition of polymorphism?",
            "Explain the theoretical foundation of algorithms",
        ]

        for i, query in enumerate(theory_queries):
            interaction = UserInteraction(
                user_id=user_id,
                query_text=query,
                response_text="Theory answer",
                interaction_type="question",
                created_at=datetime.now() - timedelta(days=3-i)
            )
            session.add(interaction)

        session.commit()

        # First classification (should update User.learning_style)
        classifier = LearningStyleClassifier()
        classified_style, scores = classifier.classify_user_style(user_id, session)

        # Refresh user to see updated learning_style
        session.expire(test_user)
        updated_user = session.query(User).filter(User.id == user_id).first()

        print_info(f"Initial learning_style: adaptive")
        print_info(f"After classification: {updated_user.learning_style}")
        print_info(f"Classified style: {classified_style}")

        # Check if User.learning_style was updated
        if updated_user.learning_style == classified_style:
            print_success(f"User.learning_style cached correctly: {updated_user.learning_style}")
            success = True
        else:
            print(f"[FAIL] learning_style not cached (expected {classified_style}, got {updated_user.learning_style})")
            success = False

        # Cleanup
        session.query(UserInteraction).filter(UserInteraction.user_id == user_id).delete()
        session.query(PersonalizationProfile).filter(PersonalizationProfile.user_id == user_id).delete()
        session.query(User).filter(User.id == user_id).delete()
        session.commit()

        return success


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 70)
    print("EXPLANATION STYLE ENGINE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Query Pattern Matching", test_query_pattern_matching),
        ("Style Classification with History", test_style_classification_with_history),
        ("Template Generation", test_template_generation),
        ("Behavioral Heuristics", test_behavioral_heuristics),
        ("Query-Specific Style Override", test_style_override),
        ("Integration with Adaptive Engine", test_integration_with_adaptive_engine),
        ("Style Caching in User Model", test_style_caching),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[EXCEPTION] in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASSED]" if result else "[FAILED]"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 70)
    print(f"FINAL RESULT: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Explanation Style Engine is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
