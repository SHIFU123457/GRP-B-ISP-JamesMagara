"""
Test script for conversation-aware quiz generation

This script simulates the scenario where a user has a conversation about
Lasso/Ridge regression, then asks "quiz me on what we discussed".

The quiz should generate questions about Lasso/Ridge regression from the
conversation history, NOT random topics from course documents.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from config.database import db_manager
from src.core.rag_pipeline import RAGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_conversation_quiz_generation():
    """Test quiz generation from conversation history"""

    logger.info("=" * 80)
    logger.info("Testing Conversation-Aware Quiz Generation")
    logger.info("=" * 80)

    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()

    # Simulate a conversation about Lasso/Ridge regression
    conversation_history = [
        {
            'query': 'What is Lasso regression?',
            'response': 'Lasso (Least Absolute Shrinkage and Selection Operator) regression is a type of linear regression that uses L1 regularization. It adds a penalty term equal to the absolute value of the magnitude of coefficients. This penalty has the effect of forcing some coefficient values to be exactly zero when the penalty is sufficiently large, which means Lasso can be used for feature selection. The objective function is: minimize(sum of squared residuals + lambda * sum of absolute values of coefficients).'
        },
        {
            'query': 'How does Ridge regression differ from Lasso?',
            'response': 'Ridge regression uses L2 regularization instead of L1. The key differences are: 1) Ridge adds a penalty equal to the SQUARE of the magnitude of coefficients, not the absolute value. 2) Ridge shrinks coefficients toward zero but never exactly to zero, while Lasso can make coefficients exactly zero. 3) Ridge is better when all features are relevant, while Lasso is better for feature selection. 4) The Ridge objective is: minimize(sum of squared residuals + lambda * sum of squared coefficients).'
        },
        {
            'query': 'When should I use Lasso vs Ridge?',
            'response': 'Use Lasso when: you suspect many features are irrelevant and want automatic feature selection, you need a sparse model for interpretability, you have high-dimensional data with many irrelevant features. Use Ridge when: all features are potentially relevant, you want to keep all features but shrink their coefficients, you need more stable predictions with correlated features. For best results, you can also use Elastic Net which combines both L1 and L2 penalties.'
        }
    ]

    # Test 1: Quiz generation WITH conversation history
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Quiz generation WITH conversation history")
    logger.info("=" * 80)

    topic = "what we discussed"  # Conversational reference
    user_id = 1  # Test user

    logger.info(f"Topic: '{topic}'")
    logger.info(f"Conversation history: {len(conversation_history)} turns")
    logger.info("\nGenerating quiz questions from conversation...")

    questions_with_history = rag_pipeline.generate_quiz_questions(
        user_id=user_id,
        topic=topic,
        num_questions=3,
        conversation_history=conversation_history
    )

    logger.info(f"\n✅ Generated {len(questions_with_history)} questions WITH conversation history:\n")

    for i, q in enumerate(questions_with_history, 1):
        logger.info(f"\nQuestion {i}: {q['question']}")
        for j, opt in enumerate(q['options']):
            marker = "✓" if j == q['correct_answer_index'] else " "
            logger.info(f"  [{marker}] {chr(65+j)}) {opt}")
        logger.info(f"  Explanation: {q['explanation']}")

    # Test 2: Quiz generation WITHOUT conversation history (for comparison)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Quiz generation WITHOUT conversation history (baseline)")
    logger.info("=" * 80)

    logger.info(f"Topic: 'regression' (without conversation context)")
    logger.info("\nGenerating quiz questions from document search...")

    questions_without_history = rag_pipeline.generate_quiz_questions(
        user_id=user_id,
        topic="regression",
        num_questions=3,
        conversation_history=None  # No conversation history
    )

    logger.info(f"\n✅ Generated {len(questions_without_history)} questions WITHOUT conversation history:\n")

    for i, q in enumerate(questions_without_history, 1):
        logger.info(f"\nQuestion {i}: {q['question']}")
        for j, opt in enumerate(q['options']):
            marker = "✓" if j == q['correct_answer_index'] else " "
            logger.info(f"  [{marker}] {chr(65+j)}) {opt}")
        logger.info(f"  Explanation: {q['explanation']}")

    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS")
    logger.info("=" * 80)

    logger.info("\nTest 1 (WITH conversation history):")
    logger.info("  - Should generate questions about Lasso/Ridge regression")
    logger.info("  - Questions should be based on conversation content")
    logger.info("  - Should NOT include random topics like speed cameras or health sites")

    logger.info("\nTest 2 (WITHOUT conversation history):")
    logger.info("  - May generate questions from various course documents about regression")
    logger.info("  - Topics may vary based on document content and similarity scores")
    logger.info("  - May include unrelated topics if similarity threshold is low")

    # Check if questions are about the right topic
    if questions_with_history:
        keywords = ['lasso', 'ridge', 'regularization', 'l1', 'l2', 'coefficient', 'penalty', 'feature selection']

        relevant_count = 0
        for q in questions_with_history:
            question_text = q['question'].lower()
            if any(kw in question_text for kw in keywords):
                relevant_count += 1

        relevance_ratio = relevant_count / len(questions_with_history)

        logger.info(f"\n✅ Relevance Check: {relevant_count}/{len(questions_with_history)} questions mention Lasso/Ridge keywords")
        logger.info(f"   Relevance ratio: {relevance_ratio*100:.1f}%")

        if relevance_ratio >= 0.5:
            logger.info("\n✅ SUCCESS: Questions are relevant to the conversation topic!")
        else:
            logger.warning("\n⚠️ WARNING: Questions may not be fully relevant to conversation topic")
    else:
        logger.error("\n❌ FAILED: No questions generated with conversation history")

    logger.info("\n" + "=" * 80)
    logger.info("Test Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        test_conversation_quiz_generation()
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
