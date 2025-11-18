"""
Test script for question extraction regex patterns.

This script tests the regex patterns used in _complete_assignment_with_questions()
to ensure they correctly extract questions from exam documents.
"""

import re

# Sample exam text similar to the one the user uploaded
SAMPLE_EXAM_TEXT = """
MAKERERE UNIVERSITY
FACULTY OF COMPUTING AND INFORMATICS TECHNOLOGY
BACHELOR OF SCIENCE IN COMPUTER SCIENCE
ICS 4102 - Machine learning
August 2020

INSTRUCTIONS: Answer ALL questions. Time allowed: 2 hours.

Question ONE (COMPULSORY)

a) Define the following terms as used in machine learning:
   i) Supervised learning
   ii) Unsupervised learning
   iii) Reinforcement learning

b) Explain the difference between classification and regression problems in machine learning?

Question TWO

a) What is overfitting in machine learning? How can you prevent it?

b) Describe the bias-variance tradeoff in machine learning models.

Question THREE

a) Explain the working principle of the k-Nearest Neighbors (k-NN) algorithm?

b) What are the advantages and disadvantages of using k-NN for classification tasks?

Question FOUR

a) Describe the structure and operation of an artificial neural network?

b) What is the role of activation functions in neural networks? Give two examples.

Question FIVE

a) What is cross-validation and why is it important in machine learning?

b) Explain the difference between training data, validation data, and test data.
"""


def test_question_extraction():
    """Test the question extraction regex patterns."""

    # Patterns from the implementation
    question_patterns = [
        # "Question ONE", "Question 1", "Q1:", "Q.1"
        r'(?:Question|Q)\.?\s*(\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|COMPULSORY)[:\.\s]+([^?]+\?(?:[^\n]*\n)*?)(?=(?:Question|Q)\.?\s*(?:\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)|$)',
        # Fallback: standalone questions ending with "?"
        r'(\d+)\.\s*([^?]+\?)',
    ]

    print("=" * 80)
    print("TESTING QUESTION EXTRACTION PATTERNS")
    print("=" * 80)

    for i, pattern in enumerate(question_patterns, 1):
        print(f"\n--- Pattern {i} ---")
        print(f"Regex: {pattern[:80]}...")

        matches = re.findall(pattern, SAMPLE_EXAM_TEXT, re.DOTALL | re.IGNORECASE)

        if matches:
            print(f"\nFound {len(matches)} matches:")
            for j, (q_num, q_text) in enumerate(matches, 1):
                print(f"\n  Match {j}:")
                print(f"    Question Number: {q_num}")
                print(f"    Question Text: {q_text.strip()[:150]}...")
                print(f"    Full length: {len(q_text)} characters")

            # Use this pattern
            print(f"\n[SUCCESS] Pattern {i} successfully extracted {len(matches)} questions!")
            return matches
        else:
            print("  No matches found with this pattern.")

    print("\n[FAILURE] No patterns successfully extracted questions!")
    return []


def test_full_document_questions():
    """Test extracting questions from the full document text."""

    print("\n" + "=" * 80)
    print("TESTING FULL DOCUMENT QUESTION EXTRACTION")
    print("=" * 80)

    # This simulates what happens in _complete_assignment_with_questions
    question_patterns = [
        r'(?:Question|Q)\.?\s*(\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|COMPULSORY)[:\.\s]+([^?]+\?(?:[^\n]*\n)*?)(?=(?:Question|Q)\.?\s*(?:\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)|$)',
        r'(\d+)\.\s*([^?]+\?)',
    ]

    questions = []
    for pattern in question_patterns:
        matches = re.findall(pattern, SAMPLE_EXAM_TEXT, re.DOTALL | re.IGNORECASE)
        if matches:
            questions.extend(matches)
            break  # Use first successful pattern

    if questions:
        print(f"\n[SUCCESS] Successfully extracted {len(questions)} questions:")
        for i, (q_num, q_text) in enumerate(questions, 1):
            print(f"\n{i}. Question {q_num}")
            print(f"   Text: {q_text.strip()[:200]}...")
    else:
        print("\n[FAILURE] No questions extracted!")

    return questions


def test_edge_cases():
    """Test edge cases and different question formats."""

    print("\n" + "=" * 80)
    print("TESTING EDGE CASES")
    print("=" * 80)

    edge_cases = {
        "Numbered questions": """
Question 1
a) What is machine learning?
b) Describe supervised learning.

Question 2
a) What is overfitting?
b) How do you prevent it?
""",
        "Q notation": """
Q1: Define machine learning?
Q2: What is the difference between classification and regression?
""",
        "Mixed format": """
Question ONE (COMPULSORY)
a) What is neural network?

Question 2
a) Explain backpropagation?
""",
        "Sub-questions": """
Question 1
a) What is deep learning?
b) How does it differ from machine learning?
c) Give three examples of deep learning applications?
"""
    }

    pattern = r'(?:Question|Q)\.?\s*(\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|COMPULSORY)[:\.\s]+([^?]+\?(?:[^\n]*\n)*?)(?=(?:Question|Q)\.?\s*(?:\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)|$)'

    for case_name, case_text in edge_cases.items():
        print(f"\n--- {case_name} ---")
        matches = re.findall(pattern, case_text, re.DOTALL | re.IGNORECASE)
        print(f"Extracted {len(matches)} questions:")
        for q_num, q_text in matches:
            print(f"  - Question {q_num}: {q_text.strip()[:80]}...")


if __name__ == "__main__":
    # Run tests
    print("\nStarting question extraction tests...\n")

    # Test 1: Pattern matching
    questions = test_question_extraction()

    # Test 2: Full document simulation
    test_full_document_questions()

    # Test 3: Edge cases
    test_edge_cases()

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
