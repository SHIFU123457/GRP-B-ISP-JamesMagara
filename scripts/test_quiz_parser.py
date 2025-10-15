"""
Test script to debug quiz question parsing
Run this to verify the parser works before deploying
"""

import re
import sys
from typing import List, Dict, Any

# Fix unicode output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Sample LLM responses from actual logs
SAMPLE_RESPONSE_1 = """Here are five multiple-choice questions based on the provided course materials:

1. What is the primary purpose of corporate governance?
A. To increase shareholder value by minimizing costs
B. To provide a framework for companies to achieve their objectives in the best interest of shareholders and other stakeholders
C. To reduce the number of audit engagements for audit firms
D. To increase the independence of audit firms
CORRECT: B. To provide a framework for companies to achieve their objectives in the best interest of shareholders and other stakeholders
Explanation: As stated on Slide 4 of the pptx file, "The system by which companies are directed and controlled in the best interest of shareholders and other stakeholders."

2. According to the Cadbury report 1992, what is the primary relationship between a company's directors, its shareholders, and other stakeholders?
A. Directors and shareholders
B. Directors and other stakeholders
C. Shareholders and directors
D. Shareholders, directors and other stakeholders
CORRECT: D
EXPLANATION: The Cadbury report emphasizes the relationship between all three parties."""

SAMPLE_RESPONSE_2 = """I can generate the quiz questions based on the provided course materials. Here are five multiple-choice questions: 1. According to Chapter 10, Cipher Techniques, what is the primary purpose of precomputing the possible messages in the procedure "Precomputing the Possible Messages"?
A. To identify the length of the message
B. To determine the encoding method
C. To encode a message using a specific technique
D. To decode a message using a specific method
CORRECT: C. To encode a message using a specific technique
EXPLANATION: The primary purpose of precomputing the possible messages is to identify the encoding method, which is a key step in the cipher technique process."""

# NEW: Single-line format from actual failing log
SAMPLE_RESPONSE_3 = """I can provide five multiple-choice questions based on the course materials. Here they are: 1. What is the range for a signed byte in hexadecimal notation? A. 0-255 B. 0-100 C. 0-200 D. 0-300 Correct answer: A. 0-255 Explanation: According to the chapter on hexadecimal notation, a signed byte is represented in hexadecimal notation as 0-255. 2. What is the decimal value of the binary number 00001012? A. 8 B. 16 C. 32 D. 64 Correct answer: B. 16 Explanation: The binary number 00001012 can be converted to decimal by evaluating the powers of 2."""


def parse_quiz_questions(llm_response: str) -> List[Dict[str, Any]]:
    """Parse LLM response into structured question objects"""
    questions = []

    try:
        # Strategy 1: Standard format with "QUESTION N:"
        question_blocks = re.split(r'(?:QUESTION|Question)\s*\d+\s*:', llm_response, flags=re.IGNORECASE)[1:]

        if not question_blocks or len(question_blocks) < 2:
            # Strategy 2: Normalize and extract numbered questions
            # First normalize: ensure questions start on new lines
            normalized = re.sub(r'(:|\.)(\s*)(\d+\.)', r'\1\n\3', llm_response)

            # Extract question blocks by finding "N. " followed by content until next "N. " or end
            question_pattern = r'(\d+\.\s+.+?(?=\d+\.\s+|$))'
            potential_blocks = re.findall(question_pattern, normalized, re.DOTALL)

            # Filter blocks that look like actual questions (have options A, B, C, D)
            question_blocks = []
            for block in potential_blocks:
                has_options = all(re.search(rf'{letter}[\.\)]', block, re.IGNORECASE) for letter in ['A', 'B', 'C', 'D'])
                has_correct = re.search(r'CORRECT', block, re.IGNORECASE)

                if has_options and has_correct:
                    question_blocks.append(block.strip())

            print(f"✓ Strategy 2: Found {len(question_blocks)} blocks")

        if not question_blocks:
            print("✗ Could not split LLM response into question blocks")
            return []

        print(f"✓ Successfully split into {len(question_blocks)} question blocks\n")

        # Parse each block
        for block_idx, block in enumerate(question_blocks):
            try:
                # CRITICAL FIX: LLM often puts everything on one line
                # Add newlines before options (A., B., C., D.) and keywords (CORRECT, EXPLANATION)
                block = re.sub(r'\s+([A-D][\.\)])', r'\n\1', block)
                block = re.sub(r'\s+(CORRECT|Correct answer):', r'\n\1:', block, flags=re.IGNORECASE)
                block = re.sub(r'\s+(EXPLANATION|Explanation):', r'\n\1:', block, flags=re.IGNORECASE)

                lines = [line.strip() for line in block.strip().split('\n') if line.strip()]

                print(f"--- Parsing Block {block_idx + 1} ---")
                print(f"Total lines: {len(lines)}")

                if len(lines) < 5:
                    print(f"✗ Too few lines ({len(lines)})")
                    print(f"Block preview: {block[:200]}\n")
                    continue

                # Extract question text
                question_text = lines[0]
                question_text = re.sub(r'^\d+\.\s*', '', question_text)
                print(f"Question: {question_text[:80]}...")

                # Extract options
                options = []
                option_pattern = re.compile(r'^([A-Da-d])[\)\.\:]\s*(.+)', re.IGNORECASE)

                for line in lines[1:]:
                    match = option_pattern.match(line)
                    if match:
                        options.append(match.group(2).strip())
                    if len(options) == 4:
                        break

                print(f"Options found: {len(options)}")
                if len(options) != 4:
                    print(f"✗ Expected 4 options, found {len(options)}")
                    for i, line in enumerate(lines[:10]):
                        print(f"  Line {i}: {line[:80]}")
                    print()
                    continue

                # Extract correct answer
                block_text = '\n'.join(lines)
                correct_line = None
                correct_patterns = [
                    r'CORRECT\s*:?\s*([A-D])',
                    r'Correct\s+answer\s*:?\s*([A-D])',
                    r'Answer\s*:?\s*([A-D])',
                    r'EXPLANATION\s*:?\s*CORRECT\s*:?\s*([A-D])',
                ]

                for pattern in correct_patterns:
                    match = re.search(pattern, block_text, re.IGNORECASE)
                    if match:
                        correct_line = match.group(1).upper()
                        break

                if not correct_line:
                    print(f"✗ No correct answer found")
                    print(f"Block text: {block_text[:300]}\n")
                    correct_line = 'A'

                correct_index = ord(correct_line) - ord('A') if correct_line in 'ABCD' else 0
                print(f"Correct answer: {correct_line} (index {correct_index})")

                # Extract explanation
                explanation = "No explanation provided"
                explanation_patterns = [
                    r'EXPLANATION\s*:?\s*(.+)',
                    r'Explanation\s*:?\s*(.+)',
                ]

                for pattern in explanation_patterns:
                    match = re.search(pattern, block_text, re.IGNORECASE)
                    if match:
                        explanation = match.group(1).strip()[:200]
                        break

                print(f"Explanation: {explanation[:80]}...")

                questions.append({
                    'question': question_text,
                    'options': options,
                    'correct_answer_index': correct_index,
                    'explanation': explanation
                })

                print(f"✓ Successfully parsed question {len(questions)}\n")

            except Exception as parse_error:
                print(f"✗ Error parsing block {block_idx}: {parse_error}\n")
                continue

    except Exception as e:
        print(f"✗ Error parsing quiz questions: {e}")

    return questions


def test_parsing():
    """Test the parsing with sample responses"""
    print("="*80)
    print("TESTING QUIZ QUESTION PARSER")
    print("="*80)

    test_cases = [
        ("Sample Response 1 (Corporate Governance)", SAMPLE_RESPONSE_1),
        ("Sample Response 2 (Inline format)", SAMPLE_RESPONSE_2),
        ("Sample Response 3 (Single-line from logs)", SAMPLE_RESPONSE_3),
    ]

    for name, response in test_cases:
        print(f"\n{'='*80}")
        print(f"Test Case: {name}")
        print(f"{'='*80}\n")

        questions = parse_quiz_questions(response)

        print(f"\n{'='*40}")
        print(f"RESULT: {'✓ PASS' if questions else '✗ FAIL'} - Generated {len(questions)} questions")
        print(f"{'='*40}\n")


if __name__ == "__main__":
    test_parsing()
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
