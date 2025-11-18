"""
Query Rewriting Module for Context-Aware Retrieval

This module handles:
1. Follow-up query detection (e.g., "elaborate further", "tell me more")
2. Context extraction from conversation history
3. Query reformulation for better RAG retrieval
4. Anaphora resolution (e.g., "what about it?", "tell me more")

Author: Study Helper Agent Team
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrite vague/follow-up queries using conversation context"""

    # COMPREHENSIVE context-referencing patterns (from adaptive_response_engine.py)
    CONTEXT_REFERENCE_PATTERNS = {
        'first': r'\b(first|initial|original)\s+(question|answer|response|message)',
        'previous': r'\b(previous|last|earlier|prior)\s+(question|answer|response|message)',
        'that': r'\b(that|the above|your)\s+(answer|response|explanation)',
        'correct': r'\bcorrect\s+(your|the|that)\s+(answer|response|explanation)',
        # Pronoun references without named subject: "explain it", "describe this"
        'pronoun': r'\b(explain|clarify|elaborate|tell me more about|describe)\s+(it|that|this|them)(?!\s+\w+)',
        # Followup requests: "explain further", "tell me more"
        'followup': r'\b(explain|clarify|elaborate|tell me|describe)\s+(further|more|better)',
        # Meta-understanding: "I haven't understood..."
        'meta_confusion': r'\b(haven\'t|havent|don\'t|dont|do\s+not)\s+(understood|understand|get|follow|grasp)\b',
        # Process questions: "how did you achieve/calculate X"
        'process': r'\b(how|why)\s+(did\s+you|have\s+you|you\'ve|youve)\s+(manage|achieve|calculate|get|reach)',
    }

    # Additional simple follow-up indicators
    SIMPLE_FOLLOWUP_PATTERNS = [
        # Elaboration requests
        r'^(elaborate|explain\s+more|tell\s+me\s+more|go\s+deeper|more\s+detail)',
        r'^(continue|keep\s+going|what\s+else)',

        # References to previous content
        r'^(what\s+about|how\s+about)\s+(it|that|this|them)',
        r'^(and\s+)?(what|how)\s+about',

        # Vague pronouns at start
        r'^(it|that|this|those|them)\b',

        # Request for examples (always context-dependent)
        r'(give|show|provide)\s+(me\s+)?(an?\s+)?examples?',

        # "How does that/it work" patterns
        r'how\s+(does|do|did)\s+(that|it|this|those|they)\s+work',

        # Comparative questions without subject
        r'^(compare|contrast|difference|similar)',

        # Very short follow-ups
        r'^\s*(why|how|when|where|who)\s*\??$',
        r'^\s*examples?\s*\??$',
    ]

    # Patterns that extract topics from previous queries
    TOPIC_EXTRACTION_PATTERNS = [
        # Questions
        r'what\s+(?:is|are|does|do|means?)\s+(.+?)\??$',
        r'explain\s+(.+?)(?:\s+to\s+me)?\??$',
        r'tell\s+me\s+about\s+(.+?)\??$',
        r'how\s+(?:does|do|is|are)\s+(.+?)\s+work',
        r'define\s+(.+?)\??$',
    ]

    def __init__(self):
        # Compile all patterns
        self.context_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.CONTEXT_REFERENCE_PATTERNS.items()
        }
        self.simple_followup_regex = [
            re.compile(p, re.IGNORECASE) for p in self.SIMPLE_FOLLOWUP_PATTERNS
        ]
        self.topic_regex = [
            re.compile(p, re.IGNORECASE) for p in self.TOPIC_EXTRACTION_PATTERNS
        ]
        self.logger = logging.getLogger(f"{__name__}.QueryRewriter")

    def is_followup_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if query is a follow-up that needs context

        Args:
            query: User's query text

        Returns:
            Tuple of (is_followup: bool, context_type: Optional[str])
        """
        query_clean = query.strip()
        query_lower = query_clean.lower()

        # Check comprehensive context patterns first (most specific)
        for context_type, pattern in self.context_patterns.items():
            if pattern.search(query_lower):
                self.logger.info(
                    f"Detected {context_type}-type context reference: '{query_clean[:50]}...'"
                )
                return True, context_type

        # Check simple follow-up patterns
        for pattern in self.simple_followup_regex:
            if pattern.search(query_clean):
                self.logger.info(
                    f"Detected simple follow-up query: '{query_clean[:50]}...' "
                    f"(pattern: {pattern.pattern})"
                )
                return True, 'followup'

        # Heuristic: Very short queries (1-2 words) are often follow-ups
        word_count = len(query_clean.split())
        if word_count <= 2 and not query_clean.endswith('?'):
            self.logger.info(
                f"Detected follow-up query by length: '{query_clean}' ({word_count} words)"
            )
            return True, 'short_query'

        return False, None

    def extract_topic_from_query(self, query: str) -> Optional[str]:
        """
        Extract main topic from a query using pattern matching

        Args:
            query: Previous query to extract topic from

        Returns:
            Extracted topic or None
        """
        query_clean = query.strip()

        # Try pattern matching
        for pattern in self.topic_regex:
            match = pattern.search(query_clean)
            if match:
                topic = match.group(1).strip()
                # Clean up the topic
                topic = self._clean_topic(topic)
                if topic:
                    self.logger.debug(f"Extracted topic '{topic}' from query using pattern")
                    return topic

        # Fallback: Remove common question words and use the rest
        question_words = r'\b(what|how|why|when|where|who|is|are|does|do|can|could|would|should|explain|define|tell|me|about)\b'
        topic = re.sub(question_words, '', query_clean, flags=re.IGNORECASE)
        topic = re.sub(r'[?!.,]+', '', topic).strip()
        topic = self._clean_topic(topic)

        if topic and len(topic.split()) >= 2:
            self.logger.debug(f"Extracted topic '{topic}' from query using fallback")
            return topic

        return None

    def _clean_topic(self, topic: str) -> str:
        """Clean and normalize extracted topic"""
        if not topic:
            return ""

        # Remove excessive whitespace
        topic = " ".join(topic.split())

        # Remove trailing prepositions and articles
        topic = re.sub(r'\s+(in|on|at|to|for|with|from|the|a|an)$', '', topic, flags=re.IGNORECASE)

        # Remove leading articles
        topic = re.sub(r'^(the|a|an)\s+', '', topic, flags=re.IGNORECASE)

        return topic.strip()

    def extract_topic_from_response(
        self,
        response: str,
        query: str,
        llm_service=None
    ) -> Optional[str]:
        """
        Extract topic from generated response using LLM analysis

        This provides much more accurate topic extraction by analyzing
        what the bot actually talked about in its response.

        Args:
            response: The bot's generated response
            query: The original user query
            llm_service: LLM service instance (optional, for advanced extraction)

        Returns:
            Extracted topic or None
        """
        try:
            # Strategy 1: Use LLM to extract topic from response (most accurate)
            if llm_service:
                topic = self._extract_topic_with_llm(response, query, llm_service)
                if topic and len(topic) > 2:
                    self.logger.info(f"Extracted topic from response via LLM: '{topic}'")
                    return topic

            # Strategy 2: Extract from response text using NLP (fallback)
            topic = self._extract_topic_from_text_advanced(response, query)
            if topic:
                self.logger.info(f"Extracted topic from response via NLP: '{topic}'")
                return topic

            return None

        except Exception as e:
            self.logger.error(f"Error extracting topic from response: {e}")
            return None

    def _extract_topic_with_llm(self, response: str, query: str, llm_service) -> Optional[str]:
        """
        Use LLM to extract the main topic from the response

        This is the most accurate method as the LLM understands context.
        """
        try:
            # Create a focused prompt for topic extraction
            prompt = f"""Based on this question and answer, identify the MAIN TOPIC in 2-5 words.

QUESTION: {query[:200]}

ANSWER: {response[:800]}

Extract ONLY the core topic/concept discussed. Reply with just the topic name, nothing else.
Examples: "biological database entries", "recursive algorithms", "neural networks"

TOPIC:"""

            # Use LLM to extract topic
            topic_response = llm_service.generate_response(
                query="Extract topic",
                context=prompt,
                user_preferences=None
            )

            if topic_response:
                # Clean the response
                topic = topic_response.strip().strip('"').strip("'").strip('.')

                # Remove common prefixes
                topic = re.sub(r'^(the topic is|topic:|the main topic is|main topic:)\s*', '', topic, flags=re.IGNORECASE)

                # Validate topic (should be 2-10 words)
                word_count = len(topic.split())
                if 2 <= word_count <= 10:
                    return self._clean_topic(topic)

            return None

        except Exception as e:
            self.logger.warning(f"LLM topic extraction failed: {e}")
            return None

    def _extract_topic_from_text_advanced(self, response: str, query: str) -> Optional[str]:
        """
        Extract topic from response text using advanced NLP techniques

        Fallback method when LLM is not available.

        Strategy:
        1. Look for capitalized multi-word phrases in response (e.g., "Machine Learning")
        2. Find terms that appear 2+ times in response
        3. Use enhanced keyword extraction as fallback
        4. Combine query terms for context clues
        """
        # Strategy 1: Extract capitalized multi-word phrases from RESPONSE (highest priority)
        # These are often proper nouns or technical terms
        response_snippet = response[:800]  # Use first 800 chars (usually the main content)
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', response_snippet)

        if capitalized_phrases:
            # Count frequency of multi-word phrases
            phrase_freq = Counter([phrase.lower() for phrase in capitalized_phrases])

            # Return most common multi-word phrase that appears 2+ times
            for phrase, count in phrase_freq.most_common(3):
                if count >= 2 and 2 <= len(phrase.split()) <= 5:
                    self.logger.debug(f"Found capitalized phrase: '{phrase}' (appears {count}x)")
                    return self._clean_topic(phrase)

        # Strategy 2: Look for ALL CAPS technical terms (DNA, API, SQL, etc.)
        all_caps_terms = re.findall(r'\b[A-Z]{2,}\b', response_snippet)
        if all_caps_terms:
            caps_freq = Counter(all_caps_terms)
            for term, count in caps_freq.most_common(3):
                if count >= 2 and 2 <= len(term) <= 10:
                    # ALL CAPS term mentioned multiple times is likely important
                    self.logger.debug(f"Found technical term: '{term}' (appears {count}x)")
                    return self._clean_topic(term.lower())

        # Strategy 3: Look for repeated significant words in response
        # Extract words that appear 3+ times (indicates topic emphasis)
        words_in_response = re.findall(r'\b[a-z]{4,}\b', response_snippet.lower())
        if words_in_response:
            word_freq = Counter(words_in_response)

            # Filter out stopwords
            stopwords = {
                'the', 'this', 'that', 'with', 'from', 'have', 'will', 'would',
                'their', 'there', 'these', 'those', 'which', 'about', 'other',
                'example', 'examples', 'such', 'also', 'used', 'help', 'called'
            }

            for word, count in word_freq.most_common(5):
                if count >= 3 and word not in stopwords and len(word) >= 5:
                    self.logger.debug(f"Found repeated term: '{word}' (appears {count}x)")
                    return self._clean_topic(word)

        # Strategy 4: Combine query + response context
        # Sometimes the query provides clues about the topic
        combined_text = f"{query} {response_snippet}"
        combined_capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', combined_text)

        if combined_capitalized:
            combined_freq = Counter([phrase.lower() for phrase in combined_capitalized])
            for phrase, count in combined_freq.most_common(5):
                if count >= 2 and len(phrase.split()) <= 5:
                    self.logger.debug(f"Found combined context phrase: '{phrase}' (appears {count}x)")
                    return self._clean_topic(phrase)

        # Strategy 5: Fallback to enhanced keyword extraction from response only
        topics = self._extract_topics_from_text(response_snippet, top_n=2)
        if topics:
            # Prefer topics that appear in both query and response
            query_lower = query.lower()
            for topic in topics:
                if topic in query_lower:
                    self.logger.debug(f"Found topic in both query and response: '{topic}'")
                    return topic

            # Otherwise return first extracted topic
            self.logger.debug(f"Fallback topic extraction: '{topics[0]}'")
            return topics[0]

        self.logger.debug("No topic could be extracted from response")
        return None

    def rewrite_with_context(
        self,
        current_query: str,
        conversation_history: List[Dict[str, str]],
        session_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Rewrite query using conversation history for better retrieval

        NEW STRATEGY: Use actual conversation content directly instead of topic extraction.
        This is more reliable because:
        - No dependency on NLP/LLM extraction accuracy
        - Preserves full context and nuance
        - Works even when topic extraction fails

        Args:
            current_query: User's current query
            conversation_history: List of {query, response} dicts (most recent last)
            session_context: Optional session metadata (primary_topic, etc.)

        Returns:
            Tuple of (rewritten_query, metadata)

        Example:
            Original: "elaborate further"
            History: [{"query": "what is a database?", "response": "A database is a structured..."}]
            Rewritten: "elaborate further. Recent discussion: 'A database is a structured...'"
        """
        metadata = {
            'is_followup': False,
            'original_query': current_query,
            'context_source': None,
            'context_snippet': None
        }

        # Check if this is a follow-up query
        is_followup, context_type = self.is_followup_query(current_query)

        if not is_followup:
            self.logger.debug(f"Query '{current_query}' is not a follow-up, no rewriting needed")
            return current_query, metadata

        metadata['is_followup'] = True
        metadata['context_type'] = context_type

        # Strategy 1: Use recent conversation snippets directly (MOST RELIABLE)
        if conversation_history and len(conversation_history) > 0:
            # Get last 1-2 turns for context
            recent_turns = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history[-1:]

            context_snippets = []
            for turn in recent_turns:
                last_query = turn.get('query', '')
                last_response = turn.get('response', '')

                # Use the most recent response (up to 300 chars for context)
                if last_response:
                    # Take first 300 chars of response (usually contains main content)
                    snippet = last_response[:300].strip()

                    # Clean up snippet (remove incomplete sentences at end)
                    last_period = snippet.rfind('.')
                    if last_period > 100:  # Only truncate if we have enough content
                        snippet = snippet[:last_period + 1]

                    context_snippets.append(snippet)

            if context_snippets:
                # Use most recent snippet as context
                context_text = context_snippets[-1]

                # Append context to query for RAG retrieval
                # Format: "user query. [Context: previous discussion]"
                rewritten = f"{current_query}. Recent context: {context_text}"

                metadata['context_source'] = 'conversation_snippet'
                metadata['context_snippet'] = context_text[:100]  # Store first 100 chars for logging

                self.logger.info(
                    f"Rewritten query with conversation snippet:\n"
                    f"  Original: '{current_query}'\n"
                    f"  Context: '{context_text[:100]}...'\n"
                    f"  (using last {len(context_snippets)} turn(s))"
                )
                return rewritten, metadata

        # Strategy 2: Fallback to session primary topic if no history
        if session_context and session_context.get('primary_topic'):
            topic = session_context['primary_topic']
            rewritten = f"{current_query} on {topic}"
            metadata['context_source'] = 'session_topic'
            metadata['context_snippet'] = topic

            self.logger.info(
                f"Rewritten query: '{current_query}' -> '{rewritten}' "
                f"(using session topic '{topic}')"
            )
            return rewritten, metadata

        # Fallback: Return original query with warning
        self.logger.warning(
            f"Could not rewrite follow-up query '{current_query}' - "
            f"no context available (history length: {len(conversation_history)})"
        )
        return current_query, metadata

    def _extract_topics_from_text(self, text: str, top_n: int = 3) -> List[str]:
        """
        Extract likely topics from text using enhanced NLP

        Args:
            text: Text to extract topics from
            top_n: Number of top topics to return

        Returns:
            List of topic strings

        Improvements:
        - Enhanced stopword list with academic/conversational terms
        - Better pattern matching (handles ALL CAPS, hyphenated, numbers)
        - Minimum frequency threshold (2+ occurrences)
        - Multi-word phrase detection
        """
        # Enhanced stopword list (includes conversational/academic terms)
        stopwords = {
            # Articles & conjunctions
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'it', 'its', 'you', 'your', 'i', 'my',
            'we', 'our', 'they', 'their', 'what', 'which', 'who', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'one', 'two', 'three',
            # Conversational fillers
            'help', 'please', 'thanks', 'thank', 'sure', 'okay', 'yes', 'here',
            'there', 'also', 'just', 'like', 'well', 'need', 'want', 'know',
            # Academic/explanation terms
            'concept', 'example', 'case', 'way', 'thing', 'explain', 'understand',
            'definition', 'explanation', 'means', 'refer', 'refers', 'called',
            # Common verbs
            'make', 'makes', 'making', 'made', 'use', 'uses', 'using', 'used',
            'work', 'works', 'working', 'worked', 'come', 'comes', 'coming',
            'get', 'gets', 'getting', 'got', 'take', 'takes', 'taking', 'took',
            'give', 'gives', 'giving', 'gave', 'tell', 'tells', 'telling', 'told',
            'think', 'thinks', 'thinking', 'thought', 'find', 'finds', 'finding'
        }

        # IMPROVED: Extract multiple patterns
        # 1. Capitalized phrases (e.g., "Machine Learning")
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b', text)

        # 2. ALL CAPS terms (e.g., "DNA", "API", "SQL")
        all_caps = re.findall(r'\b[A-Z]{2,}\b', text)

        # 3. Hyphenated terms (e.g., "machine-learning", "self-driving")
        hyphenated = re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*\b', text)

        # 4. Technical terms with numbers (e.g., "Python3", "HTTP2")
        tech_with_numbers = re.findall(r'\b[A-Za-z]+\d+\b', text)

        # 5. Regular significant words (4+ chars)
        regular_words = re.findall(r'\b[a-z]{4,}\b', text.lower())

        # Combine all extracted terms
        all_terms = []

        # Add capitalized phrases (highest priority)
        all_terms.extend(capitalized)

        # Add ALL CAPS terms (technical acronyms)
        all_terms.extend(all_caps)

        # Add hyphenated terms
        all_terms.extend(hyphenated)

        # Add technical terms with numbers
        all_terms.extend(tech_with_numbers)

        # Add regular words
        all_terms.extend(regular_words)

        # Filter out stopwords and normalize
        filtered_terms = []
        for term in all_terms:
            # Normalize to lowercase for comparison
            term_lower = term.lower()

            # Skip if it's a stopword
            if term_lower in stopwords:
                continue

            # Skip single letters
            if len(term_lower) <= 1:
                continue

            # Keep the term
            filtered_terms.append(term_lower)

        # Count frequency
        from collections import Counter
        term_counts = Counter(filtered_terms)

        # IMPROVED: Filter by minimum frequency (2+ occurrences) for reliability
        # If nothing appears 2+ times, fall back to top single occurrences
        frequent_terms = [(term, count) for term, count in term_counts.most_common() if count >= 2]

        if not frequent_terms and term_counts:
            # Fallback: return top terms even if they only appear once
            frequent_terms = term_counts.most_common(top_n)
        elif frequent_terms:
            frequent_terms = frequent_terms[:top_n]

        # IMPROVED: Try to find multi-word phrases from capitalized matches
        if capitalized:
            # Count capitalized phrases
            phrase_counts = Counter([phrase.lower() for phrase in capitalized])
            for phrase, count in phrase_counts.most_common(2):
                # If a multi-word phrase appears 2+ times, prioritize it
                if count >= 2 and len(phrase.split()) >= 2:
                    return [phrase]  # Return the multi-word phrase as top topic

        # Return the top N terms
        return [term for term, count in frequent_terms]

    def enhance_query_with_session_context(
        self,
        query: str,
        session_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Enhance query with session context for better retrieval (even if not follow-up)

        This adds implicit context from the session to improve retrieval quality.

        Args:
            query: User's current query
            session_context: Session metadata (courses, topics)
            conversation_history: Recent conversation

        Returns:
            Enhanced query string
        """
        # If query is already specific (mentions course/topic), don't modify
        if len(query.split()) > 10:
            return query

        enhancements = []

        # Add course context if available
        if session_context.get('courses_discussed'):
            # Don't add if query already mentions a course
            courses = session_context['courses_discussed']
            if courses and not any(course.lower() in query.lower() for course in courses):
                enhancements.append(f"course context: {courses[0]}")

        # Add topic context from recent conversation
        if conversation_history:
            recent_topics = []
            for turn in conversation_history[-2:]:  # Last 2 turns
                topic = self.extract_topic_from_query(turn.get('query', ''))
                if topic and topic not in recent_topics:
                    recent_topics.append(topic)

            is_followup, _ = self.is_followup_query(query)
            if recent_topics and not is_followup:
                # Only add if not already a follow-up (avoid redundancy)
                enhancements.append(f"related to: {recent_topics[-1]}")

        if enhancements:
            enhanced = f"{query} ({'; '.join(enhancements)})"
            self.logger.debug(f"Enhanced query with context: '{query}' -> '{enhanced}'")
            return enhanced

        return query


# Global instance
query_rewriter = QueryRewriter()
