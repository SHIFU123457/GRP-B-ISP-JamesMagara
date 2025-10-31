"""
Adaptive Response Engine for Study Helper Agent

This module provides advanced personalization features:
1. Response Length & Depth Control - Dynamic verbosity adjustment
2. Context-Aware Topic Bridging - Links to previous struggles
3. Real-Time Sentiment & Confusion Detection - Adjusts tone based on user state
4. Explanation Style Customization - Adapts to learning preferences
5. Content Difficulty Adaptation - Filters content by mastery level

Author: Study Helper Agent Team
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter

from sqlalchemy import desc, and_
from sqlalchemy.orm import Session

from config.database import db_manager
from config.settings import settings
from src.data.models import (
    User,
    UserInteraction,
    PersonalizationProfile,
    ConversationSession,
    QuizSession,
    ConfusionEvent,
    StruggleTopic
)
from src.services.explanation_style_engine import explanation_style_engine

logger = logging.getLogger(__name__)


class SentimentDetector:
    """Detect user sentiment and confusion in queries"""

    # Frustration/confusion markers
    # Note: Patterns handle common apostrophe omissions (dont vs don't, cant vs can't, im vs I'm)
    CONFUSION_PATTERNS = [
        r'\bstill\s+(don\'t|dont|do\s+not)\s+(get|understand)',
        r'\bconfused\b',
        r'\bmakes?\s+no\s+sense\b',
        r'\b(don\'t|dont|do\s+not)\s+(quite\s+|really\s+|fully\s+)?(understand|get|follow|grasp)\b',  # "dont understand", "dont quite understand", "dont get"
        r'\bwhat\s+does\s+that\s+mean\b',
        r'\b(i\'m|im)\s+(lost|confused)\b',  # "I'm lost", "im lost"
        r'\bthis\s+is\s+(hard|difficult|confusing|unclear)',
        r'\bhelp\s+me\b',
        r'\b(can\'t|cant|cannot)\s+(figure|understand|get)\b',  # "cant figure", "can't understand"
        r'\bnot\s+clear\b',  # "not clear to me"
        r'\b(having\s+trouble|struggling)\s+(with|understanding)\b'  # "having trouble with", "struggling understanding"
    ]

    # Follow-up indicators
    FOLLOWUP_PATTERNS = [
        r'\bwhat\s+about\b',
        r'\band\s+also\b',
        r'\bcan\s+you\s+explain\s+more\b',
        r'\btell\s+me\s+more\b',
        r'\belaborate\b',
        r'\bin\s+more\s+detail\b',
        r'\bgo\s+deeper\b'
    ]

    # Brevity preference indicators
    BREVITY_PATTERNS = [
        r'\bin\s+short\b',
        r'\bbriefly\b',
        r'\bquick\s+(answer|explanation|summary)\b',
        r'\bjust\s+the\s+basics\b',
        r'\btl;?dr\b',
        r'\bsummarize\b',
        r'\bin\s+a\s+nutshell\b'
    ]

    def __init__(self):
        self.confusion_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.CONFUSION_PATTERNS]
        self.followup_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.FOLLOWUP_PATTERNS]
        self.brevity_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.BREVITY_PATTERNS]

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query for sentiment and preference indicators

        Returns:
            dict with:
                - is_confused: bool
                - wants_more_detail: bool
                - wants_brevity: bool
                - confusion_score: float (0-1)
                - detected_patterns: list of matched patterns
        """
        query_lower = query.lower()

        # Check confusion
        confusion_matches = [p.search(query) for p in self.confusion_regex]
        is_confused = any(confusion_matches)
        confusion_score = sum(1 for m in confusion_matches if m) / len(self.confusion_regex)

        # Check follow-up desire
        followup_matches = [p.search(query) for p in self.followup_regex]
        wants_more_detail = any(followup_matches)

        # Check brevity preference
        brevity_matches = [p.search(query) for p in self.brevity_regex]
        wants_brevity = any(brevity_matches)

        detected_patterns = []
        if is_confused:
            detected_patterns.append("confusion")
        if wants_more_detail:
            detected_patterns.append("wants_detail")
        if wants_brevity:
            detected_patterns.append("wants_brevity")

        return {
            'is_confused': is_confused,
            'wants_more_detail': wants_more_detail,
            'wants_brevity': wants_brevity,
            'confusion_score': confusion_score,
            'detected_patterns': detected_patterns
        }

    def log_confusion_event(
        self,
        user_id: int,
        query: str,
        analysis: Dict[str, Any],
        session: Session,
        session_id: Optional[str] = None,
        interaction_id: Optional[int] = None,
        topic: Optional[str] = None,
        previous_topic: Optional[str] = None
    ) -> Optional[ConfusionEvent]:
        """
        Log confusion detection to database

        Args:
            user_id: User database ID
            query: The query text
            analysis: Result from analyze_query()
            session: Database session
            session_id: Optional conversation session ID
            interaction_id: Optional interaction ID
            topic: Optional current topic
            previous_topic: Optional previous topic

        Returns:
            Created ConfusionEvent or None if not confused
        """
        try:
            # Only log if there's meaningful confusion
            confusion_score = analysis['confusion_score']

            # Determine confusion type
            # Lowered thresholds to be more sensitive to subtle confusion signals
            # Even single confusion indicators (e.g., "don't quite understand") are valuable data
            if confusion_score < 0.1:
                confusion_type = 'none'        # 0-9%: No confusion detected
            elif confusion_score < 0.3:
                confusion_type = 'mild'        # 10-29%: 1-2 patterns (e.g., "dont understand")
            elif confusion_score < 0.6:
                confusion_type = 'moderate'    # 30-59%: 3-6 patterns
            else:
                confusion_type = 'severe'      # 60%+: 7+ patterns (very confused)

            # Only log if mild or above
            if confusion_type == 'none':
                return None

            # Create confusion event
            event = ConfusionEvent(
                user_id=user_id,
                session_id=session_id,
                interaction_id=interaction_id,
                confusion_score=confusion_score,
                confusion_type=confusion_type,
                confidence=0.8,  # Default confidence
                indicators=analysis.get('detected_patterns', []),
                detected_patterns=analysis.get('detected_patterns', []),
                query_text=query[:500],  # Limit length
                topic=topic,
                previous_topic=previous_topic,
                wants_more_detail=analysis.get('wants_more_detail', False),
                wants_brevity=analysis.get('wants_brevity', False)
            )

            session.add(event)
            session.commit()

            logger.info(f"Logged {confusion_type} confusion event for user {user_id}")
            return event

        except Exception as e:
            logger.error(f"Error logging confusion event: {e}", exc_info=True)
            return None

    def get_recent_confusion_pattern(
        self,
        user_id: int,
        session: Session,
        lookback_hours: int = 24
    ) -> str:
        """
        Analyze recent confusion pattern for a user

        Returns: 'none', 'occasional', 'frequent', 'persistent'
        """
        try:
            from datetime import timezone

            cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

            # Get recent confusion events
            events = session.query(ConfusionEvent).filter(
                and_(
                    ConfusionEvent.user_id == user_id,
                    ConfusionEvent.created_at >= cutoff
                )
            ).all()

            if not events:
                return 'none'

            # Count by severity
            severe_count = sum(1 for e in events if e.confusion_type == 'severe')
            moderate_count = sum(1 for e in events if e.confusion_type == 'moderate')
            total_count = len(events)

            # Classify pattern
            if severe_count >= 3 or total_count >= 8:
                return 'persistent'
            elif moderate_count >= 3 or total_count >= 5:
                return 'frequent'
            elif total_count >= 2:
                return 'occasional'
            else:
                return 'none'

        except Exception as e:
            logger.error(f"Error analyzing confusion pattern: {e}", exc_info=True)
            return 'none'


class TopicBridgeAnalyzer:
    """Analyze topic relationships and user struggles"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TopicBridgeAnalyzer")
        self._llm_service = None  # Lazy load to avoid circular imports
        self._embedding_model = None  # Lazy load embedding model

    @property
    def llm_service(self):
        """Lazy load LLM service to avoid circular imports"""
        if self._llm_service is None:
            from src.services.llm_integration import LLMService
            self._llm_service = LLMService()
        return self._llm_service

    @property
    def embedding_model(self):
        """Lazy load embedding model for semantic similarity"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                from config.settings import settings
                self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                self.logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                self._embedding_model = None
        return self._embedding_model

    def extract_topics(self, query: str) -> List[str]:
        """
        Extract key topics from query using multi-layered approach

        Multi-layered extraction strategy for maximum reliability:
        1. Try LLM with educational context (best quality, may refuse)
        2. Try proper noun extraction (fast, good for entities)
        3. Try embedding-based matching (semantic understanding)
        4. Fallback to improved regex (always works)

        Args:
            query: User's query text

        Returns:
            List of identified topics (e.g., ['Enron', 'Tuskys', 'corporate governance'])
        """
        self.logger.debug(f"Topic extraction starting for: '{query[:80]}'")

        # LAYER 1: Try LLM with improved prompt
        try:
            topics = self._extract_topics_llm(query)
            if topics and len(topics) > 0:
                self.logger.info(f"Topic extraction - SUCCESS via LLM: {topics}")
                return topics
        except Exception as e:
            self.logger.warning(f"Layer 1 (LLM) failed: {e}")

        # LAYER 2: Try proper noun extraction
        try:
            proper_nouns = self._extract_proper_nouns(query)
            if proper_nouns and len(proper_nouns) > 0:
                self.logger.info(f"Topic extraction - SUCCESS via proper nouns: {proper_nouns}")
                return proper_nouns
        except Exception as e:
            self.logger.warning(f"Layer 2 (proper nouns) failed: {e}")

        # LAYER 3: Try embedding-based extraction
        if self.embedding_model:
            try:
                topics = self._extract_topics_embedding_based(query)
                if topics and len(topics) > 0:
                    self.logger.info(f"Topic extraction - SUCCESS via embeddings: {topics}")
                    return topics
            except Exception as e:
                self.logger.warning(f"Layer 3 (embeddings) failed: {e}")

        # LAYER 4: Fallback to improved regex
        self.logger.info("Topic extraction - Using fallback regex")
        topics = self._fallback_topic_extraction(query)
        self.logger.info(f"Topic extraction - SUCCESS via regex fallback: {topics}")
        return topics

    def extract_topics_with_context(
        self,
        query: str,
        user_id: int = None,
        session_id: str = None,
        db_session = None,
        conv_session = None
    ) -> List[str]:
        """
        Context-aware topic extraction that searches conversation history

        Handles context-referencing queries like:
        - "Could you correct your previous answer?"
        - "Explain your first response"
        - "Clarify that explanation"

        Args:
            query: User's query text
            user_id: User database ID (optional, for history search)
            session_id: Conversation session ID (optional, for history search)
            db_session: Database session (optional, for history queries)
            conv_session: ConversationSession object (optional, for primary_topic)

        Returns:
            List of extracted topics, with history fallback if context query
        """
        # Try normal extraction first
        topics = self.extract_topics(query)

        # If we got good topics, return them
        if topics and topics != ['general']:
            return topics

        # Check if this is a context-referencing query
        context_patterns = {
            'first': r'\b(first|initial|original)\s+(question|answer|response|message)',
            'previous': r'\b(previous|last|earlier|prior)\s+(question|answer|response|message)',
            'that': r'\b(that|the above|your)\s+(answer|response|explanation)',
            'correct': r'\bcorrect\s+(your|the|that)\s+(answer|response|explanation)',
            'pronoun': r'\b(explain|clarify|elaborate|tell me more about)\s+(it|that|this|them)\b',
            'followup': r'^(explain|clarify|elaborate on|tell me more)\s+(further|more|better)',
        }

        query_lower = query.lower()
        context_type = None

        for ctype, pattern in context_patterns.items():
            if re.search(pattern, query_lower):
                context_type = ctype
                self.logger.info(f"Detected {ctype}-type context reference in query")
                break

        if not context_type:
            # Not a context query, return what we have
            return topics

        # STRATEGY 1: For "first" queries, try primary_topic (fastest)
        if context_type == 'first' and conv_session and conv_session.primary_topic:
            self.logger.info(f"Using primary session topic for 'first' reference: {conv_session.primary_topic}")
            return [conv_session.primary_topic]

        # STRATEGY 2: For "previous/that" queries, try session_context.current_topic
        if context_type in ['previous', 'that', 'correct'] and conv_session:
            if conv_session.session_context and conv_session.session_context.get('current_topic'):
                current_topic = conv_session.session_context['current_topic']
                self.logger.info(f"Using session_context.current_topic for '{context_type}' reference: {current_topic}")
                return [current_topic]

        # STRATEGY 3: Search conversation history (most accurate but slower)
        if user_id and session_id and db_session:
            try:
                from src.services.personalization_engine import session_manager

                # Retrieve conversation history
                history = session_manager.get_conversation_history(
                    user_id=user_id,
                    db_session=db_session,
                    limit=10,  # Last 10 messages
                    session_id=session_id
                )

                if history:
                    # Determine which message to extract from
                    if context_type == 'first':
                        target_message = history[0]  # First message in session
                        msg_desc = "first"
                    else:  # previous, that, correct
                        target_message = history[-1]  # Most recent message
                        msg_desc = "previous"

                    # Extract topics from the target message's query
                    inherited_topics = self.extract_topics(target_message['query'])

                    if inherited_topics and inherited_topics != ['general']:
                        self.logger.info(f"Inherited topics from {msg_desc} message in history: {inherited_topics}")
                        return inherited_topics
                    else:
                        self.logger.warning(f"Could not extract topics from {msg_desc} message: {target_message['query'][:80]}")

            except Exception as e:
                self.logger.warning(f"Failed to search conversation history: {e}")

        # STRATEGY 4: Fallback to primary_topic as last resort
        if conv_session and conv_session.primary_topic:
            self.logger.info(f"Fallback to primary session topic: {conv_session.primary_topic}")
            return [conv_session.primary_topic]

        # No context available, return original extraction result
        self.logger.warning("Context query detected but no history/context available")
        return topics

    def _extract_topics_llm(self, query: str) -> List[str]:
        """
        Layer 1: Extract topics using LLM
        Most accurate but may refuse on sensitive topics
        """
        # CRITICAL FIX: Detect context-referencing queries that lack concrete topics
        # These queries reference previous conversation ("previous question", "that answer", "above")
        # Small LLMs hallucinate topics from examples instead of extracting from vague queries
        context_reference_patterns = [
            r'\b(previous|earlier|above|prior|last|that)\s+(question|answer|response|explanation|query)',
            r'\b(your|the)\s+(previous|earlier|above|prior|last)\s+',
            r'\b(correct|fix|improve|clarify)\s+(your|the|that)\s+(answer|response|explanation)',
            r'\bbased\s+off\s+(this|that)\s+knowledge\b',
            r'\bas\s+I\s+(said|mentioned|asked)\s+',
        ]

        query_lower = query.lower()
        is_context_query = any(re.search(pattern, query_lower) for pattern in context_reference_patterns)

        if is_context_query:
            # Skip LLM extraction for context-referencing queries
            # LLM will hallucinate topics from examples instead of extracting from vague query
            self.logger.info(f"Topic extraction - Skipping LLM for context-referencing query: '{query[:80]}'")
            return []  # Let orchestrator try other layers

        try:
            # Use LLM to extract topics with clearer, more flexible formatting instructions
            # Added educational context to prevent safety refusals on any academic topics
            # REMOVED EXAMPLES to prevent contamination from small models copying them
            prompt = f"""SYSTEM: You are a topic extraction tool. Extract ONLY the main academic topics/concepts mentioned in the query.

CRITICAL RULES:
1. Extract topics that are EXPLICITLY mentioned in the query
2. DO NOT infer topics from context or previous conversations
3. If the query is vague or refers to "previous question", return "UNABLE_TO_EXTRACT"
4. Return ONLY topic names, NO explanations or commentary

STUDENT QUESTION: "{query}"

TOPICS (comma-separated):"""

            # Get LLM response
            llm_response = self.llm_service.generate_response(
                query="Extract topics",
                context=prompt,
                user_preferences=None
            )

            # Log the raw response for debugging
            self.logger.info(f"Topic extraction - Query: '{query[:80]}'")
            self.logger.info(f"Topic extraction - Raw LLM response: '{llm_response[:300]}'")

            # Parse the response to extract topic names
            # Strategy 1: Try to extract from structured format first
            topics = []
            lines = llm_response.strip().split('\n')

            # First, try to find lines that look like clean topic names
            for line in lines:
                line = line.strip()

                # Skip empty lines
                if not line or len(line) < 2:
                    continue

                # Remove common prefixes (numbers, bullets, dashes, "Topics:")
                cleaned = re.sub(r'^[-\*\d\.]+\s*', '', line)
                cleaned = re.sub(r'^topics?[\s:]*', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'^here are.*?:\s*', '', cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.strip()

                # NEW: Handle space-separated or comma-separated topics on a single line
                # If the line looks like multiple topics (has multiple capitalized words or commas)
                if len(cleaned) > 60 or ',' in cleaned:
                    # Split by commas first, then by spaces if no commas
                    if ',' in cleaned:
                        potential_topics = [t.strip() for t in cleaned.split(',')]
                    else:
                        # Split by spaces and group into reasonable topic names
                        words = cleaned.split()
                        # Check if these are single-word topics (common for space-separated format)
                        if len(words) >= 2 and all(len(w) > 2 and w[0].isupper() for w in words[:3]):
                            # Likely space-separated topics like "Competition Conflict Resolution"
                            potential_topics = words
                        else:
                            # Not clearly space-separated, treat as single topic
                            potential_topics = [cleaned]

                    # Process each potential topic
                    for pt in potential_topics:
                        pt = pt.strip()
                        if 2 <= len(pt) <= 50:  # Increased from 40 to 50
                            topics.append(pt.lower())
                    continue

                # Skip if it's too long (likely a full sentence, not a topic)
                if len(cleaned) > 50:  # Increased from 40 to 50
                    continue

                # Skip if it ends with punctuation (likely a sentence)
                if cleaned.endswith(('.', '?', '!')):
                    continue

                # Skip if it contains colon in the middle (likely a header)
                if ':' in cleaned and cleaned.index(':') < len(cleaned) // 2:
                    continue

                # Accept if it looks like a reasonable topic (mostly alphanumeric + spaces)
                if cleaned:
                    alpha_count = sum(c.isalnum() or c.isspace() or c == '-' for c in cleaned)
                    if alpha_count / len(cleaned) >= 0.7:  # At least 70% valid characters
                        topics.append(cleaned.lower())

            # Strategy 2: If no topics found, try to extract keywords from the full response
            if not topics:
                self.logger.debug("Strategy 1 failed, trying keyword extraction from full response")
                # Look for phrases after "topics", "topic:", "about", etc.
                patterns = [
                    r'topics?[\s:]+(.+?)(?:\.|$)',
                    r'(?:about|discussing|regarding)[\s:]+(.+?)(?:\.|$)',
                    r'main\s+(?:topic|concept|subject)[\s:]+(.+?)(?:\.|$)',
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, llm_response, re.IGNORECASE)
                    for match in matches:
                        # Split by commas or "and"
                        parts = re.split(r',|\sand\s', match)
                        for part in parts:
                            part = part.strip()
                            if 2 <= len(part) <= 50:  # Increased from 30 to 50
                                topics.append(part.lower())

            # Remove duplicates while preserving order
            seen = set()
            unique_topics = []
            for topic in topics:
                topic = topic.strip()
                if topic and topic not in seen and len(topic) >= 2:
                    seen.add(topic)
                    unique_topics.append(topic)

            # Limit to top 3 topics
            topics = unique_topics[:3]

            # Filter out garbage topics (more lenient filtering)
            filtered_topics = []
            garbage_keywords = ['listed', 'one per line', 'underlying', 'its', 'unable_to_extract', 'i can assist']

            for topic in topics:
                # Skip if topic is just a garbage word
                if any(garbage in topic.lower() for garbage in garbage_keywords):
                    continue
                # Skip single words that are too generic
                if len(topic.split()) == 1 and topic.lower() in ['concept', 'topic', 'question', 'explain', 'the', 'a', 'an']:
                    continue
                filtered_topics.append(topic)

            # CRITICAL: Validate that topics actually appear in the query
            # This prevents LLM from hallucinating topics from examples or previous context
            query_lower = query.lower()
            validated_topics = []

            for topic in filtered_topics:
                topic_words = topic.lower().split()

                # Check if ANY significant word from topic appears in query
                # Significant = more than 3 characters
                significant_words = [w for w in topic_words if len(w) > 3]

                if significant_words:
                    # At least one significant word must appear in query
                    if any(word in query_lower for word in significant_words):
                        validated_topics.append(topic)
                    else:
                        self.logger.warning(f"Topic extraction - REJECTED hallucinated topic not in query: '{topic}'")
                else:
                    # Topic has no significant words, keep it (e.g., "DNA", "SQL")
                    validated_topics.append(topic)

            if validated_topics:
                self.logger.info(f"Topic extraction - SUCCESS via LLM: {validated_topics}")
                return validated_topics
            else:
                self.logger.warning(f"Topic extraction - LLM returned garbage")
                self.logger.warning(f"Topic extraction - Unfiltered topics were: {topics}")
                self.logger.debug(f"Topic extraction - Full LLM response: {llm_response}")
                # Return empty list to signal failure, let orchestrator try next layer
                return []

        except Exception as e:
            self.logger.warning(f"Error in LLM topic extraction: {e}")
            # Return empty list to signal failure, let orchestrator try next layer
            return []

    def _fallback_topic_extraction(self, query: str) -> List[str]:
        """Simple fallback method when LLM is unavailable - improved version"""
        self.logger.info(f"Topic extraction - Using fallback regex for: '{query[:80]}'")
        query_lower = query.lower()

        # Strategy 1: Look for patterns like "concept of X", "explain X", "what is X"
        # Note: query_lower is already lowercase, so patterns use [a-z]
        # Use non-greedy matching and stop at punctuation
        patterns = [
            r'(?:concept|idea|principle|theory)\s+of\s+([a-z0-9\s-]{2,30}?)(?:\.|\?|$)',  # "concept of BERT"
            r'(?:explain|describe|define)\s+(?:the\s+)?([a-z0-9\s-]{2,30}?)(?:\.|\?|$)',  # "explain BERT"
            r'what\s+is\s+(?:the\s+)?([a-z0-9\s-]{2,30}?)(?:\.|\?|$)',  # "what is BERT"
            r'how\s+(?:does|do)\s+([a-z0-9\s-]{2,30}?)(?:\.|\?|$)',  # "how does BERT"
            r'(?:about|regarding)\s+([a-z0-9\s-]{2,30}?)(?:\.|\?|$)',  # "about BERT"
            r'(?:understand|understood)\s+(?:the\s+)?(?:concept\s+of\s+)?([a-z0-9\s-]{2,30}?)(?:\.|\?|,|$)',  # "understood BERT" or "understood the concept of BERT"
        ]

        for i, pattern in enumerate(patterns, 1):
            match = re.search(pattern, query_lower)
            if match:
                topic = match.group(1).strip()
                # Clean up the topic
                topic = re.sub(r'\s+(work|function|operate|used)$', '', topic)
                topic = re.sub(r'^(the|a|an)\s+', '', topic)
                if len(topic) > 3 and topic not in ['concept', 'idea', 'thing']:
                    self.logger.info(f"Topic extraction - SUCCESS via fallback pattern #{i}: '{topic}'")
                    return [topic]

        # Strategy 2: Extract significant words (backup)
        question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'can',
                         'could', 'would', 'should', 'do', 'does', 'did', 'explain', 'tell', 'me', 'you'}
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'about', 'as', 'into', 'like', 'through', 'this', 'that',
                    'concept', 'idea', 'thing', 'stuff'}

        # Extract significant words (longer than 3 chars, not stopwords)
        # Also preserve original casing to detect proper nouns
        words_original = query.split()
        words_lower = query_lower.split()

        significant_words = []
        for orig, lower in zip(words_original, words_lower):
            cleaned = lower.strip('?.,!;:')
            if (len(cleaned) > 3
                and cleaned not in question_words
                and cleaned not in stopwords
                and cleaned.isalpha()):
                # Keep original if it's a proper noun (starts with capital)
                if orig[0].isupper() and not words_original[0] == orig:  # Not first word of sentence
                    significant_words.append(orig.strip('?.,!;:'))
                else:
                    significant_words.append(cleaned)

        # Smart topic extraction: prefer proper nouns as standalone topics
        topics = []
        i = 0

        while i < len(significant_words):
            word = significant_words[i]

            # If it's a proper noun (capitalized), keep as single topic
            if word[0].isupper():
                topics.append(word)
                i += 1
            # If next word exists and current word looks like adjective/descriptor
            elif i < len(significant_words) - 1:
                next_word = significant_words[i+1]
                # Check if this forms a meaningful compound
                # e.g., "corporate governance", "machine learning", but NOT "some similarities"
                descriptor_words = {'some', 'many', 'between', 'about', 'their', 'these', 'those'}
                if word.lower() not in descriptor_words:
                    combined = f"{word} {next_word}"
                    topics.append(combined)
                    i += 2
                else:
                    # Skip descriptor words
                    i += 1
            else:
                topics.append(word)
                i += 1

        # Return up to 3 topics, filter out pure descriptors
        result = [t for t in topics if t.lower() not in {'some', 'between', 'about', 'their'}][:3]
        result = result if result else ['general']
        if result:
            self.logger.info(f"Topic extraction - SUCCESS via word extraction: {result}")
        else:
            self.logger.warning(f"Topic extraction - All strategies failed, using 'general'")
        return result

    def _extract_proper_nouns(self, query: str) -> List[str]:
        """
        Extract proper nouns (capitalized words/phrases) from query
        Layer 2 of multi-layered extraction
        """
        import re

        # Match capitalized words, including multi-word proper nouns
        # e.g., "Google Classroom", "DNA", "Enron"
        proper_noun_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
        matches = re.findall(proper_noun_pattern, query)

        # Filter out sentence-starting words if they're common question words
        question_starters = {'What', 'When', 'Where', 'Who', 'Why', 'How', 'Can', 'Could',
                           'Would', 'Should', 'Do', 'Does', 'Did', 'Is', 'Are', 'Explain',
                           'Tell', 'Describe', 'Compare', 'Contrast'}

        proper_nouns = [
            match for match in matches
            if match not in question_starters and len(match) > 2
        ]

        return proper_nouns[:3]  # Return top 3

    def _extract_topics_embedding_based(self, query: str) -> List[str]:
        """
        Extract topics using embedding similarity to predefined topic categories
        Layer 3 of multi-layered extraction
        """
        if not self.embedding_model:
            return []

        try:
            # Common academic topic categories with example keywords
            # These should match common topics in your course materials
            topic_categories = {
                # Business/Management
                'corporate_governance': ['corporate governance', 'board oversight', 'management ethics', 'Enron', 'Tuskys'],
                'financial_management': ['financial management', 'accounting', 'budgeting', 'financial analysis'],

                # Biology/Life Sciences
                'molecular_biology': ['DNA', 'RNA', 'genes', 'proteins', 'replication', 'transcription'],
                'cell_biology': ['cells', 'mitosis', 'meiosis', 'organelles', 'membrane'],
                'ecology': ['ecosystem', 'biodiversity', 'environment', 'species', 'habitat'],

                # Computer Science
                'algorithms': ['algorithm', 'sorting', 'searching', 'complexity', 'recursion'],
                'data_structures': ['arrays', 'linked lists', 'trees', 'graphs', 'hash tables'],
                'operating_systems': ['Linux', 'Windows', 'kernel', 'processes', 'threads'],
                'databases': ['SQL', 'database', 'queries', 'tables', 'normalization'],

                # General topics
                'security': ['security', 'encryption', 'authentication', 'vulnerabilities'],
                'networking': ['network', 'protocols', 'TCP', 'IP', 'routing'],
            }

            # Get query embedding
            query_embedding = self.embedding_model.encode([query])[0]

            # Find most similar keywords across all categories
            best_matches = []

            for category, keywords in topic_categories.items():
                # Get embeddings for this category's keywords
                keyword_embeddings = self.embedding_model.encode(keywords)

                # Calculate similarities
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    keyword_embeddings
                )[0]

                # Get top matching keywords from this category
                for idx, similarity in enumerate(similarities):
                    if similarity > 0.35:  # 35% similarity threshold
                        best_matches.append((keywords[idx], similarity, category))

            if not best_matches:
                return []

            # Sort by similarity score (highest first)
            best_matches.sort(key=lambda x: x[1], reverse=True)

            # Return top 3 topics
            topics = [topic for topic, _, _ in best_matches[:3]]

            return topics

        except Exception as e:
            self.logger.error(f"Embedding-based topic extraction failed: {e}")
            return []

    def update_struggle_topic(
        self,
        user_id: int,
        topic: str,
        session: Session,
        query_complexity: Optional[float] = None,
        user_rating: Optional[int] = None,
        is_confused: bool = False
    ) -> Optional[StruggleTopic]:
        """
        Update or create struggle topic entry in database

        This is called after each query to track persistent struggles
        """
        try:
            from datetime import timezone

            # Find existing active struggle topic
            struggle = session.query(StruggleTopic).filter(
                and_(
                    StruggleTopic.user_id == user_id,
                    StruggleTopic.topic == topic,
                    StruggleTopic.is_active == True
                )
            ).first()

            now = datetime.now(timezone.utc)

            if struggle:
                # Update existing struggle
                struggle.query_count += 1
                struggle.last_asked_at = now

                if is_confused:
                    struggle.confusion_count += 1

                if user_rating:
                    # Update rating stats
                    if struggle.avg_rating:
                        # Calculate new average
                        total_ratings = struggle.avg_rating * (struggle.query_count - 1)
                        struggle.avg_rating = (total_ratings + user_rating) / struggle.query_count
                    else:
                        struggle.avg_rating = float(user_rating)

                    if user_rating < 3:
                        struggle.low_rating_count += 1

                if query_complexity:
                    # Update complexity tracking
                    struggle.latest_query_complexity = query_complexity

                    # Determine trend
                    if struggle.first_query_complexity and struggle.latest_query_complexity:
                        if struggle.latest_query_complexity > struggle.first_query_complexity * 1.2:
                            struggle.complexity_trend = 'increasing'
                        elif struggle.latest_query_complexity < struggle.first_query_complexity * 0.8:
                            struggle.complexity_trend = 'decreasing'
                        else:
                            struggle.complexity_trend = 'flat'

                # Recalculate struggle score
                struggle.struggle_score = self._calculate_struggle_score(struggle)

                # Check if resolved (high ratings, no recent confusion)
                if struggle.avg_rating and struggle.avg_rating >= 4.0 and struggle.confusion_count == 0:
                    struggle.is_resolved = True
                    struggle.is_active = False
                    struggle.resolution_detected_at = now

                session.commit()
                return struggle

            else:
                # Create new struggle topic
                indicators = []
                if is_confused:
                    indicators.append('confusion_detected')

                struggle = StruggleTopic(
                    user_id=user_id,
                    topic=topic,
                    struggle_score=0.3,  # Initial score
                    query_count=1,
                    confusion_count=1 if is_confused else 0,
                    avg_rating=float(user_rating) if user_rating else None,
                    low_rating_count=1 if user_rating and user_rating < 3 else 0,
                    complexity_trend='flat',
                    first_query_complexity=query_complexity,
                    latest_query_complexity=query_complexity,
                    indicators=indicators,
                    first_asked_at=now,
                    last_asked_at=now,
                    is_active=True,
                    is_resolved=False
                )

                session.add(struggle)
                session.commit()

                self.logger.info(f"Created new struggle topic '{topic}' for user {user_id}")
                return struggle

        except Exception as e:
            self.logger.error(f"Error updating struggle topic: {e}", exc_info=True)
            return None

    def _calculate_struggle_score(self, struggle: StruggleTopic) -> float:
        """
        Calculate overall struggle score (0-1) based on multiple factors
        """
        score = 0.0

        # Factor 1: Query frequency (30%)
        if struggle.query_count >= 5:
            score += 1.0 * 0.30
        elif struggle.query_count >= 3:
            score += 0.6 * 0.30
        elif struggle.query_count >= 2:
            score += 0.3 * 0.30

        # Factor 2: Low ratings (35%)
        if struggle.avg_rating:
            if struggle.avg_rating < 2.5:
                score += 1.0 * 0.35
            elif struggle.avg_rating < 3.5:
                score += 0.5 * 0.35

        # Factor 3: Confusion frequency (20%)
        confusion_ratio = struggle.confusion_count / max(struggle.query_count, 1)
        score += confusion_ratio * 0.20

        # Factor 4: Complexity trend (15%)
        if struggle.complexity_trend == 'decreasing':
            score += 1.0 * 0.15  # Regressing to basics = struggling
        elif struggle.complexity_trend == 'flat':
            score += 0.6 * 0.15  # No progression = stuck

        return min(score, 1.0)

    def get_struggle_topics(
        self,
        user_id: int,
        session: Session,
        active_only: bool = True
    ) -> List[StruggleTopic]:
        """
        Get struggle topics from database (new DB-backed version)

        Args:
            user_id: User database ID
            session: Database session
            active_only: Only return active (unresolved) struggles

        Returns:
            List of StruggleTopic objects sorted by struggle_score
        """
        try:
            query = session.query(StruggleTopic).filter(
                StruggleTopic.user_id == user_id
            )

            if active_only:
                query = query.filter(StruggleTopic.is_active == True)

            struggles = query.order_by(StruggleTopic.struggle_score.desc()).all()

            return struggles

        except Exception as e:
            self.logger.error(f"Error getting struggle topics: {e}", exc_info=True)
            return []

    def find_topic_connections(
        self,
        current_topics: List[str],
        struggle_topics: List[StruggleTopic]
    ) -> List[Dict[str, Any]]:
        """
        Find connections between current query and past struggles using semantic similarity

        Uses embedding-based similarity to identify related topics across ALL domains
        (Biology, CS, Math, Physics, etc.) without hardcoded mappings.

        Args:
            current_topics: Topics from current query
            struggle_topics: List of StruggleTopic objects from DB

        Returns list of connections with:
            - related_topic: str
            - connection_type: str (strongly_related, related, weakly_related)
            - similarity_score: float (0-1)
            - struggle_score: float
            - query_count: int
        """
        if not current_topics or not struggle_topics:
            return []

        # Check if embedding model is available
        if self.embedding_model is None:
            self.logger.warning("Embedding model not available, using fallback exact match")
            return self._find_connections_fallback(current_topics, struggle_topics)

        try:
            connections = []

            # Generate embeddings for current topics
            current_embeddings = self.embedding_model.encode(current_topics)

            # Get struggle topic texts
            struggle_topic_texts = [s.topic for s in struggle_topics]
            struggle_embeddings = self.embedding_model.encode(struggle_topic_texts)

            # Calculate cosine similarity between each current topic and struggle topic
            for i, current_topic in enumerate(current_topics):
                current_emb = current_embeddings[i].reshape(1, -1)

                for j, struggle in enumerate(struggle_topics):
                    struggle_emb = struggle_embeddings[j].reshape(1, -1)

                    # Calculate cosine similarity
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity = cosine_similarity(current_emb, struggle_emb)[0][0]

                    # Define similarity thresholds for connection types
                    # High similarity (>75%) = strongly related (likely prerequisite or same concept)
                    # Medium similarity (55-75%) = related (connected concepts)
                    # Low similarity (40-55%) = weakly related (tangentially connected)
                    # Very low (<40%) = not connected

                    if similarity >= 0.75:
                        connection_type = 'strongly_related'
                    elif similarity >= 0.55:
                        connection_type = 'related'
                    elif similarity >= 0.40:
                        connection_type = 'weakly_related'
                    else:
                        continue  # Skip if similarity is too low

                    connections.append({
                        'related_topic': struggle.topic,
                        'current_topic': current_topic,
                        'connection_type': connection_type,
                        'similarity_score': float(similarity),
                        'struggle_score': struggle.struggle_score,
                        'query_count': struggle.query_count,
                        'last_asked': struggle.last_asked_at
                    })

            # Sort by similarity score (highest first)
            connections.sort(key=lambda x: x['similarity_score'], reverse=True)

            # Log top connections for debugging
            if connections:
                self.logger.info(f"Found {len(connections)} topic connections using semantic similarity")
                for conn in connections[:3]:  # Log top 3
                    self.logger.debug(
                        f"  '{conn['current_topic']}' → '{conn['related_topic']}' "
                        f"({conn['connection_type']}, similarity={conn['similarity_score']:.2f})"
                    )
            else:
                self.logger.debug(f"No semantic connections found for topics: {current_topics}")

            return connections

        except Exception as e:
            self.logger.error(f"Error in semantic topic connection finding: {e}", exc_info=True)
            # Fallback to simple exact match
            return self._find_connections_fallback(current_topics, struggle_topics)

    def _find_connections_fallback(
        self,
        current_topics: List[str],
        struggle_topics: List[StruggleTopic]
    ) -> List[Dict[str, Any]]:
        """
        Fallback method for finding connections using simple string matching
        Used when embedding model is unavailable
        """
        connections = []

        for current_topic in current_topics:
            current_lower = current_topic.lower()

            for struggle in struggle_topics:
                struggle_lower = struggle.topic.lower()

                # Check for exact match or substring match
                if current_lower == struggle_lower or \
                   current_lower in struggle_lower or \
                   struggle_lower in current_lower:
                    connections.append({
                        'related_topic': struggle.topic,
                        'current_topic': current_topic,
                        'connection_type': 'exact_match',
                        'similarity_score': 1.0,
                        'struggle_score': struggle.struggle_score,
                        'query_count': struggle.query_count,
                        'last_asked': struggle.last_asked_at
                    })

        return connections


class VerbosityController:
    """
    Simplified verbosity control based solely on DB-stored preferences

    Uses:
    1. Current query explicit signals ("briefly", "in detail")
    2. Profile preference from DB (auto-updated every 12 hours based on ratings)
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VerbosityController")

    def get_verbosity_preference(
        self,
        user_id: int,
        session: Session,
        current_query_analysis: Dict[str, Any]
    ) -> str:
        """
        Get verbosity preference: 'short', 'medium', or 'long'

        Priority:
        1. Explicit query signals (highest priority)
        2. Profile preference from DB (auto-updated based on ratings)
        3. Default to 'medium'

        Returns:
            'short', 'medium', or 'long'
        """
        # Check for explicit signals in current query
        if current_query_analysis.get('wants_brevity'):
            self.logger.debug(f"User {user_id} explicitly requested brevity")
            return 'short'

        if current_query_analysis.get('wants_more_detail'):
            self.logger.debug(f"User {user_id} explicitly requested detail")
            return 'long'

        # Get profile preference
        profile = session.query(PersonalizationProfile).filter(
            PersonalizationProfile.user_id == user_id
        ).first()

        if profile and profile.preferred_response_length:
            preference = profile.preferred_response_length
            self.logger.debug(f"User {user_id} profile preference: {preference}")
            return preference

        # Default to medium
        self.logger.debug(f"User {user_id} using default preference: medium")
        return 'medium'

    def generate_verbosity_instructions(
        self,
        verbosity_preference: str,
        user_level: str = "intermediate"
    ) -> str:
        """
        Generate LLM prompt instructions based on verbosity preference

        Args:
            verbosity_preference: 'short', 'medium', or 'long'
            user_level: beginner/intermediate/advanced

        Returns:
            String instructions to prepend to LLM prompt
        """
        # Safely handle None user_level
        level = user_level if user_level else "intermediate"

        if verbosity_preference == 'short':
            instructions = f"""Response Style Instructions:
- Keep response CONCISE (2-4 sentences maximum)
- Focus only on the core concept
- Skip detailed examples unless essential
- Assume {level} understanding
- Use bullet points for clarity"""

        elif verbosity_preference == 'long':
            instructions = f"""Response Style Instructions:
- Provide a COMPREHENSIVE explanation (6-10 sentences)
- Include multiple examples and edge cases
- Break down complex concepts step-by-step
- Add context and deeper insights
- Tailor to {level} level
- Use analogies where helpful"""

        else:  # medium
            instructions = f"""Response Style Instructions:
- Provide a BALANCED explanation (4-6 sentences)
- Include one clear example
- Cover main points without excessive detail
- Tailor to {level} level
- Use structured formatting"""

        return instructions

    def get_max_tokens_for_verbosity(self, verbosity_preference: str) -> int:
        """
        Get max_tokens parameter for LLM based on verbosity preference

        Provides soft guidance rather than strict enforcement
        System max is 8000, so these are reasonable ranges within that

        Args:
            verbosity_preference: 'short', 'medium', or 'long'

        Returns:
            Max tokens for LLM generation
        """
        max_tokens_map = {
            'short': 600,     # ~3-5 paragraphs, concise but complete
            'medium': 2400,   # ~5-8 paragraphs, balanced explanation
            'long': 5600      # ~8-15 paragraphs, comprehensive detail
        }

        return max_tokens_map.get(verbosity_preference, 1200)  # Default to medium


class AdaptiveResponseEngine:
    """Main engine coordinating all adaptive response features"""

    def __init__(self):
        self.sentiment_detector = SentimentDetector()
        self.topic_analyzer = TopicBridgeAnalyzer()
        self.verbosity_controller = VerbosityController()
        self.logger = logging.getLogger(f"{__name__}.AdaptiveResponseEngine")

    def analyze_and_enhance_prompt(
        self,
        user_id: int,
        query: str,
        base_prompt: str,
        session: Session,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        pre_extracted_topics: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze query and enhance prompt with personalization

        Args:
            user_id: User database ID
            query: User's query text
            base_prompt: Original prompt for LLM
            session: Database session
            conversation_history: Recent conversation turns for context
            pre_extracted_topics: Optional pre-extracted topics to avoid redundant LLM calls

        Returns:
            Tuple of (enhanced_prompt, analysis_metadata)
        """
        try:
            # Step 1: Sentiment analysis
            sentiment = self.sentiment_detector.analyze_query(query)
            self.logger.debug(f"Sentiment analysis: {sentiment}")

            # Step 2: Topic extraction and bridging
            # Use pre-extracted topics if provided, otherwise extract
            if pre_extracted_topics is not None:
                current_topics = pre_extracted_topics
                self.logger.debug(f"Using pre-extracted topics: {current_topics}")
            else:
                current_topics = self.topic_analyzer.extract_topics(query)
                self.logger.debug(f"Extracted topics: {current_topics}")

            struggle_topics = self.topic_analyzer.get_struggle_topics(user_id, session)
            topic_connections = self.topic_analyzer.find_topic_connections(current_topics, struggle_topics)

            self.logger.debug(f"Current topics: {current_topics}")
            self.logger.debug(f"Struggle topics: {struggle_topics}")
            self.logger.debug(f"Connections: {topic_connections}")

            # Step 3: Get verbosity preference from DB
            verbosity_preference = self.verbosity_controller.get_verbosity_preference(
                user_id, session, sentiment
            )
            self.logger.debug(f"Verbosity preference: {verbosity_preference}")

            # Step 4: Get user profile
            user = session.query(User).filter(User.id == user_id).first()
            profile = session.query(PersonalizationProfile).filter(
                PersonalizationProfile.user_id == user_id
            ).first()

            user_level = user.difficulty_preference if user else "intermediate"
            learning_style = user.learning_style if user else "adaptive"

            # Step 5: Get explanation style instructions
            try:
                style_instructions, detected_style = explanation_style_engine.generate_style_based_instructions(
                    user_id, query, session, user_level
                )
                self.logger.debug(f"Explanation style: {detected_style}")
            except Exception as style_error:
                self.logger.warning(f"Failed to get explanation style: {style_error}")
                style_instructions = ""
                detected_style = learning_style

            # Step 6: Build enhanced prompt
            enhanced_prompt = self._build_enhanced_prompt(
                base_prompt=base_prompt,
                query=query,
                sentiment=sentiment,
                topic_connections=topic_connections,
                verbosity_preference=verbosity_preference,
                user_level=user_level,
                learning_style=learning_style,
                profile=profile,
                session=session,
                style_instructions=style_instructions,
                conversation_history=conversation_history
            )

            # Metadata for logging
            metadata = {
                'verbosity_preference': verbosity_preference,
                'max_tokens': self.verbosity_controller.get_max_tokens_for_verbosity(verbosity_preference),
                'sentiment': sentiment,
                'current_topics': current_topics,
                'topic_connections': topic_connections,
                'user_level': user_level,
                'learning_style': learning_style,
                'explanation_style': detected_style
            }

            return enhanced_prompt, metadata

        except Exception as e:
            self.logger.error(f"Error in analyze_and_enhance_prompt: {e}", exc_info=True)
            # Return original prompt on error
            return base_prompt, {}

    def _build_enhanced_prompt(
        self,
        base_prompt: str,
        query: str,
        sentiment: Dict[str, Any],
        topic_connections: List[Dict[str, Any]],
        verbosity_preference: str,
        user_level: str,
        learning_style: str,
        profile: Optional[PersonalizationProfile],
        session: Session,
        style_instructions: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build enhanced prompt with all personalization elements"""

        # Start with conversation history if available
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            conversation_context = "CONVERSATION HISTORY:\n"
            for turn in conversation_history:
                conversation_context += f"User: {turn['query']}\n"
                # Truncate long responses to save tokens
                response_text = turn['response'][:300] + "..." if len(turn['response']) > 300 else turn['response']
                conversation_context += f"Assistant: {response_text}\n\n"
            conversation_context += "---\n\n"

        # Start with user profile section
        # CRITICAL FIX: Handle None values properly for numeric formatting
        # Use explicit None check instead of truthiness check to avoid treating 0.0 as False
        if profile and profile.question_complexity_level is not None:
            complexity_level = profile.question_complexity_level
        else:
            complexity_level = 0.5

        if profile and profile.learning_pace is not None:
            learning_pace = profile.learning_pace
        else:
            learning_pace = 'medium'

        # Safely handle None values for string formatting
        user_level_str = user_level.title() if user_level else 'Intermediate'
        learning_style_str = learning_style.title() if learning_style else 'Adaptive'
        verbosity_str = verbosity_preference.title() if verbosity_preference else 'Medium'
        pace_str = learning_pace.title() if learning_pace else 'Medium'

        # Final safety check: ensure complexity_level is a valid number before formatting
        if complexity_level is None or not isinstance(complexity_level, (int, float)):
            complexity_level = 0.5

        profile_section = f"""User Learning Profile:
- Mastery Level: {user_level_str}
- Learning Style: {learning_style_str}
- Response Length Preference: {verbosity_str}
- Question Complexity Level: {complexity_level:.1f}/1.0
- Learning Pace: {pace_str}"""

        if profile and profile.preferred_subjects:
            profile_section += f"\n- Familiar Topics: {', '.join(profile.preferred_subjects[:3])}"

        # Add complexity-based instructions
        if complexity_level < 0.4:
            complexity_instructions = """
Adaptation for Simple Questions:
- User typically asks foundational questions
- Start with absolute basics and build up
- Define all terms clearly before using them
- Avoid assuming prior knowledge"""
        elif complexity_level > 0.7:
            complexity_instructions = """
Adaptation for Complex Questions:
- User asks advanced, technical questions
- Skip basic definitions unless explicitly requested
- Include edge cases and nuances
- Reference related advanced concepts freely"""
        else:
            complexity_instructions = ""

        # Add pace-based instructions
        if learning_pace == 'fast':
            pace_instructions = """
Adaptation for Fast Learner:
- User is learning intensively
- Pack more information per response
- Suggest related topics to explore next
- Assume quick comprehension"""
        elif learning_pace == 'slow':
            pace_instructions = """
Adaptation for Casual Learner:
- User learns at a relaxed pace
- Don't overwhelm with too much at once
- Focus on retention and clarity over breadth
- Be patient with repetition if needed"""
        else:
            pace_instructions = ""

        # Add sentiment-aware instructions (enhanced with DB patterns)
        if sentiment['is_confused']:
            # Get confusion pattern from database for context
            confusion_pattern = 'none'
            if profile:
                try:
                    confusion_pattern = self.sentiment_detector.get_recent_confusion_pattern(
                        profile.user_id,
                        session,
                        lookback_hours=24
                    )
                except Exception as e:
                    self.logger.warning(f"Could not retrieve confusion pattern: {e}")
                    confusion_pattern = 'none'

            # Adjust tone based on confusion pattern
            if confusion_pattern == 'persistent':
                tone_instructions = """User Has Persistent Confusion Pattern - Extra Support Needed:
- CRITICAL: User has been confused repeatedly. Take extra care.
- Start from absolute fundamentals
- Use multiple analogies and examples
- Check understanding frequently
- Consider suggesting prerequisite topics
- Be very encouraging and supportive"""
            elif confusion_pattern == 'frequent':
                tone_instructions = """User Shows Frequent Confusion - Provide Clear Support:
- User has been confused several times recently
- Break down into smaller, digestible steps
- Use clear analogies
- Offer alternative explanations
- Be patient and encouraging"""
            else:
                tone_instructions = """User Appears Confused - Adjust Response:
- Acknowledge: "Let me break this down step-by-step..."
- Use simpler language and clear analogies
- Offer alternative explanations
- Be encouraging and patient"""
        else:
            tone_instructions = ""

        # Add topic bridging if connections found
        if topic_connections:
            connection = topic_connections[0]  # Use most relevant (highest similarity)
            related_topic = connection.get('related_topic', 'unknown topic')
            current_topic = connection.get('current_topic', 'current topic')
            query_count = connection.get('query_count', 0)
            connection_type = connection.get('connection_type', 'related')
            similarity_score = connection.get('similarity_score', 0)

            # Create appropriate bridging message based on connection strength
            if connection_type == 'strongly_related':
                bridging_hint = f"These topics are closely related. Consider reviewing {related_topic} fundamentals."
            elif connection_type == 'related':
                bridging_hint = f"This connects to {related_topic} concepts you've studied before."
            else:  # weakly_related
                bridging_hint = f"There may be some connection to {related_topic}."

            bridging_section = f"""Context-Aware Bridging:
- Current topic: {current_topic}
- User previously struggled with related topic: {related_topic} ({query_count} queries)
- Connection strength: {connection_type} (similarity: {similarity_score:.0%})
- INSTRUCTION: {bridging_hint}
  - If highly relevant, briefly acknowledge: "This builds on {related_topic} concepts..."
  - If user struggled significantly, review fundamentals before diving into advanced concepts
  - Be gentle and supportive given their previous struggles
"""
        else:
            bridging_section = ""

        # Add explanation style instructions (if provided)
        if style_instructions:
            explanation_style_section = f"\n{style_instructions}\n"
        else:
            # Fallback to verbosity instructions only
            explanation_style_section = self.verbosity_controller.generate_verbosity_instructions(
                verbosity_preference, user_level
            )

        # Combine all sections
        enhanced_prompt = f"""{conversation_context}{profile_section}

{tone_instructions}

{bridging_section}

{explanation_style_section}

{complexity_instructions}

{pace_instructions}

User Query: {query}

{base_prompt}"""

        return enhanced_prompt


# Global instance
adaptive_response_engine = AdaptiveResponseEngine()
