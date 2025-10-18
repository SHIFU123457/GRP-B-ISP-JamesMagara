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
    QuizSession
)
from src.services.explanation_style_engine import explanation_style_engine

logger = logging.getLogger(__name__)


class SentimentDetector:
    """Detect user sentiment and confusion in queries"""

    # Frustration/confusion markers
    CONFUSION_PATTERNS = [
        r'\bstill\s+(don\'t|dont|do\s+not)\s+(get|understand)',
        r'\bconfused\b',
        r'\bmakes?\s+no\s+sense\b',
        r'\bdon\'t\s+understand\b',
        r'\bwhat\s+does\s+that\s+mean\b',
        r'\bi\'m\s+lost\b',
        r'\bthis\s+is\s+(hard|difficult|confusing)',
        r'\bhelp\s+me\b',
        r'\bcan\'t\s+figure\b'
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


class TopicBridgeAnalyzer:
    """Analyze topic relationships and user struggles"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TopicBridgeAnalyzer")
        self._llm_service = None  # Lazy load to avoid circular imports

    @property
    def llm_service(self):
        """Lazy load LLM service to avoid circular imports"""
        if self._llm_service is None:
            from src.services.llm_integration import LLMService
            self._llm_service = LLMService()
        return self._llm_service

    def extract_topics(self, query: str) -> List[str]:
        """
        Extract key topics from query using LLM-based analysis

        This method uses an LLM to intelligently identify the main topics/concepts
        being discussed in the user's query, regardless of specific keywords.

        Args:
            query: User's query text

        Returns:
            List of identified topics (e.g., ['regression', 'machine learning'])
        """
        try:
            # Use LLM to extract topics with a simpler, more direct prompt
            prompt = f"""Extract the main academic topics from this question. List 1-3 topics, one per line.

Question: {query}

Topics:"""

            # Get LLM response
            llm_response = self.llm_service.generate_response(
                query="Extract topics",
                context=prompt,
                user_preferences=None
            )

            # Log the raw response for debugging
            self.logger.debug(f"Raw LLM response for topic extraction: {llm_response[:200]}...")

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

                # Skip if it's too long (likely a sentence, not a topic)
                if len(cleaned) > 40:
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
                            if 2 <= len(part) <= 30:
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

            if topics:
                self.logger.info(f"LLM extracted topics from '{query[:50]}...': {topics}")
                return topics
            else:
                self.logger.warning(f"LLM returned no parseable topics for query: {query[:50]}...")
                self.logger.debug(f"Full LLM response: {llm_response}")
                # Fallback to simple extraction
                return self._fallback_topic_extraction(query)

        except Exception as e:
            self.logger.warning(f"Error in LLM topic extraction: {e}. Using fallback method.")
            return self._fallback_topic_extraction(query)

    def _fallback_topic_extraction(self, query: str) -> List[str]:
        """Simple fallback method when LLM is unavailable"""
        query_lower = query.lower()

        # Remove common question words and stopwords
        question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'can',
                         'could', 'would', 'should', 'do', 'does', 'did', 'explain', 'tell', 'me'}
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'about', 'as', 'into', 'like', 'through'}

        # Extract significant words (longer than 3 chars, not stopwords)
        words = query_lower.split()
        significant_words = [
            word.strip('?.,!;:')
            for word in words
            if len(word) > 3
            and word not in question_words
            and word not in stopwords
            and word.isalpha()
        ]

        # Return up to 3 significant words as topics
        return significant_words[:3] if significant_words else ['general']

    def get_struggle_topics(self, user_id: int, session: Session, lookback_days: int = 14) -> List[Dict[str, Any]]:
        """
        Identify topics user has struggled with recently

        Returns list of dicts with:
            - topic: str
            - query_count: int
            - quiz_performance: float (if available)
            - last_asked: datetime
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        # Get recent interactions
        interactions = session.query(UserInteraction).filter(
            and_(
                UserInteraction.user_id == user_id,
                UserInteraction.created_at >= cutoff_date
            )
        ).all()

        # Extract topics and count frequency
        topic_queries = {}
        for interaction in interactions:
            topics = self.extract_topics(interaction.query_text)
            for topic in topics:
                if topic not in topic_queries:
                    topic_queries[topic] = {
                        'query_count': 0,
                        'last_asked': interaction.created_at,
                        'topic': topic
                    }
                topic_queries[topic]['query_count'] += 1
                if interaction.created_at > topic_queries[topic]['last_asked']:
                    topic_queries[topic]['last_asked'] = interaction.created_at

        # Filter to "struggle" topics (asked 2+ times)
        struggle_topics = [
            data for topic, data in topic_queries.items()
            if data['query_count'] >= 2
        ]

        # Sort by query count (desc)
        struggle_topics.sort(key=lambda x: x['query_count'], reverse=True)

        return struggle_topics

    def find_topic_connections(
        self,
        current_topics: List[str],
        struggle_topics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find connections between current query and past struggles

        Returns list of connections with:
            - related_topic: str
            - connection_type: str (prerequisite, related, builds_on)
            - query_count: int
        """
        # Topic relationship map (prerequisite knowledge)
        topic_relationships = {
            'sorting': ['arrays', 'algorithms', 'recursion'],  # sorting needs these
            'recursion': ['algorithms'],
            'data structures': ['arrays', 'object oriented'],
            'graphs': ['data structures', 'recursion'],
            'dynamic programming': ['recursion', 'algorithms'],
        }

        connections = []

        for current_topic in current_topics:
            # Check if current topic has prerequisites in struggle topics
            prerequisites = topic_relationships.get(current_topic, [])

            for struggle in struggle_topics:
                struggle_topic = struggle['topic']

                # Check if struggled topic is a prerequisite
                if struggle_topic in prerequisites:
                    connections.append({
                        'related_topic': struggle_topic,
                        'connection_type': 'prerequisite',
                        'query_count': struggle['query_count'],
                        'last_asked': struggle['last_asked']
                    })

        return connections


class VerbosityController:
    """Control response length and depth based on user preferences"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VerbosityController")

    def calculate_verbosity_score(
        self,
        user_id: int,
        session: Session,
        current_query_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate verbosity preference score (0-10)

        0-3: Concise
        4-6: Medium
        7-10: Detailed

        Based on:
        - Historical session duration patterns
        - Follow-up question frequency
        - Explicit query indicators
        - Profile preference
        """
        score = 5.0  # Default medium

        # Factor 1: Check current query for explicit preferences
        if current_query_analysis.get('wants_brevity'):
            score -= 3.0
        if current_query_analysis.get('wants_more_detail'):
            score += 2.0

        # Factor 2: Historical session duration
        recent_sessions = session.query(ConversationSession).filter(
            and_(
                ConversationSession.user_id == user_id,
                ConversationSession.is_active == False,
                ConversationSession.session_duration_minutes > 0
            )
        ).order_by(desc(ConversationSession.ended_at)).limit(5).all()

        if recent_sessions:
            avg_duration = sum(s.session_duration_minutes for s in recent_sessions) / len(recent_sessions)
            # Longer sessions suggest preference for detailed content
            if avg_duration > 15:
                score += 1.0
            elif avg_duration < 5:
                score -= 1.0

        # Factor 3: Follow-up question frequency
        recent_interactions = session.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        ).order_by(desc(UserInteraction.created_at)).limit(20).all()

        if len(recent_interactions) >= 5:
            # Look for follow-up patterns (queries within 5 minutes)
            followup_count = 0
            for i in range(1, len(recent_interactions)):
                time_diff = (recent_interactions[i-1].created_at - recent_interactions[i].created_at).total_seconds()
                if time_diff < 300:  # 5 minutes
                    followup_count += 1

            followup_ratio = followup_count / len(recent_interactions)
            if followup_ratio > 0.3:  # Many follow-ups
                score -= 1.5  # Prefer concise initial answers
            elif followup_ratio < 0.1:  # Few follow-ups
                score += 1.0  # Prefer comprehensive initial answers

        # Factor 4: Profile preference (if exists)
        profile = session.query(PersonalizationProfile).filter(
            PersonalizationProfile.user_id == user_id
        ).first()

        if profile and profile.preferred_response_length:
            length_map = {'short': -2.0, 'medium': 0.0, 'long': 2.0}
            score += length_map.get(profile.preferred_response_length, 0.0)

        # Clamp to 0-10 range
        return max(0.0, min(10.0, score))

    def generate_verbosity_instructions(self, verbosity_score: float, user_level: str = "intermediate") -> str:
        """
        Generate LLM prompt instructions based on verbosity score

        Args:
            verbosity_score: 0-10 scale
            user_level: beginner/intermediate/advanced

        Returns:
            String instructions to prepend to LLM prompt
        """
        if verbosity_score <= 3:
            # Concise
            instructions = f"""Response Style Instructions:
- Keep response CONCISE (2-4 sentences maximum)
- Focus only on the core concept
- Skip detailed examples unless essential
- Assume {user_level} understanding
- Use bullet points for clarity"""

        elif verbosity_score <= 6:
            # Medium
            instructions = f"""Response Style Instructions:
- Provide a BALANCED explanation (4-6 sentences)
- Include one clear example
- Cover main points without excessive detail
- Tailor to {user_level} level
- Use structured formatting"""

        else:
            # Detailed
            instructions = f"""Response Style Instructions:
- Provide a COMPREHENSIVE explanation (6-10 sentences)
- Include multiple examples and edge cases
- Break down complex concepts step-by-step
- Add context and deeper insights
- Tailor to {user_level} level
- Use analogies where helpful"""

        return instructions


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
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze query and enhance prompt with personalization

        Args:
            user_id: User database ID
            query: User's query text
            base_prompt: Original prompt for LLM
            session: Database session
            conversation_history: Recent conversation turns for context

        Returns:
            Tuple of (enhanced_prompt, analysis_metadata)
        """
        try:
            # Step 1: Sentiment analysis
            sentiment = self.sentiment_detector.analyze_query(query)
            self.logger.debug(f"Sentiment analysis: {sentiment}")

            # Step 2: Topic extraction and bridging
            current_topics = self.topic_analyzer.extract_topics(query)
            struggle_topics = self.topic_analyzer.get_struggle_topics(user_id, session)
            topic_connections = self.topic_analyzer.find_topic_connections(current_topics, struggle_topics)

            self.logger.debug(f"Current topics: {current_topics}")
            self.logger.debug(f"Struggle topics: {struggle_topics}")
            self.logger.debug(f"Connections: {topic_connections}")

            # Step 3: Calculate verbosity preference
            verbosity_score = self.verbosity_controller.calculate_verbosity_score(
                user_id, session, sentiment
            )
            self.logger.debug(f"Verbosity score: {verbosity_score}/10")

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
                verbosity_score=verbosity_score,
                user_level=user_level,
                learning_style=learning_style,
                profile=profile,
                style_instructions=style_instructions,
                conversation_history=conversation_history
            )

            # Metadata for logging
            metadata = {
                'verbosity_score': verbosity_score,
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
        verbosity_score: float,
        user_level: str,
        learning_style: str,
        profile: Optional[PersonalizationProfile],
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
        profile_section = f"""User Learning Profile:
- Mastery Level: {user_level.title()}
- Learning Style: {learning_style.title()}
- Verbosity Preference: {verbosity_score:.1f}/10"""

        if profile and profile.preferred_subjects:
            profile_section += f"\n- Familiar Topics: {', '.join(profile.preferred_subjects[:3])}"

        # Add sentiment-aware instructions
        if sentiment['is_confused']:
            tone_instructions = """User Appears Confused - Adjust Response:
- Acknowledge: "Let me break this down step-by-step..."
- Use simpler language and clear analogies
- Offer alternative explanations
- Be encouraging and patient"""
        else:
            tone_instructions = ""

        # Add topic bridging if connections found
        if topic_connections:
            connection = topic_connections[0]  # Use most relevant
            related_topic = connection['related_topic']
            bridging_section = f"""Context-Aware Bridging:
- User previously struggled with: {related_topic} ({connection['query_count']} queries)
- Connection type: {connection['connection_type']}
- INSTRUCTION: Briefly acknowledge this connection if relevant
  Example: "This builds on {related_topic} concepts we discussed before..."
"""
        else:
            bridging_section = ""

        # Add explanation style instructions (if provided)
        if style_instructions:
            explanation_style_section = f"\n{style_instructions}\n"
        else:
            # Fallback to verbosity instructions only
            explanation_style_section = self.verbosity_controller.generate_verbosity_instructions(
                verbosity_score, user_level
            )

        # Combine all sections
        enhanced_prompt = f"""{conversation_context}{profile_section}

{tone_instructions}

{bridging_section}

{explanation_style_section}

User Query: {query}

{base_prompt}"""

        return enhanced_prompt


# Global instance
adaptive_response_engine = AdaptiveResponseEngine()
