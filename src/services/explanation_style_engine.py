"""
Explanation Style Engine for Study Helper Agent

This module identifies and adapts to different learning styles:
1. Example-Driven - Prefers concrete examples before theory
2. Analogy-Driven - Learns best through metaphors and comparisons
3. Socratic - Responds well to guided questions and discovery
4. Theory-First - Prefers formal definitions before examples

Uses interaction pattern analysis and query language to classify users
into learning style clusters.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from collections import Counter

from sqlalchemy import desc, and_
from sqlalchemy.orm import Session
import numpy as np

from config.database import db_manager
from config.settings import settings
from src.data.models import (
    User,
    UserInteraction,
    PersonalizationProfile
)

logger = logging.getLogger(__name__)


class LearningStyleClassifier:
    """Classify users into learning style categories based on interaction patterns"""

    # Query patterns that indicate learning style preferences
    EXAMPLE_DRIVEN_PATTERNS = [
        r'\bexample\b',
        r'\bshow\s+me\b',
        r'\bcan\s+you\s+give\b',
        r'\bdemonstrate\b',
        r'\bwalk\s+through\b',
        r'\bcode\s+sample\b',
        r'\breal\s+world\b',
        r'\bpractical\b',
        r'\bhow\s+do\s+I\b',
        r'\bhow\s+to\b'
    ]

    ANALOGY_DRIVEN_PATTERNS = [
        r'\blike\s+what\b',
        r'\bsimilar\s+to\b',
        r'\bcompare\b',
        r'\banalogy\b',
        r'\bthink\s+of\s+it\s+(as|like)\b',
        r'\bin\s+simple\s+terms\b',
        r'\brelate\s+to\b',
        r'\bmetaphor\b',
        r'\bimagine\b',
        r'\breal\s+life\b'
    ]

    SOCRATIC_PATTERNS = [
        r'\bwhy\s+(does|is|do)\b',
        r'\bwhat\s+if\b',
        r'\bhow\s+come\b',
        r'\bhelp\s+me\s+(understand|figure)\b',
        r'\bguide\s+me\b',
        r'\bwalk\s+me\s+through\b',
        r'\bstep\s+by\s+step\b',
        r'\bwhat\s+happens\s+when\b',
        r'\bcan\s+you\s+ask\b',
        r'\bquestion\s+me\b',
        r'\bhow\s+would\b'
    ]

    THEORY_FIRST_PATTERNS = [
        r'\bdefine\b',
        r'\b(what\s*is|what\'s)\s+the\s+definition\b',
        r'\bformal\s+definition\b',
        r'\bformally\b',
        r'\btheory\b',
        r'\bprinciple\b',
        r'\bconcept\b',
        r'\bexplain\s+the\s+theory\b',
        r'\bwhat\s+does\s+.+\s+mean\b',
        r'\bterminology\b',
        r'\btechnical\b'
    ]

    def __init__(self):
        self.example_regex = [re.compile(p, re.IGNORECASE) for p in self.EXAMPLE_DRIVEN_PATTERNS]
        self.analogy_regex = [re.compile(p, re.IGNORECASE) for p in self.ANALOGY_DRIVEN_PATTERNS]
        self.socratic_regex = [re.compile(p, re.IGNORECASE) for p in self.SOCRATIC_PATTERNS]
        self.theory_regex = [re.compile(p, re.IGNORECASE) for p in self.THEORY_FIRST_PATTERNS]

    def analyze_query_style(self, query: str) -> Dict[str, float]:
        """
        Analyze a single query for learning style indicators

        Returns:
            Dict with scores for each style (0-1)
        """
        query_lower = query.lower()

        # Count matches for each style
        example_matches = sum(1 for p in self.example_regex if p.search(query))
        analogy_matches = sum(1 for p in self.analogy_regex if p.search(query))
        socratic_matches = sum(1 for p in self.socratic_regex if p.search(query))
        theory_matches = sum(1 for p in self.theory_regex if p.search(query))

        # Calculate scores (normalized by pattern count)
        total_patterns = len(self.example_regex)

        return {
            'example_driven': example_matches / len(self.example_regex),
            'analogy_driven': analogy_matches / len(self.analogy_regex),
            'socratic': socratic_matches / len(self.socratic_regex),
            'theory_first': theory_matches / len(self.theory_regex)
        }

    def classify_user_style(
        self,
        user_id: int,
        session: Session,
        lookback_interactions: int = 30
    ) -> Tuple[str, Dict[str, float]]:
        """
        Classify user's learning style based on interaction history

        Args:
            user_id: User database ID
            session: Database session
            lookback_interactions: Number of recent interactions to analyze

        Returns:
            Tuple of (primary_style, style_scores)
        """
        try:
            # Get recent interactions
            interactions = session.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).order_by(desc(UserInteraction.created_at)).limit(lookback_interactions).all()

            if len(interactions) < 5:
                # Not enough data, return adaptive
                return 'adaptive', {
                    'example_driven': 0.25,
                    'analogy_driven': 0.25,
                    'socratic': 0.25,
                    'theory_first': 0.25
                }

            # Aggregate scores across all interactions
            style_scores = {
                'example_driven': 0.0,
                'analogy_driven': 0.0,
                'socratic': 0.0,
                'theory_first': 0.0
            }

            for interaction in interactions:
                query_scores = self.analyze_query_style(interaction.query_text)
                for style, score in query_scores.items():
                    style_scores[style] += score

            # Normalize by number of interactions
            for style in style_scores:
                style_scores[style] /= len(interactions)

            # Additional heuristics based on interaction patterns
            style_scores = self._apply_behavioral_heuristics(
                style_scores, interactions
            )

            # Apply rating influence (amplify/dampen based on user satisfaction)
            style_scores = self._apply_rating_influence(
                style_scores, interactions
            )

            # Determine primary style (highest score)
            if max(style_scores.values()) < 0.15:
                # No clear preference
                primary_style = 'adaptive'
            else:
                primary_style = max(style_scores, key=style_scores.get)

            logger.info(f"User {user_id} classified as {primary_style} "
                       f"(scores: {style_scores})")

            return primary_style, style_scores

        except Exception as e:
            logger.error(f"Error classifying user style: {e}", exc_info=True)
            return 'adaptive', {
                'example_driven': 0.25,
                'analogy_driven': 0.25,
                'socratic': 0.25,
                'theory_first': 0.25
            }

    def _apply_behavioral_heuristics(
        self,
        style_scores: Dict[str, float],
        interactions: List[UserInteraction]
    ) -> Dict[str, float]:
        """
        Apply behavioral heuristics to refine style classification

        Analyzes:
        - Query length (theory-first users ask longer, more formal queries)
        - Follow-up patterns (socratic learners ask many why/how questions)
        - Response satisfaction (via feedback)
        """
        # Calculate average query length
        query_lengths = [len(i.query_text.split()) for i in interactions]
        avg_length = sum(query_lengths) / len(query_lengths)

        # Longer queries suggest theory-first preference
        if avg_length > 15:
            style_scores['theory_first'] += 0.1
        elif avg_length < 8:
            style_scores['example_driven'] += 0.1

        # Count "why" and "how" questions (socratic indicator)
        why_how_count = sum(
            1 for i in interactions
            if re.search(r'\b(why|how)\b', i.query_text, re.IGNORECASE)
        )
        why_how_ratio = why_how_count / len(interactions)
        if why_how_ratio > 0.4:
            style_scores['socratic'] += 0.15

        # Check for analogies in user queries (analogy-driven indicator)
        analogy_count = sum(
            1 for i in interactions
            if re.search(r'\b(like|similar|compare|analogy)\b', i.query_text, re.IGNORECASE)
        )
        if analogy_count > len(interactions) * 0.2:
            style_scores['analogy_driven'] += 0.1

        return style_scores

    def _apply_rating_influence(
        self,
        style_scores: Dict[str, float],
        interactions: List[UserInteraction]
    ) -> Dict[str, float]:
        """
        Apply rating influence to style scores (amplify/dampen based on satisfaction)

        Strategy:
        - For each style detected in queries, check if user rated those interactions
        - Calculate average rating when that style was detected
        - Apply multiplier: high ratings boost score, low ratings reduce it
        - Multiplier range: 0.7x (poor ratings) to 1.3x (excellent ratings)

        This ensures ratings INFLUENCE but don't DICTATE the classification.
        """
        # Group interactions by detected query style and collect ratings
        style_ratings = {
            'example_driven': [],
            'analogy_driven': [],
            'socratic': [],
            'theory_first': []
        }

        for interaction in interactions:
            # Skip unrated interactions
            if not interaction.user_rating or interaction.user_rating < 1:
                continue

            # Analyze which style(s) this query matches
            query_style = self.analyze_query_style(interaction.query_text)

            # Add rating to each style's list (weighted by style score)
            for style, score in query_style.items():
                if score > 0.1:  # Only consider styles with meaningful matches
                    style_ratings[style].append(interaction.user_rating)

        # Calculate rating multipliers for each style
        rating_multipliers = {}
        for style, ratings in style_ratings.items():
            if len(ratings) >= 3:  # Minimum 3 rated samples for reliability
                avg_rating = sum(ratings) / len(ratings)

                # Map rating (1-5) to multiplier (0.7-1.3)
                # 5 stars → 1.3x boost
                # 4 stars → 1.15x boost
                # 3 stars → 1.0x (neutral)
                # 2 stars → 0.85x reduction
                # 1 star → 0.7x reduction
                multiplier = 0.7 + (avg_rating - 1) * 0.15
                rating_multipliers[style] = multiplier

                logger.debug(f"Style '{style}': {len(ratings)} rated interactions, "
                           f"avg rating {avg_rating:.2f}, multiplier {multiplier:.2f}x")
            else:
                # Not enough data, use neutral multiplier
                rating_multipliers[style] = 1.0

        # Apply multipliers to style scores
        adjusted_scores = {}
        for style, score in style_scores.items():
            multiplier = rating_multipliers.get(style, 1.0)
            adjusted_scores[style] = score * multiplier

        logger.info(f"Rating influence applied. Multipliers: {rating_multipliers}")

        return adjusted_scores


class StyleBasedPromptGenerator:
    """Generate style-specific prompts for different learning types"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StyleBasedPromptGenerator")

    def generate_style_instructions(
        self,
        learning_style: str,
        topic: str = "the concept",
        user_level: str = "intermediate"
    ) -> str:
        """
        Generate prompt instructions based on learning style

        Args:
            learning_style: example_driven, analogy_driven, socratic, theory_first
            topic: The topic being explained
            user_level: beginner/intermediate/advanced

        Returns:
            Style-specific instructions for LLM
        """
        if learning_style == 'example_driven':
            return self._example_driven_template(topic, user_level)
        elif learning_style == 'analogy_driven':
            return self._analogy_driven_template(topic, user_level)
        elif learning_style == 'socratic':
            return self._socratic_template(topic, user_level)
        elif learning_style == 'theory_first':
            return self._theory_first_template(topic, user_level)
        else:
            # Adaptive/default
            return self._adaptive_template(topic, user_level)

    def _example_driven_template(self, topic: str, user_level: str) -> str:
        """Template for example-driven learners"""
        return f"""Learning Style: Example-Driven (prefers concrete examples first)

INSTRUCTIONS FOR EXPLAINING {topic.upper()}:
1. START with a simple, concrete example
2. Walk through the example step-by-step
3. Show the code/process/calculation explicitly
4. THEN explain the general principle or theory
5. Provide 1-2 additional examples showing variations
6. Use practical, real-world scenarios
7. Avoid abstract theory until examples are clear

Structure:
→ Example #1 (simple case)
→ Step-by-step breakdown
→ General principle extraction
→ Example #2 (slightly more complex)
→ Summary of pattern

User Level: {user_level.title()}"""

    def _analogy_driven_template(self, topic: str, user_level: str) -> str:
        """Template for analogy-driven learners"""
        return f"""Learning Style: Analogy-Driven (learns best through comparisons and metaphors)

INSTRUCTIONS FOR EXPLAINING {topic.upper()}:
1. START with a relatable analogy or metaphor
2. Draw clear parallels between the analogy and the concept
3. Use everyday objects/situations for comparison
4. Highlight similarities and differences explicitly
5. Build understanding by extending the analogy
6. Then provide the technical details
7. Connect back to the analogy periodically

Structure:
→ Analogy introduction (e.g., "Think of it like...")
→ Map analogy to concept explicitly
→ Explore the analogy in depth
→ Technical explanation with analogy references
→ Summary: "Just like [analogy], [concept] works by..."

User Level: {user_level.title()}
Use analogies appropriate for {user_level} understanding."""

    def _socratic_template(self, topic: str, user_level: str) -> str:
        """Template for socratic learners"""
        return f"""Learning Style: Socratic (learns through guided questions and discovery)

INSTRUCTIONS FOR EXPLAINING {topic.upper()}:
1. START with a thought-provoking question
2. Guide discovery through a series of questions
3. Let the student "discover" the concept through reasoning
4. Use questions like:
   - "What do you think happens when...?"
   - "Why might this approach work/fail?"
   - "How would you solve...?"
5. Provide hints and scaffolding, not direct answers initially
6. Build understanding progressively through questioning
7. Conclude with synthesis of discoveries

Structure:
→ Opening question to activate thinking
→ Series of guided questions (3-5)
→ Progressive hints if needed
→ Student-discovered insights
→ Formal explanation connecting discoveries
→ Challenge question for deeper thinking

User Level: {user_level.title()}
Ask questions appropriate for {user_level} knowledge."""

    def _theory_first_template(self, topic: str, user_level: str) -> str:
        """Template for theory-first learners"""
        return f"""Learning Style: Theory-First (prefers formal definitions before examples)

INSTRUCTIONS FOR EXPLAINING {topic.upper()}:
1. START with formal definition and terminology
2. Explain the theoretical foundation
3. State key principles and properties
4. Provide mathematical/logical formulation if applicable
5. THEN illustrate with examples
6. Show how examples instantiate the theory
7. Use precise, technical language

Structure:
→ Formal definition
→ Theoretical framework/principles
→ Key properties and characteristics
→ Mathematical notation/formulation (if relevant)
→ Example(s) demonstrating the theory
→ Edge cases and constraints
→ Formal summary

User Level: {user_level.title()}
Use appropriate technical depth for {user_level} level."""

    def _adaptive_template(self, topic: str, user_level: str) -> str:
        """Template for adaptive/default style"""
        return f"""Learning Style: Adaptive (balanced approach)

INSTRUCTIONS FOR EXPLAINING {topic.upper()}:
1. Brief definition to establish context
2. Simple example to illustrate
3. Explain key principles
4. Additional examples or analogies as needed
5. Balance theory and practice

User Level: {user_level.title()}"""

    def generate_prompt_template(
        self,
        learning_style: str,
        topic: str = "the concept",
        user_level: str = "intermediate"
    ) -> str:
        """
        Alias for generate_style_instructions for backward compatibility

        Args:
            learning_style: example_driven, analogy_driven, socratic, theory_first
            topic: The topic being explained
            user_level: beginner/intermediate/advanced

        Returns:
            Style-specific instructions for LLM
        """
        return self.generate_style_instructions(learning_style, topic, user_level)


class ExplanationStyleEngine:
    """Main engine for managing explanation style customization"""

    def __init__(self):
        self.classifier = LearningStyleClassifier()
        self.prompt_generator = StyleBasedPromptGenerator()
        self.logger = logging.getLogger(f"{__name__}.ExplanationStyleEngine")

    def get_user_learning_style(
        self,
        user_id: int,
        session: Session,
        force_reclassify: bool = False
    ) -> Tuple[str, Dict[str, float]]:
        """
        Get user's learning style with time-based reclassification

        Classification occurs when:
        - force_reclassify=True
        - User has never been classified (last_style_classification is None)
        - 12+ hours have passed since last classification

        Args:
            user_id: User database ID
            session: Database session
            force_reclassify: Force re-classification regardless of time

        Returns:
            Tuple of (primary_style, style_scores)
        """
        try:
            user = session.query(User).filter(User.id == user_id).first()

            if not user:
                self.logger.warning(f"User {user_id} not found")
                return 'adaptive', {
                    'example_driven': 0.25,
                    'analogy_driven': 0.25,
                    'socratic': 0.25,
                    'theory_first': 0.25
                }

            # Determine if reclassification is needed
            now_utc = datetime.now(timezone.utc)
            needs_reclassification = (
                force_reclassify
                or user.last_style_classification is None
                or (now_utc - user.last_style_classification) >= timedelta(hours=12)
            )

            if needs_reclassification:
                # Reclassify from last 30 interactions
                self.logger.info(f"Reclassifying learning style for user {user_id}...")
                primary_style, scores = self.classifier.classify_user_style(user_id, session)

                # Update database with new style and timestamp
                self._update_style_in_profile(user_id, primary_style, scores, session)
                user.last_style_classification = now_utc
                session.commit()

                self.logger.info(f"User {user_id} reclassified as {primary_style}")
                return primary_style, scores
            else:
                # Use existing database value (not older than 12 hours)
                time_since_classification = now_utc - user.last_style_classification
                hours_since = time_since_classification.total_seconds() / 3600

                self.logger.debug(
                    f"Using existing style for user {user_id}: {user.learning_style} "
                    f"(classified {hours_since:.1f}h ago)"
                )

                # Get stored scores from profile
                profile = session.query(PersonalizationProfile).filter(
                    PersonalizationProfile.user_id == user_id
                ).first()

                if profile and profile.feature_vector and 'style_scores' in profile.feature_vector:
                    scores = profile.feature_vector['style_scores']
                else:
                    # Estimate scores based on current style
                    scores = {
                        'example_driven': 0.25,
                        'analogy_driven': 0.25,
                        'socratic': 0.25,
                        'theory_first': 0.25
                    }
                    if user.learning_style != 'adaptive':
                        scores[user.learning_style] = 0.7

                return user.learning_style, scores

        except Exception as e:
            self.logger.error(f"Error getting user learning style: {e}", exc_info=True)
            return 'adaptive', {
                'example_driven': 0.25,
                'analogy_driven': 0.25,
                'socratic': 0.25,
                'theory_first': 0.25
            }

    def _update_style_in_profile(
        self,
        user_id: int,
        primary_style: str,
        scores: Dict[str, float],
        session: Session
    ):
        """Update user's learning style in database"""
        try:
            # Update User table
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.learning_style = primary_style

            # Update PersonalizationProfile
            profile = session.query(PersonalizationProfile).filter(
                PersonalizationProfile.user_id == user_id
            ).first()

            if profile:
                if not profile.feature_vector:
                    profile.feature_vector = {}
                profile.feature_vector['style_scores'] = scores
                profile.feature_vector['primary_style'] = primary_style
                profile.last_model_update = datetime.utcnow()

            session.commit()
            self.logger.info(f"Updated learning style for user {user_id}: {primary_style}")

        except Exception as e:
            self.logger.error(f"Error updating style in profile: {e}")
            session.rollback()

    def generate_style_based_instructions(
        self,
        user_id: int,
        query: str,
        session: Session,
        user_level: str = "intermediate"
    ) -> Tuple[str, str]:
        """
        Generate style-specific instructions for LLM prompt

        Args:
            user_id: User database ID
            query: User's query
            session: Database session
            user_level: User's mastery level

        Returns:
            Tuple of (instructions, detected_style)
        """
        try:
            # Get user's learning style
            primary_style, scores = self.get_user_learning_style(user_id, session)

            # Check if current query suggests a specific style override
            query_style_scores = self.classifier.analyze_query_style(query)
            max_query_score = max(query_style_scores.values())

            # If query strongly indicates a style (>0.2), consider temporary override
            if max_query_score > 0.2:
                query_suggested_style = max(query_style_scores, key=query_style_scores.get)
                self.logger.debug(f"Query suggests {query_suggested_style} style (score: {max_query_score:.2f})")

                # Use query style if significantly different from user's primary style
                # Lower threshold to 0.3 for override to be more responsive to query intent
                if query_suggested_style != primary_style and max_query_score > 0.3:
                    style_to_use = query_suggested_style
                    self.logger.info(f"Overriding user style {primary_style} with query style {query_suggested_style}")
                else:
                    style_to_use = primary_style
            else:
                style_to_use = primary_style

            # Extract topic from query (simple heuristic)
            topic = self._extract_topic(query)

            # Generate instructions
            instructions = self.prompt_generator.generate_style_instructions(
                style_to_use, topic, user_level
            )

            return instructions, style_to_use

        except Exception as e:
            self.logger.error(f"Error generating style-based instructions: {e}", exc_info=True)
            return self.prompt_generator.generate_style_instructions('adaptive', 'the concept', user_level), 'adaptive'

    def _extract_topic(self, query: str) -> str:
        """
        Extract the main topic from query (simple heuristic)

        Returns a more specific topic name if found, otherwise "the concept"
        """
        # Common CS topics
        topics = [
            'recursion', 'sorting', 'binary search', 'data structures',
            'linked lists', 'arrays', 'trees', 'graphs', 'algorithms',
            'dynamic programming', 'object oriented programming', 'oop',
            'databases', 'sql', 'stacks', 'queues', 'hash tables',
            'pointers', 'memory management', 'big o notation'
        ]

        query_lower = query.lower()
        for topic in topics:
            if topic in query_lower:
                return topic

        return "the concept"


# Global instance
explanation_style_engine = ExplanationStyleEngine()
