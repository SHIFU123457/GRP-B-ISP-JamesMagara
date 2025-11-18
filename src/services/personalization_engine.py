"""
Personalization Engine for Study Helper Agent

This module provides:
1. Session Management - Track conversation sessions with timeout support
2. Learning Analytics - Analyze user behavior and preferences
3. Adaptive Responses - Personalize responses based on user history
4. Context Continuity - Maintain conversation context across interactions
"""

import logging
import uuid
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from collections import Counter
import json

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import desc, and_, func

from config.database import db_manager
from config.settings import settings
from src.data.models import (
    User,
    UserInteraction,
    PersonalizationProfile,
    ConversationSession,
    Course,
    Document
)

logger = logging.getLogger(__name__)


class SessionManager:
    """Manage conversation sessions with automatic timeout"""

    # Session timeout in minutes (configurable)
    SESSION_TIMEOUT_MINUTES = int(getattr(settings, 'SESSION_TIMEOUT_MINUTES', 30))

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SessionManager")

    def get_or_create_session(self, user_id: int, session: Session) -> ConversationSession:
        """
        Get active session for user or create a new one if:
        - No active session exists
        - Last session timed out

        Args:
            user_id: User database ID
            session: Database session

        Returns:
            ConversationSession object
        """
        try:
            # Find most recent active session
            active_session = session.query(ConversationSession).filter(
                and_(
                    ConversationSession.user_id == user_id,
                    ConversationSession.is_active == True
                )
            ).order_by(desc(ConversationSession.last_activity_at)).first()

            # Check if session exists and hasn't timed out
            if active_session:
                time_since_activity = datetime.utcnow() - active_session.last_activity_at.replace(tzinfo=None)

                if time_since_activity.total_seconds() / 60 < self.SESSION_TIMEOUT_MINUTES:
                    # Session is still active
                    self.logger.debug(f"Resuming active session {active_session.session_id} for user {user_id}")
                    return active_session
                else:
                    # Session timed out - close it
                    self.logger.info(f"Session {active_session.session_id} timed out after {time_since_activity}")
                    self._close_session(active_session, session)

            # Create new session
            new_session = self._create_new_session(user_id, session)
            self.logger.info(f"Created new session {new_session.session_id} for user {user_id}")
            return new_session

        except Exception as e:
            self.logger.error(f"Error in get_or_create_session: {e}", exc_info=True)
            raise

    def _create_new_session(self, user_id: int, session: Session) -> ConversationSession:
        """Create a new conversation session"""
        new_session = ConversationSession(
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            is_active=True,
            session_context={},
            message_count=0,
            questions_asked=0,
            commands_used=0,
            started_at=datetime.utcnow(),
            last_activity_at=datetime.utcnow()
        )

        session.add(new_session)
        session.commit()

        return new_session

    def _close_session(self, conv_session: ConversationSession, db_session: Session) -> None:
        """Close a conversation session and calculate metrics"""
        try:
            conv_session.is_active = False
            conv_session.ended_at = datetime.utcnow()

            # Calculate session duration
            if conv_session.started_at:
                duration = conv_session.ended_at - conv_session.started_at.replace(tzinfo=None)
                conv_session.session_duration_minutes = duration.total_seconds() / 60

            db_session.commit()
            self.logger.debug(f"Closed session {conv_session.session_id} after {conv_session.session_duration_minutes:.1f} minutes")

        except Exception as e:
            self.logger.error(f"Error closing session: {e}")
            db_session.rollback()

    def update_session_activity(
        self,
        conv_session: ConversationSession,
        db_session: Session,
        interaction_type: str = "message",
        context_updates: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update session with new activity

        Args:
            conv_session: Active conversation session
            db_session: Database session
            interaction_type: Type of interaction (message, command, question)
            context_updates: Updates to session context
        """
        try:
            conv_session.last_activity_at = datetime.utcnow()
            conv_session.message_count += 1

            if interaction_type == "question":
                conv_session.questions_asked += 1
            elif interaction_type == "command":
                conv_session.commands_used += 1

            # Update context if provided
            if context_updates:
                current_context = conv_session.session_context or {}
                current_context.update(context_updates)
                conv_session.session_context = current_context
                # Mark JSON field as modified for SQLAlchemy to track changes
                flag_modified(conv_session, 'session_context')

            db_session.commit()

        except Exception as e:
            self.logger.error(f"Error updating session activity: {e}")
            db_session.rollback()

    def get_session_context(self, conv_session: ConversationSession) -> Dict[str, Any]:
        """Get current session context"""
        return conv_session.session_context or {}

    def force_close_session(self, session_id: str) -> bool:
        """Manually close a session by ID"""
        try:
            with db_manager.get_session() as db_session:
                conv_session = db_session.query(ConversationSession).filter(
                    ConversationSession.session_id == session_id
                ).first()

                if conv_session and conv_session.is_active:
                    self._close_session(conv_session, db_session)
                    return True

                return False

        except Exception as e:
            self.logger.error(f"Error force closing session: {e}")
            return False

    def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """Clean up old inactive sessions"""
        try:
            with db_manager.get_session() as db_session:
                cutoff_date = datetime.utcnow() - timedelta(days=days_old)

                old_sessions = db_session.query(ConversationSession).filter(
                    and_(
                        ConversationSession.is_active == False,
                        ConversationSession.ended_at < cutoff_date
                    )
                ).all()

                count = len(old_sessions)

                for session in old_sessions:
                    db_session.delete(session)

                db_session.commit()
                self.logger.info(f"Cleaned up {count} old sessions")
                return count

        except Exception as e:
            self.logger.error(f"Error cleaning up old sessions: {e}")
            return 0

    def get_conversation_history(
        self,
        user_id: int,
        db_session: Session,
        limit: int = 5,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation history for context from specific session

        Uses session_id foreign key for accurate session-based filtering.

        Args:
            user_id: User database ID
            db_session: Database session
            limit: Number of recent interactions to retrieve
            session_id: Optional session ID to filter by (if not provided, uses active session)

        Returns:
            List of conversation turns with query and response
        """
        try:
            # Get the target session
            if not session_id:
                active_session = db_session.query(ConversationSession).filter(
                    and_(
                        ConversationSession.user_id == user_id,
                        ConversationSession.is_active == True
                    )
                ).order_by(desc(ConversationSession.last_activity_at)).first()

                if not active_session:
                    self.logger.debug(f"No active session found for user {user_id}")
                    return []

                session_id = active_session.session_id
            else:
                # Verify session exists
                active_session = db_session.query(ConversationSession).filter(
                    ConversationSession.session_id == session_id
                ).first()

                if not active_session:
                    self.logger.warning(f"Session {session_id} not found")
                    return []

            # Query interactions by session_id (much more accurate than time-based filtering)
            recent_interactions = db_session.query(UserInteraction).filter(
                and_(
                    UserInteraction.user_id == user_id,
                    UserInteraction.session_id == session_id
                )
            ).order_by(desc(UserInteraction.created_at)).limit(limit).all()

            # Reverse to get chronological order (oldest to newest)
            recent_interactions.reverse()

            history = []
            for interaction in recent_interactions:
                history.append({
                    'query': interaction.query_text,
                    'response': interaction.response_text,
                    'timestamp': interaction.created_at.isoformat() if interaction.created_at else None
                })

            self.logger.info(f"Retrieved {len(history)} conversation turns for user {user_id} from session {session_id}")
            return history

        except Exception as e:
            self.logger.error(f"Error getting conversation history: {e}", exc_info=True)
            return []


class PersonalizationEngine:
    """Analyze user behavior and personalize learning experience"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PersonalizationEngine")
        self.session_manager = SessionManager()

    def analyze_user_interactions(self, user_id: int, session: Session) -> Dict[str, Any]:
        """
        Analyze user's interaction history to extract patterns

        Returns:
            Dictionary with analytics insights
        """
        try:
            # Get all user interactions
            interactions = session.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).order_by(desc(UserInteraction.created_at)).limit(100).all()

            if not interactions:
                return self._get_default_analytics()

            # Analyze interaction patterns
            analytics = {
                'total_interactions': len(interactions),
                'avg_response_time_ms': self._calculate_avg_response_time(interactions),
                'interaction_types': self._analyze_interaction_types(interactions),
                'preferred_topics': self._extract_preferred_topics(interactions),
                'question_complexity': self._estimate_question_complexity(interactions),
                'active_hours': self._find_active_hours(interactions),
                'response_quality_score': self._calculate_response_quality(interactions)
            }

            return analytics

        except Exception as e:
            self.logger.error(f"Error analyzing user interactions: {e}")
            return self._get_default_analytics()

    def _get_default_analytics(self) -> Dict[str, Any]:
        """Return default analytics for new users"""
        return {
            'total_interactions': 0,
            'avg_response_time_ms': 0,
            'interaction_types': {},
            'preferred_topics': [],
            'question_complexity': 0.5,
            'active_hours': [],
            'response_quality_score': 0.0
        }

    def _calculate_avg_response_time(self, interactions: List[UserInteraction]) -> float:
        """Calculate average response time"""
        times = [i.response_time_ms for i in interactions if i.response_time_ms]
        return sum(times) / len(times) if times else 0.0

    def _analyze_interaction_types(self, interactions: List[UserInteraction]) -> Dict[str, int]:
        """Count interaction types"""
        types = [i.interaction_type for i in interactions if i.interaction_type]
        return dict(Counter(types))

    def _extract_preferred_topics(self, interactions: List[UserInteraction]) -> List[str]:
        """Extract most discussed topics/courses"""
        topics = [i.course_context for i in interactions if i.course_context]
        topic_counts = Counter(topics)
        # Return top 5 topics
        return [topic for topic, count in topic_counts.most_common(5)]

    def _estimate_question_complexity(self, interactions: List[UserInteraction]) -> float:
        """
        Estimate user's question complexity preference (0-1 scale)
        Based on query length and interaction patterns, with rating influence
        """
        if not interactions:
            return 0.5

        query_lengths = [len(i.query_text.split()) for i in interactions if i.query_text]
        avg_length = sum(query_lengths) / len(query_lengths) if query_lengths else 10

        # Base complexity from query length
        if avg_length < 5:
            base_complexity = 0.3
        elif avg_length < 15:
            base_complexity = 0.5
        else:
            base_complexity = 0.8

        # Apply rating influence to adjust complexity preference
        adjusted_complexity = self._apply_complexity_rating_influence(
            base_complexity, interactions
        )

        return adjusted_complexity

    def _find_active_hours(self, interactions: List[UserInteraction]) -> List[int]:
        """Find hours when user is most active"""
        hours = [i.created_at.hour for i in interactions if i.created_at]
        hour_counts = Counter(hours)
        # Return top 3 active hours
        return [hour for hour, count in hour_counts.most_common(3)]

    def _calculate_response_quality(self, interactions: List[UserInteraction]) -> float:
        """Calculate average response quality based on user ratings"""
        ratings = [i.user_rating for i in interactions if i.user_rating]
        return sum(ratings) / len(ratings) if ratings else 0.0

    def _apply_complexity_rating_influence(
        self,
        base_complexity: float,
        interactions: List[UserInteraction]
    ) -> float:
        """
        Apply rating influence to complexity score

        Strategy:
        - Group rated interactions by query word count (proxy for complexity asked)
        - Check if higher/lower complexity questions get better ratings
        - Nudge the base_complexity toward the preferred direction
        - Adjustment range: ±0.15 (ratings suggest preference but don't override)

        Returns:
            Adjusted complexity value (0.0-1.0)
        """
        # Separate interactions into complexity buckets and collect ratings
        simple_ratings = []   # < 5 words
        medium_ratings = []   # 5-15 words
        complex_ratings = []  # > 15 words

        for interaction in interactions:
            if not interaction.user_rating or interaction.user_rating < 1:
                continue

            word_count = len(interaction.query_text.split()) if interaction.query_text else 0

            if word_count < 5:
                simple_ratings.append(interaction.user_rating)
            elif word_count < 15:
                medium_ratings.append(interaction.user_rating)
            else:
                complex_ratings.append(interaction.user_rating)

        # Need at least 5 samples in at least 2 categories for reliable comparison
        categories_with_data = sum([
            1 for ratings in [simple_ratings, medium_ratings, complex_ratings]
            if len(ratings) >= 5
        ])

        if categories_with_data < 2:
            # Not enough data, return base complexity
            return base_complexity

        # Calculate average ratings for each category
        avg_ratings = {}
        if len(simple_ratings) >= 5:
            avg_ratings[0.3] = sum(simple_ratings) / len(simple_ratings)
        if len(medium_ratings) >= 5:
            avg_ratings[0.5] = sum(medium_ratings) / len(medium_ratings)
        if len(complex_ratings) >= 5:
            avg_ratings[0.8] = sum(complex_ratings) / len(complex_ratings)

        # Find which complexity level has best ratings
        if avg_ratings:
            best_complexity = max(avg_ratings, key=avg_ratings.get)
            best_rating = avg_ratings[best_complexity]

            # Calculate adjustment: nudge toward best-rated complexity
            # Max adjustment: ±0.15 (15% of scale)
            rating_difference = best_rating - sum(avg_ratings.values()) / len(avg_ratings)
            adjustment = (best_complexity - base_complexity) * min(abs(rating_difference) / 2.0, 0.3)

            adjusted = base_complexity + adjustment

            # Clamp to valid range
            adjusted = max(0.2, min(0.9, adjusted))

            self.logger.debug(
                f"Complexity adjustment: base={base_complexity:.2f}, "
                f"best_rated={best_complexity:.2f} (rating={best_rating:.2f}), "
                f"adjusted={adjusted:.2f}"
            )

            return adjusted

        return base_complexity

    def _apply_pace_rating_influence(
        self,
        base_pace: str,
        interactions: List[UserInteraction]
    ) -> str:
        """
        Apply rating influence to learning pace

        Strategy:
        - Separate interactions by response length (proxy for information density)
        - Short responses = casual pace, Long responses = intensive pace
        - Check which gets better ratings
        - Nudge pace one level up/down if clear preference exists
        - Requires strong signal (0.5+ rating difference) to change

        Returns:
            Adjusted pace: 'slow', 'medium', or 'fast'
        """
        # Separate by response length and collect ratings
        short_responses = []   # < 500 chars (casual pace)
        medium_responses = []  # 500-1500 chars (medium pace)
        long_responses = []    # > 1500 chars (intensive pace)

        for interaction in interactions:
            if not interaction.user_rating or interaction.user_rating < 1:
                continue

            response_len = len(interaction.response_text) if interaction.response_text else 0

            if response_len < 500:
                short_responses.append(interaction.user_rating)
            elif response_len < 1500:
                medium_responses.append(interaction.user_rating)
            else:
                long_responses.append(interaction.user_rating)

        # Need at least 5 samples in at least 2 categories for comparison
        categories_with_data = sum([
            1 for ratings in [short_responses, medium_responses, long_responses]
            if len(ratings) >= 5
        ])

        if categories_with_data < 2:
            # Not enough data, return base pace
            return base_pace

        # Calculate average ratings
        avg_ratings = {}
        if len(short_responses) >= 5:
            avg_ratings['slow'] = sum(short_responses) / len(short_responses)
        if len(medium_responses) >= 5:
            avg_ratings['medium'] = sum(medium_responses) / len(medium_responses)
        if len(long_responses) >= 5:
            avg_ratings['fast'] = sum(long_responses) / len(long_responses)

        if not avg_ratings:
            return base_pace

        # Find best-rated pace
        best_pace = max(avg_ratings, key=avg_ratings.get)
        best_rating = avg_ratings[best_pace]
        avg_rating = sum(avg_ratings.values()) / len(avg_ratings)

        # Only adjust if there's a significant difference (0.5+ stars)
        rating_difference = best_rating - avg_rating

        if rating_difference >= 0.5:
            # Strong signal: prefer the best-rated pace
            self.logger.debug(
                f"Pace adjustment: base={base_pace}, "
                f"best_rated={best_pace} (rating={best_rating:.2f}), "
                f"difference={rating_difference:.2f}"
            )
            return best_pace
        else:
            # Weak signal: keep base pace
            return base_pace

    def update_response_length_preference(self, user_id: int, session: Session) -> str:
        """
        Auto-update preferred_response_length based on user ratings and explicit signals

        Analyzes last 50 rated interactions to determine which response lengths
        get the best ratings from the user.

        Strategy:
        1. Count explicit brevity/detail requests in queries
        2. If >60% explicit signals for one type → use that
        3. Otherwise, categorize responses by length and compare avg ratings
        4. Choose length category with highest avg rating (min 10 samples)

        Args:
            user_id: User database ID
            session: Database session

        Returns:
            Updated preference: 'short', 'medium', or 'long'
        """
        try:
            # Get last 50 interactions with ratings
            rated_interactions = session.query(UserInteraction).filter(
                and_(
                    UserInteraction.user_id == user_id,
                    UserInteraction.user_rating.isnot(None),
                    UserInteraction.user_rating >= 1
                )
            ).order_by(desc(UserInteraction.created_at)).limit(50).all()

            if len(rated_interactions) < 10:
                self.logger.debug(f"User {user_id}: Not enough rated interactions ({len(rated_interactions)}), keeping default")
                return 'medium'

            # Strategy 1: Check for explicit signals in queries
            from src.services.adaptive_response_engine import SentimentDetector
            sentiment_detector = SentimentDetector()

            brevity_count = 0
            detail_count = 0

            for interaction in rated_interactions:
                analysis = sentiment_detector.analyze_query(interaction.query_text)
                if analysis['wants_brevity']:
                    brevity_count += 1
                if analysis['wants_more_detail']:
                    detail_count += 1

            total_explicit = brevity_count + detail_count

            # If >60% of rated interactions have explicit signals, trust those
            if total_explicit >= len(rated_interactions) * 0.6:
                if brevity_count > detail_count * 2:
                    self.logger.info(f"User {user_id}: Strong brevity preference detected ({brevity_count}/{total_explicit})")
                    return 'short'
                elif detail_count > brevity_count * 2:
                    self.logger.info(f"User {user_id}: Strong detail preference detected ({detail_count}/{total_explicit})")
                    return 'long'

            # Strategy 2: Analyze ratings by response length
            # Categorize responses by character count
            short_responses = []  # < 500 chars
            medium_responses = []  # 500-1500 chars
            long_responses = []  # > 1500 chars

            for interaction in rated_interactions:
                if not interaction.response_text:
                    continue

                response_len = len(interaction.response_text)
                rating = interaction.user_rating

                if response_len < 500:
                    short_responses.append(rating)
                elif response_len < 1500:
                    medium_responses.append(rating)
                else:
                    long_responses.append(rating)

            # Calculate average ratings for each category (need min 10 samples)
            ratings_by_length = {}

            if len(short_responses) >= 10:
                ratings_by_length['short'] = sum(short_responses) / len(short_responses)

            if len(medium_responses) >= 10:
                ratings_by_length['medium'] = sum(medium_responses) / len(medium_responses)

            if len(long_responses) >= 10:
                ratings_by_length['long'] = sum(long_responses) / len(long_responses)

            if not ratings_by_length:
                self.logger.debug(f"User {user_id}: Not enough samples in any category, keeping medium")
                return 'medium'

            # Choose category with highest average rating
            best_length = max(ratings_by_length, key=ratings_by_length.get)
            best_rating = ratings_by_length[best_length]

            self.logger.info(
                f"User {user_id}: Response length preference updated to '{best_length}' "
                f"(avg rating: {best_rating:.2f}, ratings by length: {ratings_by_length})"
            )

            return best_length

        except Exception as e:
            self.logger.error(f"Error updating response length preference for user {user_id}: {e}", exc_info=True)
            return 'medium'  # Default on error

    def update_personalization_profile(self, user_id: int) -> Optional[PersonalizationProfile]:
        """
        Update user's personalization profile with latest analytics

        Args:
            user_id: User database ID

        Returns:
            Updated PersonalizationProfile or None on error
        """
        try:
            with db_manager.get_session() as session:
                # Get or create profile
                profile = session.query(PersonalizationProfile).filter(
                    PersonalizationProfile.user_id == user_id
                ).first()

                if not profile:
                    profile = PersonalizationProfile(user_id=user_id)
                    session.add(profile)

                # Analyze interactions
                analytics = self.analyze_user_interactions(user_id, session)

                # Update profile fields
                profile.total_interactions = analytics['total_interactions']
                profile.question_complexity_level = analytics['question_complexity']
                profile.most_active_hours = analytics['active_hours']
                profile.preferred_subjects = analytics['preferred_topics']

                # Calculate average session duration
                avg_session_duration = self._calculate_avg_session_duration(user_id, session)
                profile.avg_session_duration = avg_session_duration

                # Update learning pace based on interaction frequency
                profile.learning_pace = self._determine_learning_pace(user_id, session)

                # Calculate successful interactions
                successful_count = session.query(UserInteraction).filter(
                    and_(
                        UserInteraction.user_id == user_id,
                        UserInteraction.was_helpful == True
                    )
                ).count()
                profile.successful_interactions = successful_count

                profile.last_interaction = datetime.utcnow()
                profile.last_model_update = datetime.utcnow()

                # Store feature vector for future ML use
                profile.feature_vector = {
                    'complexity': analytics['question_complexity'],
                    'avg_response_time': analytics['avg_response_time_ms'],
                    'quality_score': analytics['response_quality_score'],
                    'interaction_count': analytics['total_interactions']
                }

                session.commit()
                self.logger.info(f"Updated personalization profile for user {user_id}")

                return profile

        except Exception as e:
            self.logger.error(f"Error updating personalization profile: {e}", exc_info=True)
            return None

    def _calculate_avg_session_duration(self, user_id: int, session: Session) -> float:
        """Calculate average session duration in minutes"""
        closed_sessions = session.query(ConversationSession).filter(
            and_(
                ConversationSession.user_id == user_id,
                ConversationSession.is_active == False,
                ConversationSession.session_duration_minutes > 0
            )
        ).limit(20).all()

        if not closed_sessions:
            return 0.0

        total_duration = sum(s.session_duration_minutes for s in closed_sessions)
        return total_duration / len(closed_sessions)

    def _determine_learning_pace(self, user_id: int, session: Session) -> str:
        """
        Determine user's learning pace (slow, medium, fast)
        Based on interaction frequency, with rating influence
        """
        # Calculate interactions per day over last 7 days
        week_ago = datetime.utcnow() - timedelta(days=7)

        recent_interactions = session.query(UserInteraction).filter(
            and_(
                UserInteraction.user_id == user_id,
                UserInteraction.created_at >= week_ago
            )
        ).all()

        interactions_per_day = len(recent_interactions) / 7.0

        # Base pace from interaction frequency
        if interactions_per_day < 2:
            base_pace = "slow"
        elif interactions_per_day < 5:
            base_pace = "medium"
        else:
            base_pace = "fast"

        # Apply rating influence to adjust pace preference
        adjusted_pace = self._apply_pace_rating_influence(
            base_pace, recent_interactions
        )

        return adjusted_pace

    def get_personalized_settings(self, user_id: int) -> Dict[str, Any]:
        """
        Get personalized settings for adaptive responses

        Returns:
            Dictionary with personalization settings
        """
        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                profile = session.query(PersonalizationProfile).filter(
                    PersonalizationProfile.user_id == user_id
                ).first()

                if not user:
                    return self._get_default_settings()

                settings = {
                    'learning_style': user.learning_style,
                    'difficulty_preference': user.difficulty_preference,
                    'response_length': profile.preferred_response_length if profile else 'medium',
                    'complexity_level': profile.question_complexity_level if profile else 0.5,
                    'preferred_topics': profile.preferred_subjects if profile else [],
                    'learning_pace': profile.learning_pace if profile else 'medium'
                }

                return settings

        except Exception as e:
            self.logger.error(f"Error getting personalized settings: {e}")
            return self._get_default_settings()

    def _get_default_settings(self) -> Dict[str, Any]:
        """Return default settings for new users"""
        return {
            'learning_style': 'adaptive',
            'difficulty_preference': 'medium',
            'response_length': 'medium',
            'complexity_level': 0.5,
            'preferred_topics': [],
            'learning_pace': 'medium'
        }

    def record_interaction(
        self,
        user_id: int,
        query: str,
        response: str,
        interaction_type: str = "question",
        course_context: Optional[str] = None,
        documents_referenced: Optional[List[int]] = None,
        response_time_ms: Optional[int] = None
    ) -> bool:
        """
        Record a user interaction for personalization

        Args:
            user_id: User database ID
            query: User's query text
            response: Bot's response text
            interaction_type: Type of interaction
            course_context: Course being discussed
            documents_referenced: List of document IDs referenced
            response_time_ms: Response time in milliseconds

        Returns:
            True if successful, False otherwise
        """
        try:
            with db_manager.get_session() as session:
                interaction = UserInteraction(
                    user_id=user_id,
                    query_text=query,
                    response_text=response,
                    interaction_type=interaction_type,
                    course_context=course_context,
                    documents_referenced=documents_referenced or [],
                    response_time_ms=response_time_ms,
                    created_at=datetime.utcnow()
                )

                session.add(interaction)
                session.commit()

                # Update profile every 5 interactions
                total_interactions = session.query(UserInteraction).filter(
                    UserInteraction.user_id == user_id
                ).count()

                if total_interactions % 5 == 0:
                    self.update_personalization_profile(user_id)

                return True

        except Exception as e:
            self.logger.error(f"Error recording interaction: {e}")
            return False

    def should_personalize(self, user_id: int) -> bool:
        """
        Check if user has enough interactions for personalization

        Returns:
            True if personalization should be active
        """
        try:
            with db_manager.get_session() as session:
                interaction_count = session.query(UserInteraction).filter(
                    UserInteraction.user_id == user_id
                ).count()

                min_interactions = getattr(settings, 'MIN_INTERACTIONS_FOR_PERSONALIZATION', 5)
                return interaction_count >= min_interactions

        except Exception as e:
            self.logger.error(f"Error checking personalization eligibility: {e}")
            return False

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a conversation session"""
        try:
            with db_manager.get_session() as db_session:
                conv_session = db_session.query(ConversationSession).filter(
                    ConversationSession.session_id == session_id
                ).first()

                if not conv_session:
                    return None

                return {
                    'session_id': conv_session.session_id,
                    'user_id': conv_session.user_id,
                    'is_active': conv_session.is_active,
                    'duration_minutes': conv_session.session_duration_minutes,
                    'message_count': conv_session.message_count,
                    'questions_asked': conv_session.questions_asked,
                    'commands_used': conv_session.commands_used,
                    'primary_topic': conv_session.primary_topic,
                    'started_at': conv_session.started_at.isoformat() if conv_session.started_at else None,
                    'ended_at': conv_session.ended_at.isoformat() if conv_session.ended_at else None
                }

        except Exception as e:
            self.logger.error(f"Error getting session summary: {e}")
            return None


# Global instance for easy access
personalization_engine = PersonalizationEngine()
session_manager = SessionManager()
