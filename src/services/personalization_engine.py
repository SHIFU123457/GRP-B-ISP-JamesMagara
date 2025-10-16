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
        Based on query length and interaction patterns
        """
        if not interactions:
            return 0.5

        query_lengths = [len(i.query_text.split()) for i in interactions if i.query_text]
        avg_length = sum(query_lengths) / len(query_lengths) if query_lengths else 10

        # Normalize: < 5 words = simple (0.3), 5-15 = medium (0.5), > 15 = complex (0.8)
        if avg_length < 5:
            return 0.3
        elif avg_length < 15:
            return 0.5
        else:
            return 0.8

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
        """Determine user's learning pace (slow, medium, fast)"""
        # Calculate interactions per day over last 7 days
        week_ago = datetime.utcnow() - timedelta(days=7)

        recent_interactions = session.query(UserInteraction).filter(
            and_(
                UserInteraction.user_id == user_id,
                UserInteraction.created_at >= week_ago
            )
        ).count()

        interactions_per_day = recent_interactions / 7.0

        if interactions_per_day < 2:
            return "slow"
        elif interactions_per_day < 5:
            return "medium"
        else:
            return "fast"

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
