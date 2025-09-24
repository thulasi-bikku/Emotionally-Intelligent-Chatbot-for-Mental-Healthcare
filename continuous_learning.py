import sqlite3
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import threading
import time
import pickle
from collections import deque, defaultdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningData:
    """Structure for learning data"""
    interaction_id: str
    user_input: str
    bot_response: str
    user_feedback: Optional[float]
    context: Dict
    timestamp: datetime
    improvement_suggested: Optional[str]
    emotional_impact_score: float
    effectiveness_score: float

@dataclass
class ResponsePattern:
    """Pattern for successful responses"""
    pattern_id: str
    intent_category: str
    input_features: Dict
    response_template: str
    success_rate: float
    usage_count: int
    last_updated: datetime
    emotional_resonance: float
    crisis_appropriateness: float

class EnhancedContinuousLearningSystem:
    """Advanced continuous learning system with RLHF and real-time adaptation"""
    
    def __init__(self, db_path: str = "learning.db"):
        self.db_path = db_path
        self.learning_lock = threading.Lock()
        
        # Feedback buffer for batch learning
        self.feedback_buffer = deque(maxlen=1000)
        self.recent_interactions = deque(maxlen=100)
        
        # Adaptation rules and patterns
        self.adaptation_rules = {}
        self.response_patterns = {}
        self.user_preferences = defaultdict(dict)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        self.confidence_threshold = 0.7
        self.min_samples_for_pattern = 5
        
        # Reinforcement learning components
        self.reward_history = deque(maxlen=1000)
        self.policy_updates = []
        
        # Real-time learning metrics
        self.learning_metrics = {
            'total_interactions': 0,
            'positive_feedback_rate': 0.0,
            'pattern_accuracy': 0.0,
            'adaptation_success_rate': 0.0,
            'user_satisfaction_trend': 0.0
        }
        
        # Initialize learning database
        self._init_learning_database()
        
        # Load existing patterns and rules
        self._load_learning_data()
        
        # Start background learning process
        self._start_background_learning()
    
    def _init_learning_database(self):
        """Initialize comprehensive learning database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enhanced user feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    interaction_id TEXT,
                    user_input TEXT,
                    bot_response TEXT,
                    feedback_score REAL,
                    feedback_type TEXT,
                    feedback_text TEXT,
                    emotional_impact_score REAL,
                    effectiveness_score REAL,
                    context_data TEXT,
                    user_demographics TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Response patterns with advanced metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS response_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT UNIQUE,
                    intent_category TEXT,
                    input_features TEXT,
                    response_template TEXT,
                    success_rate REAL,
                    usage_count INTEGER DEFAULT 1,
                    emotional_resonance REAL DEFAULT 0.0,
                    crisis_appropriateness REAL DEFAULT 0.0,
                    user_group_effectiveness TEXT,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Adaptation rules with reinforcement learning
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS adaptation_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT UNIQUE,
                    rule_type TEXT,
                    condition_pattern TEXT,
                    action_type TEXT,
                    action_data TEXT,
                    confidence_score REAL,
                    reward_history TEXT,
                    application_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_applied TEXT
                )
            ''')
            
            # Learning metrics and analytics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    metric_context TEXT,
                    aggregation_period TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User preference profiles
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    preference_type TEXT,
                    preference_data TEXT,
                    confidence_score REAL,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Reward signals for reinforcement learning
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reward_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT,
                    session_id TEXT,
                    action_taken TEXT,
                    immediate_reward REAL,
                    delayed_reward REAL,
                    reward_source TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def record_interaction_with_rlhf(self, session_id: str, user_input: str, bot_response: str,
                                   context: Dict, analysis_result: Dict,
                                   user_feedback: Optional[Dict] = None):
        """Record interaction with comprehensive data for RLHF"""
        interaction_id = f"{session_id}_{int(time.time())}"
        
        # Create learning data
        learning_data = LearningData(
            interaction_id=interaction_id,
            user_input=user_input,
            bot_response=bot_response,
            user_feedback=user_feedback.get('score') if user_feedback else None,
            context=context,
            timestamp=datetime.now(),
            improvement_suggested=user_feedback.get('suggestion') if user_feedback else None,
            emotional_impact_score=self._calculate_emotional_impact(analysis_result),
            effectiveness_score=self._calculate_effectiveness(analysis_result, user_feedback)
        )
        
        with self.learning_lock:
            self.feedback_buffer.append(learning_data)
            self.recent_interactions.append(learning_data)
            self.learning_metrics['total_interactions'] += 1
        
        # Store in database
        self._store_learning_data(learning_data, analysis_result)
        
        # Real-time adaptation
        if user_feedback and user_feedback.get('score', 0) < 0.3:
            self._trigger_immediate_adaptation(learning_data, analysis_result)
        
        # Update user preferences
        self._update_user_preferences(context.get('user_id'), analysis_result, user_feedback)
        
        logger.info(f"Recorded interaction with RLHF: {interaction_id}")
    
    def _calculate_emotional_impact(self, analysis_result: Dict) -> float:
        """Calculate emotional impact score of the response"""
        try:
            emotions = analysis_result.get('emotions', {})
            sentiment = analysis_result.get('sentiment', {})
            crisis = analysis_result.get('crisis_assessment', {})
            
            # Base emotional score from sentiment
            emotional_score = abs(sentiment.get('compound', 0))
            
            # Adjust for emotional appropriateness
            primary_emotion = emotions.get('primary_emotion', 'neutral')
            if primary_emotion in ['joy', 'optimism']:
                emotional_score *= 1.2
            elif primary_emotion in ['sadness', 'fear', 'anger']:
                emotional_score *= 0.8  # More careful in negative emotional states
            
            # Crisis situations require special handling
            if crisis.get('is_crisis', False):
                emotional_score *= 0.6  # Lower score for crisis - need human intervention
            
            return min(emotional_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating emotional impact: {e}")
            return 0.5
    
    def _calculate_effectiveness(self, analysis_result: Dict, user_feedback: Optional[Dict]) -> float:
        """Calculate response effectiveness score"""
        try:
            effectiveness = 0.5  # Base effectiveness
            
            # User feedback is primary indicator
            if user_feedback:
                feedback_score = user_feedback.get('score', 0.5)
                effectiveness = feedback_score * 0.7
                
                # Text feedback analysis
                feedback_text = user_feedback.get('text', '')
                if any(word in feedback_text.lower() for word in ['helpful', 'good', 'better', 'thanks']):
                    effectiveness += 0.2
                elif any(word in feedback_text.lower() for word in ['not helpful', 'bad', 'wrong']):
                    effectiveness -= 0.2
            
            # Intent matching accuracy
            intent_match = analysis_result.get('intent_classification', {}).get('confidence', 0)
            effectiveness += intent_match * 0.2
            
            # Crisis handling appropriateness
            crisis_assessment = analysis_result.get('crisis_assessment', {})
            if crisis_assessment.get('is_crisis') and crisis_assessment.get('intervention_needed'):
                effectiveness += 0.3  # Good crisis detection
            
            return max(0.0, min(effectiveness, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating effectiveness: {e}")
            return 0.5
    
    def _trigger_immediate_adaptation(self, learning_data: LearningData, analysis_result: Dict):
        """Trigger immediate adaptation for poor responses"""
        try:
            # Identify what went wrong
            issues = []
            
            if learning_data.effectiveness_score < 0.3:
                issues.append('low_effectiveness')
            
            if learning_data.emotional_impact_score < 0.2:
                issues.append('poor_emotional_resonance')
            
            if learning_data.user_feedback and learning_data.user_feedback < 0.3:
                issues.append('negative_user_feedback')
            
            # Create adaptation rule
            adaptation_rule = {
                'rule_id': f"immediate_adapt_{int(time.time())}",
                'rule_type': 'immediate_correction',
                'trigger_conditions': issues,
                'original_response': learning_data.bot_response,
                'context_features': self._extract_context_features(learning_data.context),
                'improvement_needed': True,
                'priority': 'high'
            }
            
            # Store adaptation rule
            self.adaptation_rules[adaptation_rule['rule_id']] = adaptation_rule
            self._store_adaptation_rule(adaptation_rule)
            
            logger.info(f"Triggered immediate adaptation for issues: {issues}")
            
        except Exception as e:
            logger.error(f"Error in immediate adaptation: {e}")
    
    def _extract_context_features(self, context: Dict) -> Dict:
        """Extract relevant features from context for learning"""
        features = {}
        
        try:
            # User demographic features
            if 'user_profile' in context:
                profile = context['user_profile']
                features['age_group'] = profile.get('age_group', 'unknown')
                features['previous_sessions'] = profile.get('session_count', 0)
                features['risk_history'] = profile.get('crisis_history', False)
            
            # Conversation context
            if 'conversation_stage' in context:
                features['conversation_stage'] = context['conversation_stage']
            
            # Emotional context
            if 'emotional_state' in context:
                features['user_emotional_state'] = context['emotional_state']
            
            # Session context
            if 'session_duration' in context:
                features['session_duration'] = context['session_duration']
            
            # Time context
            features['time_of_day'] = datetime.now().hour
            features['day_of_week'] = datetime.now().weekday()
            
        except Exception as e:
            logger.error(f"Error extracting context features: {e}")
        
        return features
    
    def learn_from_human_feedback(self, feedback_batch: List[Dict]) -> Dict:
        """Implement Reinforcement Learning from Human Feedback (RLHF)"""
        try:
            learning_results = {
                'patterns_updated': 0,
                'new_patterns_created': 0,
                'rules_modified': 0,
                'average_reward': 0.0
            }
            
            # Process feedback batch
            rewards = []
            for feedback in feedback_batch:
                reward = self._calculate_reward(feedback)
                rewards.append(reward)
                
                # Update response patterns based on reward
                self._update_response_patterns_with_reward(feedback, reward)
                
                # Update adaptation rules
                self._update_adaptation_rules_with_reward(feedback, reward)
            
            # Calculate policy updates using proximal policy optimization concepts
            if rewards:
                learning_results['average_reward'] = np.mean(rewards)
                self._update_learning_policy(rewards)
            
            # Store learning metrics
            self._store_learning_metrics(learning_results)
            
            logger.info(f"RLHF learning completed: {learning_results}")
            return learning_results
            
        except Exception as e:
            logger.error(f"Error in RLHF learning: {e}")
            return {'error': str(e)}
    
    def _calculate_reward(self, feedback: Dict) -> float:
        """Calculate reward signal from user feedback"""
        try:
            # Base reward from explicit feedback
            explicit_reward = feedback.get('satisfaction_score', 0.5)
            
            # Implicit reward from engagement
            engagement_reward = 0.0
            if feedback.get('conversation_continued', False):
                engagement_reward += 0.2
            if feedback.get('follow_up_questions', 0) > 0:
                engagement_reward += 0.1
            
            # Safety reward
            safety_reward = 0.0
            if feedback.get('crisis_handled_well', False):
                safety_reward += 0.5
            elif feedback.get('crisis_missed', False):
                safety_reward -= 0.8
            
            # Emotional appropriateness reward
            emotional_reward = feedback.get('emotional_appropriateness', 0.5)
            
            # Combine rewards with weights
            total_reward = (
                explicit_reward * 0.4 +
                engagement_reward * 0.2 +
                safety_reward * 0.3 +
                emotional_reward * 0.1
            )
            
            return max(-1.0, min(total_reward, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _update_response_patterns_with_reward(self, feedback: Dict, reward: float):
        """Update response patterns using reward signal"""
        try:
            pattern_id = feedback.get('pattern_id')
            if not pattern_id or pattern_id not in self.response_patterns:
                return
            
            pattern = self.response_patterns[pattern_id]
            
            # Update success rate using exponential moving average
            new_success_rate = (
                pattern.success_rate * 0.9 + 
                (1.0 if reward > 0.5 else 0.0) * 0.1
            )
            
            # Update emotional resonance
            emotional_feedback = feedback.get('emotional_impact', 0.5)
            new_emotional_resonance = (
                pattern.emotional_resonance * 0.8 + 
                emotional_feedback * 0.2
            )
            
            # Update pattern
            pattern.success_rate = new_success_rate
            pattern.emotional_resonance = new_emotional_resonance
            pattern.usage_count += 1
            pattern.last_updated = datetime.now()
            
            # Store updated pattern
            self._store_response_pattern(pattern)
            
        except Exception as e:
            logger.error(f"Error updating response pattern with reward: {e}")
    
    def get_personalized_response_suggestions(self, context: Dict, analysis_result: Dict) -> List[Dict]:
        """Get personalized response suggestions based on learning"""
        try:
            suggestions = []
            user_id = context.get('user_id')
            
            # Get user preferences
            user_prefs = self.user_preferences.get(user_id, {})
            
            # Find matching patterns
            intent = analysis_result.get('intent_classification', {}).get('primary_intent', 'general')
            emotion = analysis_result.get('emotions', {}).get('primary_emotion', 'neutral')
            
            # Score and rank response patterns
            pattern_scores = []
            for pattern_id, pattern in self.response_patterns.items():
                if pattern.intent_category == intent:
                    score = self._calculate_pattern_score(pattern, context, analysis_result, user_prefs)
                    pattern_scores.append((pattern, score))
            
            # Sort by score and take top suggestions
            pattern_scores.sort(key=lambda x: x[1], reverse=True)
            
            for pattern, score in pattern_scores[:3]:
                suggestions.append({
                    'response_template': pattern.response_template,
                    'confidence_score': score,
                    'pattern_id': pattern.pattern_id,
                    'success_rate': pattern.success_rate,
                    'emotional_resonance': pattern.emotional_resonance,
                    'personalization_applied': True
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating personalized suggestions: {e}")
            return []
    
    def _calculate_pattern_score(self, pattern: ResponsePattern, context: Dict, 
                               analysis_result: Dict, user_prefs: Dict) -> float:
        """Calculate relevance score for a response pattern"""
        try:
            score = pattern.success_rate * 0.4
            
            # Emotional appropriateness
            user_emotion = analysis_result.get('emotions', {}).get('primary_emotion', 'neutral')
            if user_prefs.get('preferred_emotional_tone'):
                if user_prefs['preferred_emotional_tone'] == 'supportive':
                    score += pattern.emotional_resonance * 0.3
                elif user_prefs['preferred_emotional_tone'] == 'direct':
                    score += (1 - pattern.emotional_resonance) * 0.3
            
            # Crisis appropriateness
            if analysis_result.get('crisis_assessment', {}).get('is_crisis', False):
                score += pattern.crisis_appropriateness * 0.4
            
            # Recency bonus
            days_since_update = (datetime.now() - pattern.last_updated).days
            recency_factor = math.exp(-days_since_update / 30)  # Decay over 30 days
            score *= (0.7 + 0.3 * recency_factor)
            
            # Usage frequency (popularity)
            if pattern.usage_count > 10:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating pattern score: {e}")
            return 0.0
    
    def _start_background_learning(self):
        """Start background learning process"""
        def learning_worker():
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    self._process_feedback_buffer()
                    self._update_learning_metrics()
                    self._cleanup_old_data()
                except Exception as e:
                    logger.error(f"Error in background learning: {e}")
        
        learning_thread = threading.Thread(target=learning_worker, daemon=True)
        learning_thread.start()
        logger.info("Background learning process started")
    
    def _process_feedback_buffer(self):
        """Process accumulated feedback for batch learning"""
        if len(self.feedback_buffer) < 10:
            return
        
        feedback_batch = list(self.feedback_buffer)
        self.feedback_buffer.clear()
        
        # Convert to RLHF format
        rlhf_feedback = []
        for learning_data in feedback_batch:
            if learning_data.user_feedback is not None:
                rlhf_feedback.append({
                    'satisfaction_score': learning_data.user_feedback,
                    'emotional_impact': learning_data.emotional_impact_score,
                    'effectiveness': learning_data.effectiveness_score,
                    'context': learning_data.context
                })
        
        if rlhf_feedback:
            self.learn_from_human_feedback(rlhf_feedback)
    
    def get_learning_analytics(self) -> Dict:
        """Get comprehensive learning analytics"""
        try:
            analytics = dict(self.learning_metrics)
            
            # Add recent performance metrics
            if self.recent_interactions:
                recent_feedback = [
                    data.user_feedback for data in self.recent_interactions 
                    if data.user_feedback is not None
                ]
                
                if recent_feedback:
                    analytics['recent_satisfaction'] = np.mean(recent_feedback)
                    analytics['recent_feedback_count'] = len(recent_feedback)
            
            # Pattern effectiveness
            if self.response_patterns:
                pattern_success_rates = [p.success_rate for p in self.response_patterns.values()]
                analytics['average_pattern_success_rate'] = np.mean(pattern_success_rates)
                analytics['total_patterns'] = len(self.response_patterns)
            
            # Adaptation metrics
            if self.adaptation_rules:
                analytics['total_adaptation_rules'] = len(self.adaptation_rules)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating learning analytics: {e}")
            return self.learning_metrics
    
    # Database helper methods would go here...
    def _store_learning_data(self, learning_data: LearningData, analysis_result: Dict):
        """Store learning data in database"""
        # Implementation details...
        pass
    
    def _store_adaptation_rule(self, rule: Dict):
        """Store adaptation rule in database"""
        # Implementation details...
        pass
    
    def _store_response_pattern(self, pattern: ResponsePattern):
        """Store response pattern in database"""
        # Implementation details...
        pass
    
    def _load_learning_data(self):
        """Load existing learning data"""
        # Implementation details...
        pass
    
    def _store_learning_metrics(self, metrics: Dict):
        """Store learning metrics"""
        # Implementation details...
        pass
    
    def _update_learning_policy(self, rewards: List[float]):
        """Update learning policy based on rewards"""
        # Implementation details...
        pass
    
    def _update_user_preferences(self, user_id: str, analysis_result: Dict, feedback: Optional[Dict]):
        """Update user preference profiles"""
        # Implementation details...
        pass
    
    def _update_adaptation_rules_with_reward(self, feedback: Dict, reward: float):
        """Update adaptation rules with reward signals"""
        # Implementation details...
        pass
    
    def _update_learning_metrics(self):
        """Update learning metrics periodically"""
        # Implementation details...
        pass
    
    def _cleanup_old_data(self):
        """Clean up old learning data"""
        # Implementation details...
        pass

# Alias for backward compatibility
ContinuousLearningSystem = EnhancedContinuousLearningSystem
        """Record user interaction for learning"""
        interaction_id = f"{session_id}_{datetime.now().timestamp()}"
        
        learning_data = LearningData(
            interaction_id=interaction_id,
            user_input=user_input,
            bot_response=bot_response,
            user_feedback=user_feedback,
            context=context,
            timestamp=datetime.now(),
            improvement_suggested=None
        )
        
        with self.learning_lock:
            self.feedback_buffer.append(learning_data)
        
        # Save to database
        self._save_interaction_to_db(session_id, learning_data)
        
        # Process immediate learning if feedback is available
        if user_feedback is not None:
            self._process_immediate_feedback(learning_data)
    
    def _save_interaction_to_db(self, session_id: str, learning_data: LearningData):
        """Save interaction to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_feedback 
                (session_id, interaction_id, user_input, bot_response, 
                 feedback_score, context_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                learning_data.interaction_id,
                learning_data.user_input,
                learning_data.bot_response,
                learning_data.user_feedback,
                json.dumps(learning_data.context),
                learning_data.timestamp.isoformat()
            ))
            conn.commit()
    
    def _process_immediate_feedback(self, learning_data: LearningData):
        """Process immediate user feedback"""
        feedback_score = learning_data.user_feedback
        
        if feedback_score >= 4.0:  # Positive feedback (scale 1-5)
            self._reinforce_successful_pattern(learning_data)
        elif feedback_score <= 2.0:  # Negative feedback
            self._learn_from_failure(learning_data)
        
        # Log learning metric
        self._log_learning_metric("immediate_feedback", feedback_score, learning_data.context)
    
    def _reinforce_successful_pattern(self, learning_data: LearningData):
        """Reinforce successful response patterns"""
        intent = learning_data.context.get('intent', 'unknown')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if pattern exists
            cursor.execute('''
                SELECT id, success_rate, usage_count FROM response_patterns
                WHERE intent_category = ? AND input_pattern LIKE ?
            ''', (intent, f"%{learning_data.user_input[:50]}%"))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing pattern
                new_success_rate = (existing[1] * existing[2] + learning_data.user_feedback) / (existing[2] + 1)
                cursor.execute('''
                    UPDATE response_patterns 
                    SET success_rate = ?, usage_count = usage_count + 1, last_updated = ?
                    WHERE id = ?
                ''', (new_success_rate, datetime.now().isoformat(), existing[0]))
            else:
                # Create new pattern
                cursor.execute('''
                    INSERT INTO response_patterns 
                    (intent_category, input_pattern, successful_response, success_rate)
                    VALUES (?, ?, ?, ?)
                ''', (intent, learning_data.user_input[:100], learning_data.bot_response[:500], 
                      learning_data.user_feedback))
            
            conn.commit()
    
    def _learn_from_failure(self, learning_data: LearningData):
        """Learn from unsuccessful interactions"""
        # Create adaptation rule to avoid similar responses
        rule_data = {
            'avoid_response': learning_data.bot_response[:200],
            'for_input_pattern': learning_data.user_input[:100],
            'context': learning_data.context
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO adaptation_rules 
                (rule_type, condition_pattern, action_type, action_data, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            ''', ('avoid_response', learning_data.user_input[:100], 'exclude_response',
                  json.dumps(rule_data), 0.8))
            conn.commit()
        
        logger.info(f"Created avoidance rule for unsuccessful response")
    
    def get_improved_response(self, user_input: str, intent: str, context: Dict) -> Optional[str]:
        """Get improved response based on learning"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Look for successful patterns
            cursor.execute('''
                SELECT successful_response, success_rate, usage_count
                FROM response_patterns
                WHERE intent_category = ? AND success_rate >= 3.5
                ORDER BY success_rate DESC, usage_count DESC
                LIMIT 3
            ''', (intent,))
            
            patterns = cursor.fetchall()
            
            if patterns:
                # Return the most successful response pattern
                return patterns[0][0]
            
            return None
    
    def analyze_learning_trends(self, days: int = 7) -> Dict:
        """Analyze learning trends over time"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Average feedback score
            cursor.execute('''
                SELECT AVG(feedback_score), COUNT(*)
                FROM user_feedback
                WHERE timestamp >= ? AND feedback_score IS NOT NULL
            ''', (cutoff_date.isoformat(),))
            
            avg_feedback = cursor.fetchone()
            
            # Most successful intents
            cursor.execute('''
                SELECT intent_category, AVG(success_rate), COUNT(*)
                FROM response_patterns
                WHERE last_updated >= ?
                GROUP BY intent_category
                ORDER BY AVG(success_rate) DESC
            ''', (cutoff_date.isoformat(),))
            
            intent_performance = cursor.fetchall()
            
            # Learning metrics
            cursor.execute('''
                SELECT metric_name, AVG(metric_value), COUNT(*)
                FROM learning_metrics
                WHERE timestamp >= ?
                GROUP BY metric_name
            ''', (cutoff_date.isoformat(),))
            
            metrics = cursor.fetchall()
        
        return {
            'average_feedback': avg_feedback[0] if avg_feedback[0] else 0,
            'total_interactions': avg_feedback[1] if avg_feedback[1] else 0,
            'intent_performance': [
                {'intent': row[0], 'avg_success': row[1], 'count': row[2]}
                for row in intent_performance
            ],
            'learning_metrics': [
                {'metric': row[0], 'avg_value': row[1], 'count': row[2]}
                for row in metrics
            ]
        }
    
    def _load_adaptation_rules(self):
        """Load adaptation rules from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT rule_type, condition_pattern, action_type, action_data, confidence_score
                    FROM adaptation_rules
                    WHERE confidence_score >= 0.5
                ''')
                
                for row in cursor.fetchall():
                    rule_key = f"{row[0]}_{row[1][:50]}"
                    self.adaptation_rules[rule_key] = {
                        'rule_type': row[0],
                        'condition_pattern': row[1],
                        'action_type': row[2],
                        'action_data': json.loads(row[3]),
                        'confidence_score': row[4]
                    }
        
        except Exception as e:
            logger.error(f"Error loading adaptation rules: {e}")
    
    def _log_learning_metric(self, metric_name: str, value: float, context: Dict):
        """Log learning metric"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO learning_metrics (metric_name, metric_value, metric_context)
                VALUES (?, ?, ?)
            ''', (metric_name, value, json.dumps(context)))
            conn.commit()
    
    def should_avoid_response(self, user_input: str, proposed_response: str) -> bool:
        """Check if a response should be avoided based on learning"""
        for rule_key, rule in self.adaptation_rules.items():
            if rule['rule_type'] == 'avoid_response':
                if (rule['condition_pattern'] in user_input and 
                    rule['action_data']['avoid_response'] in proposed_response):
                    return True
        return False
    
    def get_personalization_data(self, user_id: str) -> Dict:
        """Get personalization data for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get user's interaction patterns
            cursor.execute('''
                SELECT context_data FROM user_feedback
                WHERE session_id LIKE ?
                ORDER BY timestamp DESC
                LIMIT 20
            ''', (f"%{user_id}%",))
            
            contexts = []
            for row in cursor.fetchall():
                try:
                    context = json.loads(row[0])
                    contexts.append(context)
                except:
                    continue
            
            # Analyze patterns
            preferred_response_style = self._analyze_response_preferences(contexts)
            common_intents = self._analyze_common_intents(contexts)
            
            return {
                'preferred_response_style': preferred_response_style,
                'common_intents': common_intents,
                'interaction_count': len(contexts)
            }
    
    def _analyze_response_preferences(self, contexts: List[Dict]) -> Dict:
        """Analyze user's response preferences"""
        # Simple analysis of preferred response characteristics
        style_preferences = {
            'formal': 0,
            'casual': 0,
            'empathetic': 0,
            'directive': 0
        }
        
        for context in contexts:
            sentiment = context.get('sentiment', {})
            if sentiment.get('label') == 'POSITIVE':
                # User responded positively, reinforce this style
                response_style = context.get('response_style', 'empathetic')
                if response_style in style_preferences:
                    style_preferences[response_style] += 1
        
        return style_preferences
    
    def _analyze_common_intents(self, contexts: List[Dict]) -> List[str]:
        """Analyze user's common intents"""
        intent_counts = {}
        
        for context in contexts:
            intent = context.get('intent', 'unknown')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Return top 3 most common intents
        sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
        return [intent for intent, count in sorted_intents[:3]]
    
    def batch_learning_update(self):
        """Perform batch learning updates"""
        logger.info("Starting batch learning update...")
        
        with self.learning_lock:
            if not self.feedback_buffer:
                return
            
            # Process accumulated feedback
            batch_data = self.feedback_buffer.copy()
            self.feedback_buffer.clear()
        
        # Analyze patterns in batch
        intent_performance = {}
        response_effectiveness = {}
        
        for data in batch_data:
            if data.user_feedback is not None:
                intent = data.context.get('intent', 'unknown')
                
                if intent not in intent_performance:
                    intent_performance[intent] = []
                intent_performance[intent].append(data.user_feedback)
                
                response_key = data.bot_response[:100]
                if response_key not in response_effectiveness:
                    response_effectiveness[response_key] = []
                response_effectiveness[response_key].append(data.user_feedback)
        
        # Update adaptation rules based on batch analysis
        for intent, scores in intent_performance.items():
            avg_score = np.mean(scores)
            if avg_score < 2.5:  # Poor performance
                self._create_improvement_rule(intent, avg_score)
        
        logger.info(f"Batch learning update completed for {len(batch_data)} interactions")
    
    def _create_improvement_rule(self, intent: str, performance_score: float):
        """Create improvement rule for poorly performing intents"""
        rule_data = {
            'intent': intent,
            'current_performance': performance_score,
            'improvement_needed': True
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO adaptation_rules 
                (rule_type, condition_pattern, action_type, action_data, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            ''', ('improve_intent', intent, 'enhance_response', 
                  json.dumps(rule_data), 0.7))
            conn.commit()

# Usage example
if __name__ == "__main__":
    learning_system = ContinuousLearningSystem()
    
    # Record some test interactions
    context = {'intent': 'depression', 'sentiment': {'label': 'NEGATIVE', 'score': 0.8}}
    
    learning_system.record_interaction(
        "test_session_1",
        "I feel very sad today",
        "I'm sorry to hear you're feeling sad. Can you tell me more about what's troubling you?",
        context,
        4.5  # Positive feedback
    )
    
    # Analyze trends
    trends = learning_system.analyze_learning_trends(days=7)
    print(f"Learning trends: {trends}")
    
    # Get improved response
    improved = learning_system.get_improved_response(
        "I feel sad", "depression", context
    )
    print(f"Improved response: {improved}")
