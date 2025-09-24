import sqlite3
import uuid
import datetime
import threading
import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import redis
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SessionData:
    """Data class to store session information"""
    session_id: str
    user_id: str
    start_time: datetime.datetime
    last_interaction: datetime.datetime
    conversation_history: List[Dict]
    user_state: Dict
    risk_level: str
    assessment_scores: Dict
    is_active: bool
    emotional_state_history: List[Dict]
    intervention_history: List[Dict]
    preferences: Dict

class EnhancedSessionManager:
    """Enhanced session manager with Redis caching, advanced security, and analytics"""
    
    def __init__(self, 
                 db_path: str = "sessions.db", 
                 encryption_key: Optional[bytes] = None,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 use_redis: bool = False):
        
        self.db_path = db_path
        self.active_sessions: Dict[str, SessionData] = {}
        self.session_lock = threading.Lock()
        self.use_redis = use_redis
        
        # Initialize encryption
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(encryption_key)
        
        # Initialize Redis if available
        if use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host, 
                    port=redis_port, 
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                # Test Redis connection
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis not available, falling back to memory storage: {e}")
                self.use_redis = False
                self.redis_client = None
        else:
            self.redis_client = None
        
        # Initialize database
        self._init_database()
        
        # Load active sessions from database
        self._load_active_sessions()
        
        # Session analytics
        self.session_metrics = {
            'total_sessions': 0,
            'active_sessions': 0,
            'average_session_duration': 0.0,
            'crisis_interventions': 0,
            'user_satisfaction': 0.0
        }
    
    def _init_database(self):
        """Initialize the SQLite database for session storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enhanced sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    last_interaction TEXT NOT NULL,
                    conversation_history TEXT,
                    user_state TEXT,
                    risk_level TEXT DEFAULT 'low',
                    assessment_scores TEXT,
                    emotional_state_history TEXT,
                    intervention_history TEXT,
                    preferences TEXT,
                    session_duration_minutes INTEGER,
                    message_count INTEGER DEFAULT 0,
                    crisis_events_count INTEGER DEFAULT 0,
                    user_satisfaction_score REAL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Enhanced interactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_input TEXT,
                    user_input_encrypted BLOB,
                    bot_response TEXT,
                    sentiment_score REAL,
                    emotion_detected TEXT,
                    intent_detected TEXT,
                    confidence_score REAL,
                    crisis_score REAL,
                    intervention_triggered BOOLEAN DEFAULT 0,
                    response_time_ms INTEGER,
                    user_feedback_score INTEGER,
                    context_data TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Enhanced crisis events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crisis_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_id TEXT,
                    crisis_type TEXT,
                    risk_score REAL,
                    risk_level TEXT,
                    crisis_indicators TEXT,
                    intervention_type TEXT,
                    intervention_details TEXT,
                    resolution_status TEXT,
                    follow_up_required BOOLEAN DEFAULT 1,
                    follow_up_completed BOOLEAN DEFAULT 0,
                    escalation_level TEXT DEFAULT 'standard',
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # User analytics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    metric_metadata TEXT,
                    aggregation_period TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Session security logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    event_type TEXT,
                    event_details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    risk_assessment TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def create_session(self, user_id: str, initial_context: Dict = None) -> str:
        """Create a new session with enhanced security and context tracking"""
        session_id = str(uuid.uuid4())
        current_time = datetime.datetime.now()
        
        session_data = SessionData(
            session_id=session_id,
            user_id=user_id,
            start_time=current_time,
            last_interaction=current_time,
            conversation_history=[],
            user_state=initial_context or {},
            risk_level='low',
            assessment_scores={},
            is_active=True,
            emotional_state_history=[],
            intervention_history=[],
            preferences={}
        )
        
        with self.session_lock:
            self.active_sessions[session_id] = session_data
            
            # Cache in Redis if available
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.setex(
                        f"session:{session_id}",
                        3600,  # 1 hour TTL
                        json.dumps(asdict(session_data), default=str)
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache session in Redis: {e}")
        
        # Persist to database
        self._save_session_to_db(session_data)
        
        # Update metrics
        self.session_metrics['total_sessions'] += 1
        self.session_metrics['active_sessions'] += 1
        
        logger.info(f"Created new session {session_id} for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data with caching support"""
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Check Redis cache
        if self.use_redis and self.redis_client:
            try:
                cached_session = self.redis_client.get(f"session:{session_id}")
                if cached_session:
                    session_dict = json.loads(cached_session)
                    session_data = self._dict_to_session_data(session_dict)
                    self.active_sessions[session_id] = session_data
                    return session_data
            except Exception as e:
                logger.warning(f"Failed to retrieve session from Redis: {e}")
        
        # Load from database
        session_data = self._load_session_from_db(session_id)
        if session_data:
            self.active_sessions[session_id] = session_data
        
        return session_data
    
    def update_session(self, session_id: str, updates: Dict) -> bool:
        """Update session data with optimistic locking"""
        session_data = self.get_session(session_id)
        if not session_data:
            return False
        
        with self.session_lock:
            # Apply updates
            for key, value in updates.items():
                if hasattr(session_data, key):
                    setattr(session_data, key, value)
            
            session_data.last_interaction = datetime.datetime.now()
            
            # Update cache
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.setex(
                        f"session:{session_id}",
                        3600,
                        json.dumps(asdict(session_data), default=str)
                    )
                except Exception as e:
                    logger.warning(f"Failed to update session cache: {e}")
        
        # Persist to database
        self._save_session_to_db(session_data)
        return True
    
    def add_interaction(self, session_id: str, user_input: str, bot_response: str, 
                       analysis_result: Dict, response_time_ms: int = 0) -> bool:
        """Add interaction with comprehensive logging and analytics"""
        session_data = self.get_session(session_id)
        if not session_data:
            return False
        
        interaction = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'analysis': analysis_result,
            'response_time_ms': response_time_ms
        }
        
        with self.session_lock:
            session_data.conversation_history.append(interaction)
            
            # Update emotional state history
            if 'emotions' in analysis_result:
                emotional_state = {
                    'timestamp': interaction['timestamp'],
                    'primary_emotion': analysis_result['emotions'].get('primary_emotion'),
                    'confidence': analysis_result['emotions'].get('confidence', 0),
                    'sentiment_score': analysis_result.get('sentiment', {}).get('compound', 0)
                }
                session_data.emotional_state_history.append(emotional_state)
            
            # Track crisis events
            crisis_assessment = analysis_result.get('crisis_assessment', {})
            if crisis_assessment.get('is_crisis'):
                intervention = {
                    'timestamp': interaction['timestamp'],
                    'crisis_type': crisis_assessment.get('crisis_type'),
                    'risk_level': crisis_assessment.get('risk_level'),
                    'risk_score': crisis_assessment.get('risk_score'),
                    'intervention_triggered': crisis_assessment.get('intervention_needed')
                }
                session_data.intervention_history.append(intervention)
                session_data.risk_level = crisis_assessment.get('risk_level', session_data.risk_level)
        
        # Log interaction to database
        self._log_interaction_to_db(session_id, user_input, bot_response, analysis_result, response_time_ms)
        
        # Update session in storage
        self.update_session(session_id, {'conversation_history': session_data.conversation_history})
        
        return True
    
    def end_session(self, session_id: str, user_feedback: Optional[Dict] = None) -> bool:
        """End session with analytics and cleanup"""
        session_data = self.get_session(session_id)
        if not session_data:
            return False
        
        with self.session_lock:
            session_data.is_active = False
            end_time = datetime.datetime.now()
            duration = (end_time - session_data.start_time).total_seconds() / 60  # minutes
            
            # Update session analytics
            session_analytics = {
                'end_time': end_time.isoformat(),
                'duration_minutes': duration,
                'message_count': len(session_data.conversation_history),
                'crisis_events_count': len(session_data.intervention_history),
                'user_satisfaction': user_feedback.get('satisfaction_score', 0) if user_feedback else 0
            }
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Remove from Redis cache
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.delete(f"session:{session_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove session from Redis: {e}")
        
        # Final database update
        self._finalize_session_in_db(session_id, session_analytics)
        
        # Update global metrics
        self.session_metrics['active_sessions'] -= 1
        
        logger.info(f"Session {session_id} ended after {duration:.1f} minutes")
        return True
    
    def get_user_session_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's session history for personalization"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT session_id, start_time, end_time, duration_minutes, 
                       message_count, risk_level, user_satisfaction_score
                FROM sessions 
                WHERE user_id = ? 
                ORDER BY start_time DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'session_id': row[0],
                    'start_time': row[1],
                    'end_time': row[2],
                    'duration_minutes': row[3],
                    'message_count': row[4],
                    'risk_level': row[5],
                    'satisfaction_score': row[6]
                })
            
            return history
    
    def get_session_analytics(self, time_period: str = '24h') -> Dict:
        """Get comprehensive session analytics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Calculate time filter
            if time_period == '24h':
                time_filter = (datetime.datetime.now() - datetime.timedelta(hours=24)).isoformat()
            elif time_period == '7d':
                time_filter = (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat()
            elif time_period == '30d':
                time_filter = (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat()
            else:
                time_filter = '1970-01-01'  # All time
            
            # Get session statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_sessions,
                    AVG(duration_minutes) as avg_duration,
                    AVG(message_count) as avg_messages,
                    AVG(user_satisfaction_score) as avg_satisfaction,
                    COUNT(CASE WHEN crisis_events_count > 0 THEN 1 END) as sessions_with_crisis
                FROM sessions 
                WHERE created_at >= ?
            ''', (time_filter,))
            
            stats = cursor.fetchone()
            
            return {
                'time_period': time_period,
                'total_sessions': stats[0] or 0,
                'average_duration_minutes': round(stats[1] or 0, 2),
                'average_messages_per_session': round(stats[2] or 0, 1),
                'average_satisfaction_score': round(stats[3] or 0, 2),
                'sessions_with_crisis_events': stats[4] or 0,
                'crisis_rate': round((stats[4] or 0) / (stats[0] or 1) * 100, 2),
                'current_active_sessions': len(self.active_sessions)
            }
    
    def _dict_to_session_data(self, session_dict: Dict) -> SessionData:
        """Convert dictionary to SessionData object"""
        # Handle datetime conversion
        session_dict['start_time'] = datetime.datetime.fromisoformat(session_dict['start_time'])
        session_dict['last_interaction'] = datetime.datetime.fromisoformat(session_dict['last_interaction'])
        
        return SessionData(**session_dict)
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        finally:
            if conn:
                conn.close()
    
    def _encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode())
    
    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data).decode()
    
    def create_session(self, user_id: str = None) -> str:
        """Create a new session for a user"""
        session_id = str(uuid.uuid4())
        if user_id is None:
            user_id = f"anonymous_{str(uuid.uuid4())[:8]}"
        
        current_time = datetime.datetime.now()
        
        session_data = SessionData(
            session_id=session_id,
            user_id=user_id,
            start_time=current_time,
            last_interaction=current_time,
            conversation_history=[],
            user_state={
                'name': None,
                'mood_history': [],
                'assessment_completed': False,
                'crisis_flags': [],
                'preferences': {}
            },
            risk_level='low',
            assessment_scores={},
            is_active=True
        )
        
        with self.session_lock:
            self.active_sessions[session_id] = session_data
        
        # Save to database
        self._save_session_to_db(session_data)
        
        logger.info(f"Created new session: {session_id} for user: {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve session data"""
        with self.session_lock:
            return self.active_sessions.get(session_id)
    
    def update_session(self, session_id: str, **kwargs):
        """Update session data"""
        with self.session_lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.last_interaction = datetime.datetime.now()
                
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                
                # Save to database
                self._save_session_to_db(session)
    
    def add_interaction(self, session_id: str, user_input: str, bot_response: str, 
                       sentiment_score: float = 0.0, intent_detected: str = 'unknown'):
        """Add interaction to session history and database"""
        interaction_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'sentiment_score': sentiment_score,
            'intent_detected': intent_detected
        }
        
        with self.session_lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.conversation_history.append(interaction_data)
                session.last_interaction = datetime.datetime.now()
        
        # Save interaction to database for learning
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interactions 
                (session_id, user_input, bot_response, sentiment_score, intent_detected)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, self._encrypt_data(user_input), self._encrypt_data(bot_response), 
                  sentiment_score, intent_detected))
            conn.commit()
    
    def log_crisis_event(self, session_id: str, crisis_type: str, risk_score: float, 
                        intervention_taken: str):
        """Log crisis events for monitoring and learning"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO crisis_events 
                (session_id, crisis_type, risk_score, intervention_taken, resolution_status)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, crisis_type, risk_score, intervention_taken, 'ongoing'))
            conn.commit()
        
        logger.warning(f"Crisis event logged: {crisis_type} for session {session_id}")
    
    def end_session(self, session_id: str):
        """End a session and mark it as inactive"""
        with self.session_lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.is_active = False
                self._save_session_to_db(session)
                del self.active_sessions[session_id]
        
        logger.info(f"Session ended: {session_id}")
    
    def _save_session_to_db(self, session_data: SessionData):
        """Save session to database"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO sessions 
                (session_id, user_id, start_time, last_interaction, 
                 conversation_history, user_state, risk_level, assessment_scores,
                 emotional_state_history, intervention_history, preferences, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_data.session_id,
                session_data.user_id,
                session_data.start_time.isoformat(),
                session_data.last_interaction.isoformat(),
                json.dumps(session_data.conversation_history),
                json.dumps(session_data.user_state),
                session_data.risk_level,
                json.dumps(session_data.assessment_scores),
                json.dumps(session_data.emotional_state_history),
                json.dumps(session_data.intervention_history),
                json.dumps(session_data.preferences),
                session_data.is_active
            ))
            conn.commit()
    
    def _load_session_from_db(self, session_id: str) -> Optional[SessionData]:
        """Load session from database"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT session_id, user_id, start_time, last_interaction,
                       conversation_history, user_state, risk_level, assessment_scores,
                       emotional_state_history, intervention_history, preferences, is_active
                FROM sessions WHERE session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            if row:
                return SessionData(
                    session_id=row[0],
                    user_id=row[1],
                    start_time=datetime.datetime.fromisoformat(row[2]),
                    last_interaction=datetime.datetime.fromisoformat(row[3]),
                    conversation_history=json.loads(row[4]) if row[4] else [],
                    user_state=json.loads(row[5]) if row[5] else {},
                    risk_level=row[6],
                    assessment_scores=json.loads(row[7]) if row[7] else {},
                    emotional_state_history=json.loads(row[8]) if row[8] else [],
                    intervention_history=json.loads(row[9]) if row[9] else [],
                    preferences=json.loads(row[10]) if row[10] else {},
                    is_active=bool(row[11])
                )
        return None
    
    def _load_active_sessions(self):
        """Load active sessions from database"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT session_id FROM sessions WHERE is_active = 1')
            for row in cursor.fetchall():
                session_id = row[0]
                session_data = self._load_session_from_db(session_id)
                if session_data:
                    self.active_sessions[session_id] = session_data
    
    def _log_interaction_to_db(self, session_id: str, user_input: str, bot_response: str, 
                              analysis_result: Dict, response_time_ms: int):
        """Log interaction to database"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Extract analysis data
            sentiment = analysis_result.get('sentiment', {})
            emotions = analysis_result.get('emotions', {})
            intent = analysis_result.get('intent_classification', {})
            crisis = analysis_result.get('crisis_assessment', {})
            
            cursor.execute('''
                INSERT INTO interactions 
                (session_id, user_input, user_input_encrypted, bot_response,
                 sentiment_score, emotion_detected, intent_detected, 
                 confidence_score, crisis_score, intervention_triggered,
                 response_time_ms, context_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                user_input,
                self._encrypt_data(user_input),
                bot_response,
                sentiment.get('compound', 0),
                emotions.get('primary_emotion', 'unknown'),
                intent.get('primary_intent', 'unknown'),
                emotions.get('confidence', 0),
                crisis.get('risk_score', 0),
                crisis.get('intervention_needed', False),
                response_time_ms,
                json.dumps(analysis_result)
            ))
            conn.commit()
    
    def _finalize_session_in_db(self, session_id: str, analytics: Dict):
        """Finalize session in database"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions 
                SET end_time = ?, session_duration_minutes = ?, 
                    message_count = ?, crisis_events_count = ?,
                    user_satisfaction_score = ?, is_active = 0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            ''', (
                analytics.get('end_time'),
                analytics.get('duration_minutes', 0),
                analytics.get('message_count', 0),
                analytics.get('crisis_events_count', 0),
                analytics.get('user_satisfaction', 0),
                session_id
            ))
            conn.commit()
    
    def cleanup_expired_sessions(self, hours: int = 24):
        """Clean up expired sessions"""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions 
                SET is_active = 0 
                WHERE last_interaction < ? AND is_active = 1
            ''', (cutoff_time.isoformat(),))
            
            expired_count = cursor.rowcount
            conn.commit()
        
        # Remove from active sessions
        expired_sessions = []
        for session_id, session_data in self.active_sessions.items():
            if session_data.last_interaction < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.delete(f"session:{session_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove expired session from Redis: {e}")
        
        logger.info(f"Cleaned up {expired_count} expired sessions")
        return expired_count
    
    def get_crisis_sessions(self) -> List[Dict]:
        """Get all sessions with active crisis events"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT s.session_id, s.user_id, s.risk_level, 
                       ce.crisis_type, ce.risk_score, ce.timestamp
                FROM sessions s
                INNER JOIN crisis_events ce ON s.session_id = ce.session_id
                WHERE s.is_active = 1 AND ce.resolution_status != 'resolved'
                ORDER BY ce.risk_score DESC, ce.timestamp DESC
            ''')
            
            crisis_sessions = []
            for row in cursor.fetchall():
                crisis_sessions.append({
                    'session_id': row[0],
                    'user_id': row[1],
                    'risk_level': row[2],
                    'crisis_type': row[3],
                    'risk_score': row[4],
                    'timestamp': row[5]
                })
            
            return crisis_sessions

# Alias for backward compatibility
SessionManager = EnhancedSessionManager

# Usage example and testing
if __name__ == "__main__":
    # Create session manager
    session_manager = SessionManager()
    
    # Create a test session
    session_id = session_manager.create_session("test_user")
    
    # Add some interactions
    session_manager.add_interaction(
        session_id, 
        "I feel sad today", 
        "I'm sorry to hear that. Can you tell me more about what's making you feel sad?",
        sentiment_score=-0.7,
        intent_detected="depression"
    )
    
    # Test crisis logging
    session_manager.log_crisis_event(
        session_id,
        "suicidal_ideation",
        0.8,
        "emergency_contact_provided"
    )
    
    # Get session data
    session = session_manager.get_session(session_id)
    print(f"Session created: {session.session_id}")
    print(f"Conversation history: {len(session.conversation_history)} interactions")
    
    # End session
    session_manager.end_session(session_id)
