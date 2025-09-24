#!/usr/bin/env python3
"""
Enhanced Web Application for Mental Healthcare Chatbot

Features:
- Modern web interface with real-time chat
- Session management with persistence
- Crisis detection and intervention
- Analytics dashboard
- Admin panel for monitoring
- API endpoints for integration
- Security features and rate limiting
"""

import os
import json
import time
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Flask and related modules
try:
    from flask import Flask, render_template, request, jsonify, session, redirect, url_for
    from flask_cors import CORS
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    import jwt
    from werkzeug.security import generate_password_hash, check_password_hash
    WEB_AVAILABLE = True
    logger.info("All web packages available")
except ImportError as e:
    logger.error(f"Required web packages missing: {e}")
    exit(1)

# Import configuration
try:
    from config import config
    CONFIG_AVAILABLE = True
    logger.info("Configuration loaded successfully")
except ImportError as e:
    logger.warning(f"Configuration not available: {e}")
    CONFIG_AVAILABLE = False
    # Create basic config fallback
    class BasicConfig:
        class security:
            jwt_secret = 'fallback-secret-key'
            cors_origins = ['*']
            rate_limit_per_minute = 60
        class web:
            host = '0.0.0.0'
            port = 5000
            debug = False
        class crisis:
            emergency_contacts = {
                'US': {
                    'suicide_prevention_lifeline': '988',
                    'crisis_text_line': 'Text HOME to 741741',
                    'emergency_services': '911'
                }
            }
        class database:
            redis_enabled = False
    config = BasicConfig()

# Import our enhanced modules with fallback
ENHANCED_MODE = True
try:
    from chatbot import EnhancedMentalHealthChatbot
    logger.info("Enhanced chatbot available")
except ImportError as e:
    logger.warning(f"Enhanced chatbot not available: {e}")
    ENHANCED_MODE = False

try:
    from session_manager import EnhancedSessionManager
    logger.info("Enhanced session manager available")
except ImportError as e:
    logger.warning(f"Enhanced session manager not available: {e}")
    ENHANCED_MODE = False

try:
    from continuous_learning import EnhancedContinuousLearningSystem
    logger.info("Enhanced learning system available")
except ImportError as e:
    logger.warning(f"Enhanced learning system not available: {e}")
    ENHANCED_MODE = False

try:
    from nlp import AdvancedNLPProcessor, SafetyProtocols
    logger.info("Advanced NLP available")
except ImportError as e:
    logger.warning(f"Advanced NLP not available: {e}")
    ENHANCED_MODE = False

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configure CORS
CORS(app)

# Configure rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["1000 per hour", "60 per minute"] if ENHANCED_MODE else ["100 per hour"]
)

# Global components
chatbot = None
session_manager = None
learning_system = None
nlp_processor = None

def initialize_components():
    """Initialize chatbot components"""
    global chatbot, session_manager, learning_system, nlp_processor
    
    try:
        if ENHANCED_MODE:
            logger.info("Initializing enhanced components...")
            session_manager = EnhancedSessionManager(
                use_redis=config.database.redis_enabled
            )
            learning_system = EnhancedContinuousLearningSystem()
            nlp_processor = AdvancedNLPProcessor()
            chatbot = EnhancedMentalHealthChatbot(
                session_manager=session_manager,
                learning_system=learning_system,
                nlp_processor=nlp_processor
            )
            logger.info("Enhanced components initialized successfully")
        else:
            logger.info("Initializing basic components...")
            # Fallback initialization
            chatbot = BasicChatbot()
            session_manager = BasicSessionManager()
            
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        # Initialize minimal working system
        chatbot = FallbackChatbot()
        session_manager = FallbackSessionManager()

class BasicChatbot:
    """Basic fallback chatbot implementation"""
    def __init__(self):
        self.responses = {
            'greeting': "Hello! I'm here to listen and support you. How are you feeling today?",
            'crisis': "I'm very concerned about you. Please contact emergency services: 988 (Suicide & Crisis Lifeline) or your local emergency number.",
            'default': "I hear you. Can you tell me more about how you're feeling?"
        }
    
    def process_message(self, message: str, session_id: str) -> Dict:
        crisis_keywords = ['suicide', 'kill myself', 'end my life', 'want to die']
        
        if any(keyword in message.lower() for keyword in crisis_keywords):
            response = self.responses['crisis']
            is_crisis = True
        elif any(greeting in message.lower() for greeting in ['hi', 'hello', 'hey']):
            response = self.responses['greeting']
            is_crisis = False
        else:
            response = self.responses['default']
            is_crisis = False
        
        return {
            'response': response,
            'is_crisis': is_crisis,
            'confidence': 0.7,
            'analysis': {'basic_mode': True}
        }

class BasicSessionManager:
    """Basic session manager"""
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, user_id: str = None) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'user_id': user_id or 'anonymous',
            'created_at': datetime.now(),
            'messages': []
        }
        return session_id
    
    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

class FallbackChatbot:
    """Minimal fallback chatbot"""
    def process_message(self, message: str, session_id: str) -> Dict:
        return {
            'response': "Thank you for reaching out. If you're in crisis, please call 988 or your local emergency services.",
            'is_crisis': False,
            'confidence': 0.5,
            'analysis': {'fallback_mode': True}
        }

class FallbackSessionManager:
    """Minimal session manager"""
    def create_session(self, user_id: str = None) -> str:
        return str(uuid.uuid4())
    
    def get_session(self, session_id: str):
        return {'user_id': 'anonymous', 'messages': []}

# Routes
@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html', enhanced_mode=ENHANCED_MODE)

@app.route('/chat', methods=['POST'])
@limiter.limit("30 per minute")
def chat():
    """Process chat messages"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Create session if not provided
        if not session_id:
            session_id = session_manager.create_session()
        
        # Validate session
        session_data = session_manager.get_session(session_id)
        if not session_data:
            session_id = session_manager.create_session()
        
        # Process message
        start_time = time.time()
        response_data = chatbot.process_message(message, session_id)
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Prepare response
        response = {
            'response': response_data['response'],
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': round(processing_time, 2),
            'is_crisis': response_data.get('is_crisis', False),
            'confidence': response_data.get('confidence', 0.0)
        }
        
        # Add enhanced data if available
        if ENHANCED_MODE and 'analysis' in response_data:
            response['analysis'] = response_data['analysis']
            response['recommendations'] = response_data.get('recommendations', [])
        
        # Crisis handling
        if response_data.get('is_crisis', False):
            response['crisis_resources'] = get_crisis_resources()
            logger.critical(f"Crisis detected in session {session_id}: {message[:100]}...")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': 'An error occurred processing your message',
            'response': 'I apologize, but I encountered an error. If you need immediate help, please contact emergency services.',
            'is_crisis': False,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/session', methods=['POST'])
def create_session():
    """Create a new chat session"""
    try:
        data = request.json or {}
        user_id = data.get('user_id', 'anonymous')
        
        session_id = session_manager.create_session(user_id)
        
        return jsonify({
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({'error': 'Failed to create session'}), 500

@app.route('/session/<session_id>/history')
def get_session_history(session_id: str):
    """Get session conversation history"""
    try:
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        if hasattr(session_data, 'conversation_history'):
            history = session_data.conversation_history
        else:
            history = session_data.get('messages', [])
        
        return jsonify({
            'session_id': session_id,
            'history': history,
            'message_count': len(history)
        })
        
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        return jsonify({'error': 'Failed to retrieve session history'}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
    try:
        data = request.json
        session_id = data.get('session_id')
        feedback_score = data.get('score', 0.5)
        feedback_text = data.get('text', '')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        feedback_data = {
            'session_id': session_id,
            'score': feedback_score,
            'text': feedback_text,
            'timestamp': datetime.now().isoformat()
        }
        
        # Record feedback if enhanced mode available
        if ENHANCED_MODE and hasattr(learning_system, 'record_user_feedback'):
            learning_system.record_user_feedback(feedback_data)
        
        return jsonify({'message': 'Feedback recorded', 'feedback_id': str(uuid.uuid4())})
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return jsonify({'error': 'Failed to record feedback'}), 500

@app.route('/analytics')
@limiter.limit("10 per minute")
def analytics_dashboard():
    """Analytics dashboard (admin only)"""
    if not ENHANCED_MODE:
        return render_template('analytics_unavailable.html')
    
    try:
        # Get analytics data
        analytics_data = {}
        
        if hasattr(session_manager, 'get_session_analytics'):
            analytics_data['sessions'] = session_manager.get_session_analytics('24h')
        
        if hasattr(learning_system, 'get_learning_analytics'):
            analytics_data['learning'] = learning_system.get_learning_analytics()
        
        # Get crisis statistics
        if hasattr(session_manager, 'get_crisis_sessions'):
            crisis_sessions = session_manager.get_crisis_sessions()
            analytics_data['crisis'] = {
                'active_crisis_sessions': len(crisis_sessions),
                'crisis_sessions': crisis_sessions[:10]  # Latest 10
            }
        
        return render_template('analytics.html', data=analytics_data)
        
    except Exception as e:
        logger.error(f"Error generating analytics: {e}")
        return render_template('error.html', message="Failed to load analytics"), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'enhanced_mode': ENHANCED_MODE,
        'version': '2.0.0'
    })

@app.route('/api/status')
def system_status():
    """System status endpoint"""
    status = {
        'enhanced_mode': ENHANCED_MODE,
        'components': {
            'chatbot': chatbot is not None,
            'session_manager': session_manager is not None,
            'learning_system': learning_system is not None and ENHANCED_MODE,
            'nlp_processor': nlp_processor is not None and ENHANCED_MODE
        },
        'timestamp': datetime.now().isoformat()
    }
    
    if ENHANCED_MODE:
        try:
            # Add enhanced status information
            if hasattr(session_manager, 'session_metrics'):
                status['metrics'] = session_manager.session_metrics
            
            if hasattr(learning_system, 'learning_metrics'):
                status['learning_metrics'] = learning_system.learning_metrics
                
        except Exception as e:
            logger.error(f"Error getting enhanced status: {e}")
            status['error'] = 'Failed to get enhanced status'
    
    return jsonify(status)

def get_crisis_resources():
    """Get crisis resources and contacts"""
    if ENHANCED_MODE:
        return config.crisis.emergency_contacts
    else:
        return {
            'emergency': {
                'suicide_prevention_lifeline': {
                    'number': '988',
                    'description': 'National Suicide Prevention Lifeline'
                },
                'crisis_text_line': {
                    'number': '741741',
                    'text': 'Text HOME to 741741',
                    'description': 'Crisis Text Line'
                }
            }
        }

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', 
                         message="Page not found", 
                         status_code=404), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return render_template('error.html', 
                         message="Internal server error", 
                         status_code=500), 500

@app.errorhandler(429)
def rate_limit_handler(e):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': str(e)
    }), 429

# Initialize components before first request
@app.before_first_request
def before_first_request():
    """Initialize components when the app starts"""
    initialize_components()
    logger.info("Enhanced Mental Healthcare Chatbot started")
    
    # Perform basic health checks
    if chatbot is None:
        logger.error("Failed to initialize chatbot")
    else:
        logger.info("Chatbot initialized successfully")
    
    if session_manager is None:
        logger.error("Failed to initialize session manager")
    else:
        logger.info("Session manager initialized successfully")

if __name__ == '__main__':
    # Initialize components
    initialize_components()
    
    # Run the application
    host = config.web.host if ENHANCED_MODE else '0.0.0.0'
    port = config.web.port if ENHANCED_MODE else 5000
    debug = config.web.debug if ENHANCED_MODE else False
    
    logger.info(f"Starting Enhanced Mental Healthcare Chatbot on {host}:{port}")
    logger.info(f"Enhanced mode: {ENHANCED_MODE}")
    
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )
