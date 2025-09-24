#!/usr/bin/env python3
"""
Enhanced Configuration System for Mental Healthcare Chatbot

This module provides a comprehensive configuration system with proper
structure and validation for the enhanced chatbot system.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    sessions_db: str = 'sessions.db'
    learning_db: str = 'learning.db'
    encryption_enabled: bool = True
    backup_interval_hours: int = 24
    redis_enabled: bool = False
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_password: Optional[str] = None


@dataclass
class NLPConfig:
    """NLP configuration settings"""
    use_advanced_nlp: bool = True
    sentiment_model: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    emotion_model: str = 'j-hartmann/emotion-english-distilroberta-base'
    sentence_transformer: str = 'all-MiniLM-L6-v2'
    fallback_to_basic: bool = True
    max_sequence_length: int = 512
    batch_size: int = 16


@dataclass
class SafetyConfig:
    """Safety and crisis detection configuration"""
    crisis_detection_threshold: float = 0.6
    auto_escalate_threshold: float = 0.8
    require_professional_referral: bool = True
    log_all_crisis_events: bool = True
    max_crisis_responses: int = 3
    cooldown_period_minutes: int = 30


@dataclass
class LearningConfig:
    """Continuous learning configuration"""
    enable_continuous_learning: bool = True
    feedback_collection: bool = True
    batch_learning_interval_hours: int = 6
    min_interactions_for_learning: int = 10
    personalization_enabled: bool = True
    max_memory_size: int = 10000
    learning_rate: float = 0.001


@dataclass
class SessionConfig:
    """Session management configuration"""
    max_session_duration_hours: int = 4
    auto_save_interval_minutes: int = 5
    cleanup_old_sessions_days: int = 30
    max_concurrent_sessions: int = 100
    session_timeout_minutes: int = 30
    max_messages_per_session: int = 200


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
    cors_origins: Optional[List[str]] = None
    rate_limit_per_minute: int = 60
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ['http://localhost:3000', 'http://localhost:5000']


@dataclass
class WebConfig:
    """Web application configuration"""
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = False
    template_folder: str = 'templates'
    static_folder: str = 'static'


@dataclass
class CrisisConfig:
    """Crisis intervention configuration"""
    emergency_contacts: Optional[Dict] = None
    
    def __post_init__(self):
        if self.emergency_contacts is None:
            self.emergency_contacts = {
                'US': {
                    'suicide_prevention_lifeline': '988',
                    'crisis_text_line': 'Text HOME to 741741',
                    'emergency_services': '911'
                },
                'India': {
                    'aasra': '91-22-27546669',
                    'sneha_foundation': '91-44-24640050',
                    'sumaitri': '91-11-23389090',
                    'jeevan': '91-44-26564444'
                },
                'UK': {
                    'samaritans': '116 123',
                    'emergency_services': '999'
                },
                'Australia': {
                    'lifeline': '13 11 14',
                    'emergency_services': '000'
                }
            }


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = 'INFO'
    log_file: str = 'chatbot.log'
    max_log_size_mb: int = 50
    backup_count: int = 5
    enable_console_logging: bool = True


@dataclass
class ConfigSystem:
    """Main configuration system"""
    database: Optional[DatabaseConfig] = None
    nlp: Optional[NLPConfig] = None
    safety: Optional[SafetyConfig] = None
    learning: Optional[LearningConfig] = None
    session: Optional[SessionConfig] = None
    security: Optional[SecurityConfig] = None
    web: Optional[WebConfig] = None
    crisis: Optional[CrisisConfig] = None
    logging: Optional[LoggingConfig] = None
    
    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig()
        if self.nlp is None:
            self.nlp = NLPConfig()
        if self.safety is None:
            self.safety = SafetyConfig()
        if self.learning is None:
            self.learning = LearningConfig()
        if self.session is None:
            self.session = SessionConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.web is None:
            self.web = WebConfig()
        if self.crisis is None:
            self.crisis = CrisisConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


# Global configuration instance
config = ConfigSystem()

# Response templates
RESPONSE_TEMPLATES = {
    'crisis_high': """🚨 I'm very concerned about your safety. Please reach out for immediate help:

**EMERGENCY RESOURCES:**
• National Suicide Prevention Lifeline: 988 (US)
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911

You don't have to face this alone. Please reach out now.""",
    
    'crisis_medium': """I'm concerned about what you've shared. There are people who want to help:

**24/7 Support Available:**
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741

Can you reach out to one of these resources right now?""",
    
    'supportive_ending': """Thank you for talking with me today. Remember:

• You've taken a positive step by reaching out
• Your feelings are valid and temporary
• Support is available whenever you need it
• You deserve care and compassion

Take care of yourself. 💙"""
}

# Legacy configurations for backward compatibility
def get_database_config():
    assert config.database is not None
    return {
        'sessions_db': config.database.sessions_db,
        'learning_db': config.database.learning_db,
        'encryption_enabled': config.database.encryption_enabled,
        'backup_interval_hours': config.database.backup_interval_hours
    }

def get_nlp_config():
    assert config.nlp is not None
    return {
        'use_advanced_nlp': config.nlp.use_advanced_nlp,
        'sentiment_model': config.nlp.sentiment_model,
        'emotion_model': config.nlp.emotion_model,
        'sentence_transformer': config.nlp.sentence_transformer,
        'fallback_to_basic': config.nlp.fallback_to_basic
    }

def get_safety_config():
    assert config.safety is not None
    return {
        'crisis_detection_threshold': config.safety.crisis_detection_threshold,
        'auto_escalate_threshold': config.safety.auto_escalate_threshold,
        'require_professional_referral': config.safety.require_professional_referral,
        'log_all_crisis_events': config.safety.log_all_crisis_events
    }

def get_learning_config():
    assert config.learning is not None
    return {
        'enable_continuous_learning': config.learning.enable_continuous_learning,
        'feedback_collection': config.learning.feedback_collection,
        'batch_learning_interval_hours': config.learning.batch_learning_interval_hours,
        'min_interactions_for_learning': config.learning.min_interactions_for_learning,
        'personalization_enabled': config.learning.personalization_enabled
    }

def get_session_config():
    assert config.session is not None
    return {
        'max_session_duration_hours': config.session.max_session_duration_hours,
        'auto_save_interval_minutes': config.session.auto_save_interval_minutes,
        'cleanup_old_sessions_days': config.session.cleanup_old_sessions_days,
        'max_concurrent_sessions': config.session.max_concurrent_sessions
    }

# Lazy initialization of legacy configs
DATABASE_CONFIG = None
NLP_CONFIG = None
SAFETY_CONFIG = None
LEARNING_CONFIG = None
SESSION_CONFIG = None

def _init_legacy_configs():
    global DATABASE_CONFIG, NLP_CONFIG, SAFETY_CONFIG, LEARNING_CONFIG, SESSION_CONFIG
    if DATABASE_CONFIG is None:
        DATABASE_CONFIG = get_database_config()
        NLP_CONFIG = get_nlp_config()
        SAFETY_CONFIG = get_safety_config()
        LEARNING_CONFIG = get_learning_config()
        SESSION_CONFIG = get_session_config()

# Initialize on import
_init_legacy_configs()

# Emergency Contacts
EMERGENCY_CONTACTS = {
    'primary': {
        'US': {
            'suicide_prevention_lifeline': '988',
            'crisis_text_line': 'Text HOME to 741741',
            'emergency_services': '911'
        },
        'India': {
            'aasra': '91-22-27546669',
            'sneha_foundation': '91-44-24640050',
            'sumaitri': '91-11-23389090',
            'jeevan': '91-44-26564444'
        },
        'UK': {
            'samaritans': '116 123',
            'emergency_services': '999'
        },
        'Australia': {
            'lifeline': '13 11 14',
            'emergency_services': '000'
        }
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'log_file': 'chatbot.log',
    'max_log_size_mb': 50,
    'backup_count': 5,
    'enable_console_logging': True
}

# Response Templates
RESPONSE_TEMPLATES = {
    'crisis_high': """🚨 I'm very concerned about your safety. Please reach out for immediate help:

**EMERGENCY RESOURCES:**
• National Suicide Prevention Lifeline: 988 (US)
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911

You don't have to face this alone. Please reach out now.""",
    
    'crisis_medium': """I'm concerned about what you've shared. There are people who want to help:

**24/7 Support Available:**
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741

Can you reach out to one of these resources right now?""",
    
    'supportive_ending': """Thank you for talking with me today. Remember:

• You've taken a positive step by reaching out
• Support is always available when you need it
• Your mental health matters

Take care of yourself. 🌟"""
}

# Privacy Settings
PRIVACY_CONFIG = {
    'encrypt_user_data': True,
    'anonymize_logs': True,
    'data_retention_days': 90,
    'require_consent': True,
    'gdpr_compliant': True
}
