#!/usr/bin/env python3
"""
Enhanced Mental Healthcare Chatbot with Modern NLP and Safety Protocols

This chatbot incorporates:
1. Session management with secure data handling
2. Advanced NLP using transformer models (with fallback to basic models)
3. Comprehensive safety protocols for crisis situations
4. Continuous learning from user interactions
5. Personalized responses based on user history
"""

import sys
import os
import time
import logging
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our custom modules (with fallback handling)
try:
    from session_manager import SessionManager
    from continuous_learning import ContinuousLearningSystem
    SESSION_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Session management not available: {e}")
    SESSION_MANAGEMENT_AVAILABLE = False

try:
    from nlp import AdvancedNLPProcessor, SafetyProtocols
    ADVANCED_NLP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced NLP not available, using fallback: {e}")
    ADVANCED_NLP_AVAILABLE = False

# Fallback imports for basic functionality
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class EnhancedMentalHealthChatbot:
    """Enhanced Mental Healthcare Chatbot with modern features"""
    
    def __init__(self, load_data: bool = True):
        """Initialize the enhanced chatbot"""
        logger.info("Initializing Enhanced Mental Healthcare Chatbot...")
        
        # Initialize components
        self.session_manager = SessionManager() if SESSION_MANAGEMENT_AVAILABLE else None
        self.learning_system = ContinuousLearningSystem() if SESSION_MANAGEMENT_AVAILABLE else None
        self.nlp_processor = AdvancedNLPProcessor() if ADVANCED_NLP_AVAILABLE else None
        self.safety_protocols = SafetyProtocols() if ADVANCED_NLP_AVAILABLE else None
        
        # Fallback NLP components
        self.vectorizer = None
        self.model = None
        self.encoder = None
        
        # Enhanced intents and responses
        self.intents = {
            'greeting': ['hello', 'hey', 'hi', 'good morning', 'good afternoon', 'good evening'],
            'goodbye': ['goodbye', 'bye', 'see you later', 'farewell', 'take care'],
            'depression': ['depressed', 'sad', 'hopeless', 'worthless', 'empty', 'down', 'despair', 'misery'],
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'fear', 'stressed', 'tension', 'overwhelmed'],
            'suicidal_ideation': ['suicide', 'kill myself', 'end my life', 'want to die', 'better off dead'],
            'self_harm': ['cut myself', 'hurt myself', 'self harm', 'cutting', 'burning myself'],
            'substance_abuse': ['drinking', 'drugs', 'high', 'drunk', 'addiction', 'alcohol', 'substance'],
            'relationship_issues': ['relationship', 'boyfriend', 'girlfriend', 'marriage', 'family problems'],
            'work_stress': ['work', 'job', 'boss', 'career', 'unemployment', 'workplace'],
            'grief': ['loss', 'death', 'died', 'grief', 'mourning', 'bereavement'],
            'trauma': ['trauma', 'abuse', 'assault', 'ptsd', 'flashbacks'],
            'eating_disorder': ['eating', 'food', 'weight', 'anorexia', 'bulimia', 'binge'],
            'sleep_issues': ['sleep', 'insomnia', 'nightmares', 'tired', 'exhausted'],
            'seeking_help': ['help', 'therapist', 'counselor', 'therapy', 'treatment', 'professional'],
            'positive': ['good', 'great', 'happy', 'better', 'improved', 'fine', 'okay']
        }
        
        self.responses = {
            'greeting': "Hello! I'm here to provide mental health support. I'm a safe space where you can share your thoughts and feelings. How are you doing today?",
            'goodbye': "Thank you for talking with me today. Remember, support is always available when you need it. Take care of yourself.",
            'depression': "I hear that you're struggling with difficult feelings. Depression can feel overwhelming, but you're not alone. Would you like to talk about what's been weighing on your mind?",
            'anxiety': "Anxiety can be really challenging to deal with. Your feelings are valid. Let's work through this together. Can you tell me what's been making you feel anxious?",
            'seeking_help': "It's a positive step that you're seeking help. Professional support can make a real difference. Would you like information about finding a therapist or counselor?",
            'positive': "I'm glad to hear you're doing well! It's important to acknowledge and appreciate the good moments. What's been going well for you?",
            'default': "I want to understand how you're feeling. Can you tell me more about what's on your mind? Remember, this is a safe space."
        }
        
        # Crisis response templates
        self.crisis_responses = {
            'immediate_danger': """🚨 I'm very concerned about your safety. Please reach out for immediate help:

**EMERGENCY RESOURCES:**
• National Suicide Prevention Lifeline: 988 (US)
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911
• Go to your nearest emergency room

**International Resources:**
• India - AASRA: 91-22-27546669
• India - Sneha Foundation: 91-44-24640050
• UK - Samaritans: 116 123
• Australia - Lifeline: 13 11 14

You don't have to face this alone. Please reach out now.""",
            
            'high_risk': """I'm concerned about what you've shared. Your life has value and there are people who want to help:

**24/7 Support Available:**
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• SAMHSA National Helpline: 1-800-662-4357

Can you reach out to one of these resources or call a trusted friend or family member right now?""",
            
            'self_harm': """I'm worried about you harming yourself. Your safety is important:

**Immediate Steps:**
• Remove any items you might use to hurt yourself
• Call someone you trust
• Go to a public place if you're alone
• Contact: Crisis Text Line (Text HOME to 741741)

Would you be willing to call someone right now?"""
        }
        
        # Load training data if available
        if load_data:
            self._load_training_data()
        
        # Current session
        self.current_session_id = None
        self.conversation_context = {
            'user_name': None,
            'mood_history': [],
            'crisis_flags': [],
            'assessment_completed': False,
            'risk_level': 'low'
        }
        
        logger.info("Enhanced Mental Healthcare Chatbot initialized successfully")
    
    def _load_training_data(self):
        """Load and train the fallback model"""
        try:
            # Check if emotion.csv exists
            if os.path.exists("emotion.csv"):
                logger.info("Loading emotion dataset...")
                data = pd.read_csv("emotion.csv")
                
                # Prepare data
                if 'Unnamed: 0' in data.columns:
                    data = data.drop(['Unnamed: 0'], axis=1)
                
                X = data['text']
                y = data['emotions']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Vectorize text
                self.vectorizer = TfidfVectorizer(max_df=0.9, max_features=5000)
                X_train_vec = self.vectorizer.fit_transform(X_train)
                
                # Encode labels
                self.encoder = LabelEncoder()
                y_train_enc = self.encoder.fit_transform(y_train)
                
                # Train model
                self.model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)
                self.model.fit(X_train_vec, y_train_enc)
                
                logger.info("Fallback emotion model trained successfully")
                
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            logger.info("Continuing without trained emotion model")
    
    def start_conversation(self, user_id: str = None) -> str:
        """Start a new conversation session"""
        # Create new session if session management is available
        if self.session_manager:
            self.current_session_id = self.session_manager.create_session(user_id)
            session = self.session_manager.get_session(self.current_session_id)
            if session:
                self.conversation_context = session.user_state
        else:
            self.current_session_id = f"session_{datetime.now().timestamp()}"
        
        logger.info(f"Started new conversation session: {self.current_session_id}")
        
        welcome_message = """
🌟 Welcome to your Mental Health Support Companion 🌟

I'm here to provide you with a safe, confidential space to discuss your mental health and wellbeing. 

**What I can help with:**
• Emotional support and active listening
• Coping strategies for stress, anxiety, and depression
• Information about mental health resources
• Crisis intervention and safety planning

**Important to know:**
• This is not a replacement for professional therapy
• If you're in immediate danger, please call emergency services
• Your privacy and safety are my top priorities

Let's start - how are you feeling today?
        """
        
        return welcome_message.strip()
    
    def process_message(self, user_input: str) -> str:
        """Process user message and generate response"""
        if not user_input.strip():
            return "I'm here to listen. Please share what's on your mind."
        
        try:
            # Update session activity
            if self.session_manager and self.current_session_id:
                session = self.session_manager.get_session(self.current_session_id)
                if session:
                    session.last_interaction = datetime.now()
            
            # Analyze input using advanced NLP if available
            analysis_results = self._analyze_user_input(user_input)
            
            # Check for crisis situations
            crisis_response = self._handle_crisis_situations(user_input, analysis_results)
            if crisis_response:
                return crisis_response
            
            # Detect intent
            intent = self._detect_intent(user_input, analysis_results)
            
            # Generate response
            bot_response = self._generate_response(user_input, intent, analysis_results)
            
            # Record interaction for learning
            if self.learning_system and self.current_session_id:
                context = {
                    'intent': intent,
                    'analysis': analysis_results,
                    'conversation_context': self.conversation_context
                }
                self.learning_system.record_interaction(
                    self.current_session_id, user_input, bot_response, context
                )
            
            # Update conversation context
            self._update_conversation_context(user_input, intent, analysis_results)
            
            # Save session state
            if self.session_manager and self.current_session_id:
                self.session_manager.update_session(
                    self.current_session_id,
                    user_state=self.conversation_context
                )
                self.session_manager.add_interaction(
                    self.current_session_id, user_input, bot_response,
                    analysis_results.get('sentiment', {}).get('polarity', 0.0), intent
                )
            
            return bot_response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm having trouble processing that right now. Can you please rephrase what you'd like to talk about?"
    
    def _analyze_user_input(self, user_input: str) -> Dict:
        """Analyze user input using available NLP tools"""
        if self.nlp_processor:
            # Use advanced NLP processor
            return self.nlp_processor.analyze_text(user_input, self.conversation_context)
        else:
            # Use fallback analysis
            return self._fallback_analysis(user_input)
    
    def _fallback_analysis(self, text: str) -> Dict:
        """Fallback text analysis using basic methods"""
        analysis = {
            'sentiment': {'label': 'NEUTRAL', 'score': 0.0, 'polarity': 0.0},
            'emotions': {'emotion': 'unknown', 'score': 0.0},
            'crisis_assessment': None,
            'intent': 'unknown',
            'key_phrases': [],
            'text_quality': {'length': len(text), 'word_count': len(text.split())}
        }
        
        # Basic sentiment analysis
        negative_words = ['sad', 'depressed', 'hopeless', 'anxious', 'worried', 'scared', 'angry', 'frustrated']
        positive_words = ['happy', 'good', 'great', 'better', 'fine', 'okay', 'wonderful']
        
        text_lower = text.lower()
        neg_count = sum(1 for word in negative_words if word in text_lower)
        pos_count = sum(1 for word in positive_words if word in text_lower)
        
        if neg_count > pos_count:
            analysis['sentiment'] = {'label': 'NEGATIVE', 'score': 0.7, 'polarity': -0.7}
        elif pos_count > neg_count:
            analysis['sentiment'] = {'label': 'POSITIVE', 'score': 0.7, 'polarity': 0.7}
        
        # Basic crisis assessment
        if self.safety_protocols:
            analysis['crisis_assessment'] = self.safety_protocols.assess_crisis_risk(text, self.conversation_context)
        else:
            crisis_keywords = ['suicide', 'kill myself', 'end my life', 'want to die', 'hurt myself']
            has_crisis = any(keyword in text_lower for keyword in crisis_keywords)
            if has_crisis:
                analysis['crisis_assessment'] = type('obj', (object,), {
                    'is_crisis': True, 'risk_level': 'high', 'intervention_needed': True
                })
        
        return analysis
    
    def _handle_crisis_situations(self, user_input: str, analysis: Dict) -> Optional[str]:
        """Handle crisis situations with appropriate safety protocols"""
        crisis_assessment = analysis.get('crisis_assessment')
        
        if not crisis_assessment:
            return None
        
        # Check if this is a crisis situation
        if hasattr(crisis_assessment, 'is_crisis') and crisis_assessment.is_crisis:
            # Log crisis event
            if self.session_manager and self.current_session_id:
                self.session_manager.log_crisis_event(
                    self.current_session_id,
                    crisis_assessment.crisis_type,
                    crisis_assessment.risk_score,
                    "crisis_intervention_initiated"
                )
            
            # Update conversation context
            self.conversation_context['crisis_flags'].append({
                'timestamp': datetime.now().isoformat(),
                'type': crisis_assessment.crisis_type,
                'risk_level': crisis_assessment.risk_level
            })
            self.conversation_context['risk_level'] = crisis_assessment.risk_level
            
            # Return appropriate crisis response
            if self.safety_protocols:
                return self.safety_protocols.get_crisis_response(crisis_assessment)
            else:
                return self._get_fallback_crisis_response(crisis_assessment.risk_level)
        
        return None
    
    def _get_fallback_crisis_response(self, risk_level: str) -> str:
        """Fallback crisis response when advanced NLP is not available"""
        if risk_level == 'critical' or risk_level == 'high':
            return self.crisis_responses['immediate_danger']
        else:
            return self.crisis_responses['high_risk']
    
    def _detect_intent(self, user_input: str, analysis: Dict) -> str:
        """Detect user intent from input"""
        if analysis.get('intent') and analysis['intent'] != 'unknown':
            return analysis['intent']
        
        # Fallback intent detection
        text_lower = user_input.lower()
        
        for intent, keywords in self.intents.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent
        
        return 'general_support'
    
    def _generate_response(self, user_input: str, intent: str, analysis: Dict) -> str:
        """Generate contextual response"""
        # Check for improved response from learning system
        if self.learning_system:
            improved_response = self.learning_system.get_improved_response(
                user_input, intent, self.conversation_context
            )
            if improved_response:
                return improved_response
        
        # Get base response
        base_response = self.responses.get(intent, self.responses['default'])
        
        # Personalize response based on context
        personalized_response = self._personalize_response(base_response, intent, analysis)
        
        return personalized_response
    
    def _personalize_response(self, base_response: str, intent: str, analysis: Dict) -> str:
        """Personalize response based on user context and history"""
        response = base_response
        
        # Add user name if available
        if self.conversation_context.get('user_name'):
            name = self.conversation_context['user_name']
            response = f"{name}, {response.lower()}"
        
        # Adjust tone based on sentiment
        sentiment = analysis.get('sentiment', {})
        if sentiment.get('label') == 'NEGATIVE' and sentiment.get('polarity', 0) < -0.5:
            # More empathetic response for very negative sentiment
            empathy_prefixes = [
                "I can hear that you're really struggling. ",
                "That sounds incredibly difficult. ",
                "I'm sorry you're going through this. "
            ]
            if not any(prefix.lower() in response.lower() for prefix in empathy_prefixes):
                response = empathy_prefixes[0] + response
        
        # Add follow-up questions based on intent
        follow_ups = {
            'depression': " What's been the most challenging part for you lately?",
            'anxiety': " Have you noticed any specific triggers that make the anxiety worse?",
            'relationship_issues': " How long have you been dealing with these relationship challenges?",
            'work_stress': " Is this a recent development at work, or has it been building up over time?"
        }
        
        if intent in follow_ups and not response.endswith('?'):
            response += follow_ups[intent]
        
        return response
    
    def _update_conversation_context(self, user_input: str, intent: str, analysis: Dict):
        """Update conversation context with new information"""
        # Update mood history
        sentiment = analysis.get('sentiment', {})
        if sentiment.get('polarity') is not None:
            mood_entry = {
                'timestamp': datetime.now().isoformat(),
                'polarity': sentiment['polarity'],
                'intent': intent,
                'text_sample': user_input[:50]
            }
            self.conversation_context['mood_history'].append(mood_entry)
            
            # Keep only last 10 mood entries
            if len(self.conversation_context['mood_history']) > 10:
                self.conversation_context['mood_history'] = self.conversation_context['mood_history'][-10:]
        
        # Extract name if mentioned
        if not self.conversation_context.get('user_name'):
            name_patterns = [
                r"my name is (\w+)",
                r"i'm (\w+)",
                r"call me (\w+)"
            ]
            for pattern in name_patterns:
                match = re.search(pattern, user_input.lower())
                if match:
                    self.conversation_context['user_name'] = match.group(1).title()
                    break
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation"""
        if not self.session_manager or not self.current_session_id:
            return {"error": "No active session"}
        
        session = self.session_manager.get_session(self.current_session_id)
        if not session:
            return {"error": "Session not found"}
        
        # Calculate mood trend
        mood_history = self.conversation_context.get('mood_history', [])
        if mood_history:
            recent_moods = [entry['polarity'] for entry in mood_history[-5:]]
            mood_trend = "improving" if np.mean(recent_moods) > -0.2 else "declining" if np.mean(recent_moods) < -0.5 else "stable"
        else:
            mood_trend = "unknown"
        
        return {
            'session_id': session.session_id,
            'duration': str(datetime.now() - session.start_time),
            'interaction_count': len(session.conversation_history),
            'mood_trend': mood_trend,
            'risk_level': self.conversation_context.get('risk_level', 'low'),
            'crisis_flags': len(self.conversation_context.get('crisis_flags', [])),
            'main_concerns': self._identify_main_concerns()
        }
    
    def _identify_main_concerns(self) -> List[str]:
        """Identify main concerns from conversation"""
        mood_history = self.conversation_context.get('mood_history', [])
        intent_counts = {}
        
        for entry in mood_history:
            intent = entry.get('intent', 'unknown')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Return top 3 concerns
        sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
        return [intent for intent, count in sorted_intents[:3] if intent != 'positive']
    
    def end_conversation(self) -> str:
        """End the current conversation"""
        summary = self.get_conversation_summary()
        
        if self.session_manager and self.current_session_id:
            self.session_manager.end_session(self.current_session_id)
        
        ending_message = """
Thank you for talking with me today. Remember:

• You've taken a positive step by reaching out
• Support is always available when you need it
• Your mental health matters

**Resources for ongoing support:**
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• SAMHSA Helpline: 1-800-662-4357

Take care of yourself. 🌟
        """
        
        self.current_session_id = None
        self.conversation_context = {
            'user_name': None,
            'mood_history': [],
            'crisis_flags': [],
            'assessment_completed': False,
            'risk_level': 'low'
        }
        
        return ending_message.strip()

def main():
    """Main function to run the chatbot"""
    print("🌟 Enhanced Mental Healthcare Chatbot 🌟")
    print("Initializing... Please wait.")
    
    # Initialize chatbot
    chatbot = EnhancedMentalHealthChatbot()
    
    # Start conversation
    welcome_message = chatbot.start_conversation()
    print(welcome_message)
    
    print("\n" + "="*50)
    print("You can type 'quit', 'exit', or 'bye' to end the conversation")
    print("="*50 + "\n")
    
    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                ending_message = chatbot.end_conversation()
                print(f"\nBot: {ending_message}")
                break
            
            # Process message and get response
            response = chatbot.process_message(user_input)
            print(f"\nBot: {response}\n")
            
            # Add small delay for better user experience
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nConversation interrupted by user.")
        ending_message = chatbot.end_conversation()
        print(f"Bot: {ending_message}")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        print("An unexpected error occurred. Please restart the chatbot.")

if __name__ == "__main__":
    main()
