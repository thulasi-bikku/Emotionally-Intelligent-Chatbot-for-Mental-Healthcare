import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import time
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    pipeline, AutoModel, AutoConfig, RobertaTokenizer, RobertaForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import nltk
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity

"""nlp.py

Module-level documentation for the repository's advanced NLP components.

Includes precise descriptions of the proposed Cultural Attention Layer and
the Metaphor-Aware Transformer used in the manuscript. The equations below
are provided to enable reproducibility and to indicate where to place the
corresponding modules inside a standard transformer backbone.

1) Cultural Attention Layer
     - Let $X=[x_1,\dots,x_n]$ be token embeddings and $c$ a cultural embedding.
     - Compute standard projections:
         $Q=XW_Q,\;K=XW_K,\;V=XW_V$.
     - Base scaled-dot-product logits:
         $A_{ij}^{(base)}=\frac{Q_iK_j^\top}{\sqrt{d_k}}$.
     - Cultural bias term (project cultural embedding into key space):
         $c'=cW_C,\;B_{ij}=\lambda\,(Q_i\cdot c')$ where $\lambda$ is learned.
     - Combined logits and attention output:
         $A_{ij}=A_{ij}^{(base)}+B_{ij},\;\alpha_{ij}=\mathrm{softmax}_j(A_{ij}),\;z_i=\sum_j\alpha_{ij}V_j$.

     - Gated fusion variant: $g_i=\sigma(W_g[Q_i;c'])$,
         $z_i=g_i\odot z_i^{(att)}+(1-g_i)\odot C_f(c)$ where $C_f$ is an MLP.

     Integration: Add as an additional attention head or modify per-head logits
     inside the transformer's multi-head attention. Ensure $c$ is supplied
     per-example (metadata, demographic embedding, or auxiliary encoder).

2) Metaphor-Aware Transformer
     - Let $T(\cdot)$ denote a standard transformer block output.
     - Compute base output: $y_i^{(base)}=T(x_i)$.
     - Project metaphor embedding $m$ and compute residual:
         $m'=mW_M,\;r_i=\tanh(W_r[x_i;m'])$.
     - Gate and fuse:
         $g_i=\sigma(W_g[y_i^{(base)};r_i]+b_g),\;y_i=g_i\odot y_i^{(base)}+(1-g_i)\odot f_m(r_i)$.

     Integration: Apply fusion after the transformer's feed-forward sub-layer or
     as an extra residual branch before layer normalization. Initialize gates
     near 1.0 to preserve pretrained behaviour initially.

Pseudocode and separate markdown docs (`docs/Cultural_Attention.md` and
`docs/Metaphor_Aware_Transformer.md`) are included in the repository. For
reproducibility, consult `REPRODUCIBILITY.md` which lists the commit hash and
package versions used for the experiments.
"""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

@dataclass
class CrisisDetectionResult:
    """Result from crisis detection analysis"""
    is_crisis: bool
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    risk_score: float
    crisis_type: str
    confidence: float
    intervention_needed: bool

class SafetyProtocols:
    """Safety protocols for crisis situations"""
    
    def __init__(self):
        self.emergency_contacts = {
            'national_suicide_prevention_lifeline': '988',
            'crisis_text_line': 'Text HOME to 741741',
            'international_emergency': '112',
            'local_emergency': {
                'india': {
                    'aasra': '91-22-27546669',
                    'sneha': '91-44-24640050',
                    'sumaitri': '91-11-23389090',
                    'jeevan': '91-44-26564444'
                }
            }
        }
        
        self.crisis_keywords = [
            # Suicide-related
            'kill myself', 'end my life', 'suicide', 'suicidal', 'want to die',
            'not worth living', 'better off dead', 'end it all', 'take my own life',
            
            # Self-harm
            'cut myself', 'hurt myself', 'self harm', 'self-harm', 'cutting',
            'burning myself', 'overdose',
            
            # Severe depression
            'hopeless', 'no point', 'cant go on', 'give up', 'worthless',
            'burden', 'everyone would be better without me',
            
            # Violence
            'hurt others', 'kill someone', 'violence', 'harm others'
        ]
        
        self.warning_phrases = [
            'feeling overwhelmed', 'very depressed', 'extremely anxious',
            'panic attack', 'cant cope', 'losing control', 'breaking down'
        ]
    
    def assess_crisis_risk(self, text: str, context: Optional[Dict] = None) -> CrisisDetectionResult:
        """Assess crisis risk in user input"""
        text_lower = text.lower()
        
        # Initialize scores
        crisis_score = 0.0
        crisis_indicators = []
        crisis_type = 'none'
        
        # Check for direct crisis keywords
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                if 'kill' in keyword or 'suicide' in keyword or 'die' in keyword:
                    crisis_score += 0.8
                    crisis_type = 'suicidal_ideation'
                elif 'harm' in keyword or 'cut' in keyword:
                    crisis_score += 0.6
                    crisis_type = 'self_harm'
                else:
                    crisis_score += 0.4
                    crisis_type = 'severe_distress'
                
                crisis_indicators.append(keyword)
        
        # Check for warning phrases
        for phrase in self.warning_phrases:
            if phrase in text_lower:
                crisis_score += 0.2
                crisis_indicators.append(phrase)
        
        # Sentiment analysis for additional context
        sentiment = TextBlob(text).sentiment
        if sentiment.polarity < -0.7:
            crisis_score += 0.3
        
        # Contextual factors
        if context:
            # Check for escalation patterns
            previous_messages = context.get('previous_messages', [])
            if self._detect_escalation_pattern(previous_messages + [text]):
                crisis_score += 0.4
                crisis_indicators.append('escalation_pattern')
            
            # Check user history
            user_history = context.get('user_history', {})
            if user_history.get('previous_crisis_events', 0) > 0:
                crisis_score += 0.2
                crisis_indicators.append('historical_risk')
        
        # Advanced pattern detection
        advanced_patterns = self._detect_advanced_crisis_patterns(text_lower)
        crisis_score += advanced_patterns['score']
        crisis_indicators.extend(advanced_patterns['indicators'])
        
        # Determine risk level and intervention need
        risk_level = self._calculate_risk_level(crisis_score)
        intervention_needed = crisis_score >= 0.6 or any(
            keyword in text_lower for keyword in ['kill myself', 'suicide', 'end my life']
        )
        
        # Determine crisis type if not already set
        if crisis_type == 'none' and crisis_score >= 0.4:
            crisis_type = 'general_distress'
        
        return CrisisDetectionResult(
            is_crisis=crisis_score >= 0.4,
            risk_level=risk_level,
            risk_score=crisis_score,
            crisis_type=crisis_type,
            confidence=min(crisis_score * 1.2, 1.0),
            intervention_needed=intervention_needed
        )
    
    def _detect_advanced_crisis_patterns(self, text: str) -> Dict:
        """Detect advanced crisis patterns using NLP techniques"""
        patterns = {
            'temporal_indicators': [
                r'\b(?:tonight|today|soon|now|immediately|cant wait)\b',
                r'\b(?:this is it|final|last|end|over|done)\b'
            ],
            'method_indicators': [
                r'\b(?:pills|rope|bridge|gun|knife|overdose)\b',
                r'\b(?:jump|hang|cut|shoot|crash|drown)\b'
            ],
            'social_indicators': [
                r'\b(?:goodbye|farewell|sorry everyone|tell.*i love)\b',
                r'\b(?:take care of|look after|without me|remember me)\b'
            ],
            'hopelessness_indicators': [
                r'\b(?:no hope|hopeless|pointless|useless|worthless)\b',
                r'\b(?:trap|stuck|no way out|no escape|no future)\b'
            ],
            'isolation_indicators': [
                r'\b(?:alone|lonely|no one|nobody|isolated)\b',
                r'\b(?:burden|bother|better off without)\b'
            ]
        }
        
        detected_indicators = []
        total_score = 0.0
        
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    detected_indicators.append(f"{category}:{','.join(matches)}")
                    # Weight temporal and method indicators higher
                    if category in ['temporal_indicators', 'method_indicators']:
                        total_score += 0.6 * len(matches)
                    else:
                        total_score += 0.3 * len(matches)
        
        return {
            'score': min(total_score, 1.0),
            'indicators': detected_indicators
        }
    
    def _detect_escalation_pattern(self, messages: List[str]) -> bool:
        """Detect escalation pattern in conversation"""
        if len(messages) < 2:
            return False
        
        escalation_keywords = [
            'getting worse', 'cant take it', 'losing hope', 'giving up',
            'no point', 'tired of', 'done trying', 'had enough'
        ]
        
        recent_messages = messages[-3:]  # Check last 3 messages
        escalation_count = 0
        
        for message in recent_messages:
            message_lower = message.lower()
            for keyword in escalation_keywords:
                if keyword in message_lower:
                    escalation_count += 1
                    break
        
        return escalation_count >= 2
    
    def _calculate_risk_level(self, score: float) -> str:
        """Calculate risk level from crisis score"""
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def get_crisis_response(self, crisis_type: str) -> str:
        """Get appropriate crisis response"""
        responses = {
            'suicidal_ideation': {
                'message': "I'm very concerned about what you've shared with me. Your life has value and meaning. Please reach out for immediate help.",
                'actions': [
                    "Call 988 (Suicide & Crisis Lifeline) immediately",
                    "Text HOME to 741741 (Crisis Text Line)",
                    "Go to your nearest emergency room",
                    "Call 911 if in immediate danger"
                ]
            },
            'self_harm': {
                'message': "I'm worried about your safety. Self-harm might provide temporary relief, but there are healthier ways to cope with these feelings.",
                'actions': [
                    "Call 988 for support",
                    "Try the butterfly project or drawing on your skin instead",
                    "Reach out to a trusted friend or family member",
                    "Contact your therapist or counselor"
                ]
            },
            'severe_distress': {
                'message': "I can sense you're in significant emotional pain right now. You don't have to face this alone.",
                'actions': [
                    "Consider calling a helpline: 988",
                    "Try grounding techniques (5-4-3-2-1)",
                    "Reach out to your support network",
                    "Contact a mental health professional"
                ]
            },
            'general_distress': {
                'message': "I hear that you're struggling right now. It's brave of you to share these feelings.",
                'actions': [
                    "Practice self-care activities",
                    "Try deep breathing exercises",
                    "Consider talking to someone you trust",
                    "Remember that these feelings will pass"
                ]
            }
        }
        
        response_data = responses.get(crisis_type, responses['general_distress'])
        message = response_data['message']
        actions = response_data['actions']
        
        return f"{message}\n\nImmediate steps you can take:\n" + "\n".join(f"• {action}" for action in actions)
    
    def log_crisis_event(self, session_id: str, crisis_result: CrisisDetectionResult):
        """Log crisis event for monitoring and follow-up"""
        try:
            crisis_data = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'crisis_type': crisis_result.crisis_type,
                'risk_level': crisis_result.risk_level,
                'risk_score': crisis_result.risk_score,
                'intervention_needed': crisis_result.intervention_needed
            }
            
            # Log to file for now (could be database in production)
            with open('crisis_events.log', 'a') as f:
                f.write(json.dumps(crisis_data) + '\n')
                
            logger.critical(f"Crisis event logged: {crisis_data}")
            
        except Exception as e:
            logger.error(f"Failed to log crisis event: {e}")
        if sentiment.polarity < -0.8:  # Very negative sentiment
            crisis_score += 0.3
        
        # Context-based adjustments
        if context:
            if context.get('previous_crisis_flags'):
                crisis_score += 0.2
            if context.get('mood_history', []):
                avg_mood = np.mean([m.get('score', 0) for m in context['mood_history'][-5:]])
                if avg_mood < -0.5:
                    crisis_score += 0.2
        
        # Determine risk level
        if crisis_score >= 0.8:
            risk_level = 'critical'
        elif crisis_score >= 0.6:
            risk_level = 'high'
        elif crisis_score >= 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        is_crisis = crisis_score >= 0.6
        intervention_needed = crisis_score >= 0.3
        
        return CrisisDetectionResult(
            is_crisis=is_crisis,
            risk_level=risk_level,
            risk_score=crisis_score,
            crisis_type=crisis_type,
            confidence=min(crisis_score, 1.0),
            intervention_needed=intervention_needed
        )
    
    def get_crisis_response(self, crisis_result: CrisisDetectionResult) -> str:
        """Generate appropriate crisis response"""
        if crisis_result.risk_level == 'critical':
            return self._get_critical_response()
        elif crisis_result.risk_level == 'high':
            return self._get_high_risk_response()
        elif crisis_result.risk_level == 'medium':
            return self._get_medium_risk_response()
        else:
            return self._get_supportive_response()
    
    def _get_critical_response(self) -> str:
        """Response for critical risk situations"""
        return """🚨 I'm very concerned about what you've shared. Your life has value and there are people who want to help you right now.

**IMMEDIATE HELP AVAILABLE:**
• Call 988 (Suicide & Crisis Lifeline) - Available 24/7
• Text HOME to 741741 (Crisis Text Line)
• Go to your nearest emergency room
• Call emergency services: 911

**In India:**
• AASRA: 91-22-27546669
• Sneha Foundation: 91-44-24640050
• Jeevan Suicide Prevention Hotline: 91-44-26564444

You don't have to go through this alone. Please reach out to one of these resources immediately. They have trained professionals who understand what you're going through."""
    
    def _get_high_risk_response(self) -> str:
        """Response for high risk situations"""
        return """I can hear that you're going through something really difficult right now. What you're feeling is valid, and there are people trained to help.

**Support Resources:**
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• SAMHSA National Helpline: 1-800-662-4357

**Immediate coping strategies:**
• Take slow, deep breaths
• Call a trusted friend or family member
• Go to a public place if you're alone
• Remove any means of self-harm from your immediate area

Would you like me to help you think through some immediate steps to stay safe?"""
    
    def _get_medium_risk_response(self) -> str:
        """Response for medium risk situations"""
        return """I can tell you're struggling right now, and I want you to know that your feelings are valid. It's important to reach out for support when things feel overwhelming.

**Helpful Resources:**
• Crisis Text Line: Text HOME to 741741
• National Suicide Prevention Lifeline: 988
• Online chat: suicidepreventionlifeline.org

**Some things that might help right now:**
• Talk to someone you trust
• Try grounding techniques (5 things you can see, 4 you can hear, etc.)
• Consider calling a counselor or therapist

How are you feeling about reaching out to someone for support?"""
    
    def _get_supportive_response(self) -> str:
        """General supportive response"""
        return """Thank you for sharing with me. It sounds like you're dealing with some challenging feelings, and that takes courage to express.

Remember that difficult feelings are temporary, even when they don't feel that way. If you ever feel like things are getting too overwhelming, please don't hesitate to reach out for support:

• Crisis Text Line: Text HOME to 741741
• National Suicide Prevention Lifeline: 988

Is there anything specific you'd like to talk about or any way I can better support you right now?"""

class AdvancedNLPProcessor:
    """Advanced NLP processor with modern transformer models for mental health analysis"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models with fallback
        self._initialize_models()
        
        # Safety protocols
        self.safety_protocols = SafetyProtocols()
        
        # Mental health specific patterns
        self.mental_health_patterns = self._load_mental_health_patterns()
        
        # Session context for personalization
        self.session_contexts = {}
        
    def _initialize_models(self):
        """Initialize transformer models with fallback options"""
        try:
            # Primary model: RoBERTa for mental health classification
            logger.info("Loading RoBERTa model for mental health analysis...")
            self.mental_health_tokenizer = AutoTokenizer.from_pretrained(
                "j-hartmann/emotion-english-distilroberta-base"
            )
            self.mental_health_model = AutoModelForSequenceClassification.from_pretrained(
                "j-hartmann/emotion-english-distilroberta-base"
            ).to(self.device)
            
            # Secondary model: BERT for suicide risk assessment
            logger.info("Loading BERT model for suicide risk assessment...")
            self.suicide_risk_tokenizer = AutoTokenizer.from_pretrained(
                "martin-ha/toxic-comment-model"
            )
            self.suicide_risk_model = AutoModelForSequenceClassification.from_pretrained(
                "martin-ha/toxic-comment-model"
            ).to(self.device)
            
            # Sentence transformer for semantic similarity
            logger.info("Loading sentence transformer...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create mental health classification pipeline
            self.emotion_classifier = pipeline(
                "text-classification",
                model=self.mental_health_model,
                tokenizer=self.mental_health_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading transformer models: {e}")
            logger.info("Falling back to basic models")
            self._initialize_fallback_models()
            
    def _initialize_fallback_models(self):
        """Initialize fallback models when transformers fail"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.models_loaded = False
            logger.info("Fallback models initialized")
        except Exception as e:
            logger.error(f"Error loading fallback models: {e}")
            self.models_loaded = False
    
    def _load_mental_health_patterns(self) -> Dict:
        """Load mental health specific patterns and responses"""
        return {
            'depression_indicators': [
                r'\b(?:depressed|depression|sad|hopeless|worthless|empty|numb)\b',
                r'\b(?:no energy|tired|exhausted|fatigue|sleeping too much|insomnia)\b',
                r'\b(?:guilt|shame|failure|disappointed|let down|burden)\b',
                r'\b(?:nothing matters|no point|give up|cant go on)\b'
            ],
            'anxiety_indicators': [
                r'\b(?:anxious|anxiety|worried|nervous|panic|fear|scared)\b',
                r'\b(?:racing thoughts|cant concentrate|restless|on edge)\b',
                r'\b(?:heart racing|sweating|shaking|trembling|nausea)\b',
                r'\b(?:avoid|avoiding|phobia|terrified|overwhelmed)\b'
            ],
            'trauma_indicators': [
                r'\b(?:trauma|traumatic|flashback|nightmare|triggered)\b',
                r'\b(?:abuse|abused|assault|violence|accident)\b',
                r'\b(?:hypervigilant|startled|jumpy|on guard|unsafe)\b'
            ],
            'positive_coping': [
                r'\b(?:therapy|counseling|medication|support group|help)\b',
                r'\b(?:exercise|meditation|breathing|mindfulness|yoga)\b',
                r'\b(?:family|friends|support|better|improving|progress)\b'
            ]
        }
    
    def analyze_text(self, text: str, context: Optional[Dict] = None) -> Dict:
        """Comprehensive text analysis using multiple approaches"""
        analysis_start = time.time()
        
        # Initialize analysis result
        analysis = {
            'input_text': text,
            'timestamp': datetime.now().isoformat(),
            'emotions': {},
            'sentiment': {},
            'mental_health_indicators': {},
            'crisis_assessment': {},
            'intent_classification': {},
            'response_recommendations': [],
            'confidence_scores': {},
            'processing_time': 0.0
        }
        
        try:
            # 1. Crisis risk assessment (highest priority)
            crisis_result = self.safety_protocols.assess_crisis_risk(text, context)
            analysis['crisis_assessment'] = {
                'is_crisis': crisis_result.is_crisis,
                'risk_level': crisis_result.risk_level,
                'risk_score': crisis_result.risk_score,
                'crisis_type': crisis_result.crisis_type,
                'confidence': crisis_result.confidence,
                'intervention_needed': crisis_result.intervention_needed
            }
            
            # 2. Advanced emotion analysis
            if self.models_loaded:
                emotions = self._analyze_emotions_transformer(text)
                analysis['emotions'] = emotions
                analysis['confidence_scores']['emotions'] = 0.85
            else:
                emotions = self._analyze_emotions_fallback(text)
                analysis['emotions'] = emotions
                analysis['confidence_scores']['emotions'] = 0.65
            
            # 3. Sentiment analysis
            sentiment = self._analyze_sentiment(text)
            analysis['sentiment'] = sentiment
            
            # 4. Mental health pattern matching
            mh_indicators = self._detect_mental_health_patterns(text)
            analysis['mental_health_indicators'] = mh_indicators
            
            # 5. Intent classification
            intent = self._classify_intent(text, context)
            analysis['intent_classification'] = intent
            
            # 6. Generate response recommendations
            recommendations = self._generate_response_recommendations(analysis, context)
            analysis['response_recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            analysis['error'] = str(e)
        
        analysis['processing_time'] = time.time() - analysis_start
        return analysis
    
    def _analyze_emotions_transformer(self, text: str) -> Dict:
        """Analyze emotions using transformer model"""
        try:
            # Use the emotion classifier
            emotions = self.emotion_classifier(text, top_k=None)
            
            # Convert to our format
            emotion_scores = {}
            for emotion in emotions:
                emotion_scores[emotion['label'].lower()] = emotion['score']
            
            # Determine primary emotion
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            return {
                'primary_emotion': primary_emotion[0],
                'confidence': primary_emotion[1],
                'all_emotions': emotion_scores,
                'model_used': 'transformer'
            }
        except Exception as e:
            logger.error(f"Error in transformer emotion analysis: {e}")
            return self._analyze_emotions_fallback(text)
    
    def _analyze_emotions_fallback(self, text: str) -> Dict:
        """Fallback emotion analysis using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Map polarity to emotions
            if polarity > 0.5:
                primary_emotion = 'joy'
            elif polarity > 0.1:
                primary_emotion = 'optimism'
            elif polarity > -0.1:
                primary_emotion = 'neutral'
            elif polarity > -0.5:
                primary_emotion = 'sadness'
            else:
                primary_emotion = 'anger'
            
            return {
                'primary_emotion': primary_emotion,
                'confidence': abs(polarity),
                'polarity': polarity,
                'subjectivity': subjectivity,
                'model_used': 'textblob'
            }
        except Exception as e:
            logger.error(f"Error in fallback emotion analysis: {e}")
            return {'primary_emotion': 'neutral', 'confidence': 0.0, 'error': str(e)}
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Comprehensive sentiment analysis"""
        try:
            # TextBlob sentiment
            blob = TextBlob(text)
            
            # Try VADER if available
            if hasattr(self, 'sentiment_analyzer'):
                vader_scores = self.sentiment_analyzer.polarity_scores(text)
                return {
                    'compound': vader_scores['compound'],
                    'positive': vader_scores['pos'],
                    'negative': vader_scores['neg'],
                    'neutral': vader_scores['neu'],
                    'textblob_polarity': blob.sentiment.polarity,
                    'textblob_subjectivity': blob.sentiment.subjectivity
                }
            else:
                return {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity,
                    'compound': blob.sentiment.polarity  # Use polarity as compound score
                }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'compound': 0.0, 'error': str(e)}
    
    def _detect_mental_health_patterns(self, text: str) -> Dict:
        """Detect mental health indicators using pattern matching"""
        indicators = {}
        text_lower = text.lower()
        
        for category, patterns in self.mental_health_patterns.items():
            matches = []
            total_score = 0.0
            
            for pattern in patterns:
                matches_found = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches_found:
                    matches.extend(matches_found)
                    total_score += len(matches_found) * 0.2
            
            if matches:
                indicators[category] = {
                    'detected': True,
                    'matches': list(set(matches)),
                    'score': min(total_score, 1.0),
                    'severity': self._determine_severity(total_score)
                }
            else:
                indicators[category] = {'detected': False, 'score': 0.0}
        
        return indicators
    
    def _determine_severity(self, score: float) -> str:
        """Determine severity level from score"""
        if score >= 0.8:
            return 'high'
        elif score >= 0.5:
            return 'moderate'
        elif score >= 0.2:
            return 'mild'
        else:
            return 'low'
    
    def _classify_intent(self, text: str, context: Optional[Dict] = None) -> Dict:
        """Classify user intent"""
        intents = {
            'seeking_help': [r'\b(?:help|support|need|assistance|guidance)\b'],
            'expressing_feelings': [r'\b(?:feel|feeling|emotions|mood|state)\b'],
            'asking_questions': [r'\b(?:\?|what|how|why|when|where|can you|do you)\b'],
            'crisis_expression': [r'\b(?:emergency|crisis|urgent|immediate|now)\b'],
            'gratitude': [r'\b(?:thank|thanks|grateful|appreciate)\b'],
            'goodbye': [r'\b(?:bye|goodbye|farewell|see you|talk later)\b']
        }
        
        text_lower = text.lower()
        detected_intents = {}
        
        for intent, patterns in intents.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 0.3
            
            if score > 0:
                detected_intents[intent] = min(score, 1.0)
        
        if detected_intents:
            primary_intent = max(detected_intents.items(), key=lambda x: x[1])
            return {
                'primary_intent': primary_intent[0],
                'confidence': primary_intent[1],
                'all_intents': detected_intents
            }
        else:
            return {'primary_intent': 'general_conversation', 'confidence': 0.5}
    
    def _generate_response_recommendations(self, analysis: Dict, context: Optional[Dict] = None) -> List[Dict]:
        """Generate personalized response recommendations"""
        recommendations = []
        
        # Crisis intervention takes priority
        if analysis['crisis_assessment']['is_crisis']:
            recommendations.append({
                'type': 'crisis_intervention',
                'priority': 'critical',
                'message': self.safety_protocols.get_crisis_response(
                    analysis['crisis_assessment']['crisis_type']
                ),
                'action_required': True
            })
        
        # Emotional support based on detected emotions
        primary_emotion = analysis['emotions'].get('primary_emotion', 'neutral')
        if primary_emotion in ['sadness', 'anger', 'fear']:
            recommendations.append({
                'type': 'emotional_support',
                'priority': 'high',
                'emotion_addressed': primary_emotion,
                'message': self._get_emotional_support_message(primary_emotion),
                'action_required': False
            })
        
        # Mental health resource recommendations
        mh_indicators = analysis['mental_health_indicators']
        for indicator, data in mh_indicators.items():
            if data['detected'] and data['score'] > 0.3:
                recommendations.append({
                    'type': 'resource_recommendation',
                    'priority': 'medium',
                    'indicator': indicator,
                    'resources': self._get_resources_for_indicator(indicator),
                    'action_required': False
                })
        
        return recommendations
    
    def _get_emotional_support_message(self, emotion: str) -> str:
        """Get appropriate emotional support message"""
        messages = {
            'sadness': "I can hear that you're going through a difficult time. Your feelings are valid, and it's okay to feel sad. Would you like to talk about what's been weighing on your heart?",
            'anger': "It sounds like you're feeling frustrated or angry. These are natural emotions, and it's important to acknowledge them. What's been causing these feelings?",
            'fear': "I sense that you might be feeling anxious or fearful. It's completely understandable to feel this way. Would you like to explore some coping strategies together?",
            'anxiety': "I can tell you're feeling anxious. Anxiety can be overwhelming, but there are ways to manage these feelings. Let's work through this together."
        }
        return messages.get(emotion, "I'm here to listen and support you through whatever you're experiencing.")
    
    def _get_resources_for_indicator(self, indicator: str) -> List[Dict]:
        """Get relevant resources for mental health indicators"""
        resources = {
            'depression_indicators': [
                {'type': 'helpline', 'name': 'National Suicide Prevention Lifeline', 'contact': '988'},
                {'type': 'technique', 'name': 'Behavioral Activation', 'description': 'Gradual increase in meaningful activities'},
                {'type': 'app', 'name': 'Mood tracking apps', 'description': 'Track daily mood patterns'}
            ],
            'anxiety_indicators': [
                {'type': 'technique', 'name': 'Deep Breathing', 'description': '4-7-8 breathing technique'},
                {'type': 'technique', 'name': 'Grounding Exercise', 'description': '5-4-3-2-1 sensory technique'},
                {'type': 'resource', 'name': 'Anxiety and Depression Association', 'website': 'adaa.org'}
            ],
            'trauma_indicators': [
                {'type': 'helpline', 'name': 'RAINN National Sexual Assault Hotline', 'contact': '1-800-656-HOPE'},
                {'type': 'therapy', 'name': 'Trauma-Focused CBT', 'description': 'Evidence-based treatment for trauma'},
                {'type': 'technique', 'name': 'EMDR', 'description': 'Eye Movement Desensitization and Reprocessing'}
            ]
        }
        return resources.get(indicator, [])

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            if hasattr(self, 'sentence_model'):
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            else:
                # Fallback to simple token overlap
                tokens1 = set(text1.lower().split())
                tokens2 = set(text2.lower().split())
                intersection = tokens1.intersection(tokens2)
                union = tokens1.union(tokens2)
                return len(intersection) / len(union) if union else 0.0
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0

# Usage example
if __name__ == "__main__":
    nlp_processor = AdvancedNLPProcessor()
    
    # Test crisis detection
    test_texts = [
        "I feel really sad today",
        "I can't take it anymore, I want to end my life",
        "I'm having a good day",
        "I hurt myself last night"
    ]
    
    for text in test_texts:
        result = nlp_processor.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Emotion: {result['emotions']}")
        print(f"Crisis Risk: {result['crisis_assessment'].risk_level}")
        print(f"Intent: {result['intent']}")
        
        if result['crisis_assessment'].intervention_needed:
            response = nlp_processor.safety_protocols.get_crisis_response(
                result['crisis_assessment']
            )
            print(f"Crisis Response: {response[:100]}...")
