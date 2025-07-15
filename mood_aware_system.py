"""
Mood-Aware Response System
Detects user emotions and adjusts AI response style accordingly
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class MoodType(Enum):
    """Different mood types"""
    HAPPY = "happy"
    SAD = "sad"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    TIRED = "tired"
    STRESSED = "stressed"
    RELAXED = "relaxed"
    CONFUSED = "confused"
    CONFIDENT = "confident"
    NEUTRAL = "neutral"

class ResponseStyle(Enum):
    """Different response styles"""
    SUPPORTIVE = "supportive"
    ENERGETIC = "energetic"
    CALMING = "calming"
    ENCOURAGING = "encouraging"
    GENTLE = "gentle"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    DETAILED = "detailed"
    CONCISE = "concise"

@dataclass
class MoodIndicator:
    """Mood detection indicator"""
    patterns: List[str]
    weight: float
    context_required: bool = False

@dataclass
class MoodAnalysis:
    """Result of mood analysis"""
    primary_mood: MoodType
    confidence: float
    secondary_moods: List[Tuple[MoodType, float]]
    indicators: List[str]
    suggested_style: ResponseStyle
    tone_adjustments: Dict[str, float]

class MoodAwareSystem:
    """System for detecting user mood and adjusting responses"""
    
    def __init__(self):
        self.mood_patterns = self._initialize_mood_patterns()
        self.response_styles = self._initialize_response_styles()
        
    def _initialize_mood_patterns(self) -> Dict[MoodType, List[MoodIndicator]]:
        """Initialize mood detection patterns"""
        return {
            MoodType.HAPPY: [
                MoodIndicator([r"great", r"awesome", r"wonderful", r"amazing", r"fantastic", r"love", r"excited"], 0.8),
                MoodIndicator([r"😊", r"😄", r"😁", r"🎉", r"❤️", r"👍"], 0.9),
                MoodIndicator([r"thank you so much", r"this is perfect", r"you're the best"], 0.7),
                MoodIndicator([r"!+", r"!!+"], 0.3)  # Multiple exclamation marks
            ],
            
            MoodType.SAD: [
                MoodIndicator([r"sad", r"depressed", r"down", r"upset", r"disappointed", r"heartbroken"], 0.8),
                MoodIndicator([r"😢", r"😭", r"😔", r"💔", r"😞"], 0.9),
                MoodIndicator([r"i feel like", r"i'm so", r"everything is"], 0.4, context_required=True),
                MoodIndicator([r"nothing works", r"give up", r"can't do this"], 0.6)
            ],
            
            MoodType.FRUSTRATED: [
                MoodIndicator([r"frustrated", r"annoyed", r"irritated", r"angry", r"mad", r"pissed"], 0.8),
                MoodIndicator([r"this is stupid", r"doesn't work", r"waste of time", r"ridiculous"], 0.7),
                MoodIndicator([r"😤", r"😡", r"🤬", r"😠"], 0.9),
                MoodIndicator([r"why won't", r"still not", r"keeps failing"], 0.5)
            ],
            
            MoodType.EXCITED: [
                MoodIndicator([r"excited", r"thrilled", r"pumped", r"stoked", r"can't wait"], 0.8),
                MoodIndicator([r"🎉", r"🚀", r"⚡", r"🔥", r"💪"], 0.7),
                MoodIndicator([r"let's do this", r"bring it on", r"ready to go"], 0.6),
                MoodIndicator([r"!!!+"], 0.4)  # Multiple exclamation marks
            ],
            
            MoodType.TIRED: [
                MoodIndicator([r"tired", r"exhausted", r"drained", r"worn out", r"beat"], 0.8),
                MoodIndicator([r"😴", r"😪", r"🥱"], 0.9),
                MoodIndicator([r"long day", r"so much work", r"need rest"], 0.5),
                MoodIndicator([r"can't think", r"brain fog", r"no energy"], 0.6)
            ],
            
            MoodType.STRESSED: [
                MoodIndicator([r"stressed", r"overwhelmed", r"anxious", r"pressure", r"deadline"], 0.8),
                MoodIndicator([r"😰", r"😫", r"😓", r"💀"], 0.7),
                MoodIndicator([r"too much", r"can't handle", r"falling behind"], 0.6),
                MoodIndicator([r"urgent", r"asap", r"quickly", r"emergency"], 0.4)
            ],
            
            MoodType.CONFUSED: [
                MoodIndicator([r"confused", r"lost", r"don't understand", r"what does", r"how do"], 0.7),
                MoodIndicator([r"🤔", r"😕", r"🤷", r"❓"], 0.8),
                MoodIndicator([r"\?+"], 0.3),  # Multiple question marks
                MoodIndicator([r"not sure", r"unclear", r"doesn't make sense"], 0.5)
            ],
            
            MoodType.CONFIDENT: [
                MoodIndicator([r"confident", r"sure", r"definitely", r"absolutely", r"certain"], 0.7),
                MoodIndicator([r"got this", r"piece of cake", r"easy", r"no problem"], 0.6),
                MoodIndicator([r"💪", r"👌", r"✅", r"🎯"], 0.5),
                MoodIndicator([r"ready", r"prepared", r"know exactly"], 0.4)
            ]
        }
    
    def _initialize_response_styles(self) -> Dict[MoodType, Dict[str, any]]:
        """Initialize response style mappings"""
        return {
            MoodType.HAPPY: {
                "style": ResponseStyle.ENERGETIC,
                "tone": "enthusiastic",
                "formality": 0.3,
                "verbosity": 0.7,
                "encouragement": 0.8,
                "prefix_options": [
                    "That's fantastic!",
                    "I'm so glad to hear that!",
                    "Wonderful!",
                    "Great to see you're doing well!"
                ]
            },
            
            MoodType.SAD: {
                "style": ResponseStyle.SUPPORTIVE,
                "tone": "gentle",
                "formality": 0.4,
                "verbosity": 0.6,
                "encouragement": 0.9,
                "prefix_options": [
                    "I'm sorry you're feeling this way.",
                    "That sounds really difficult.",
                    "I understand this is tough for you.",
                    "I'm here to help you through this."
                ]
            },
            
            MoodType.FRUSTRATED: {
                "style": ResponseStyle.CALMING,
                "tone": "understanding",
                "formality": 0.5,
                "verbosity": 0.5,
                "encouragement": 0.8,
                "prefix_options": [
                    "I can understand your frustration.",
                    "Let's work through this together.",
                    "That does sound frustrating.",
                    "I'm here to help make this easier."
                ]
            },
            
            MoodType.EXCITED: {
                "style": ResponseStyle.ENERGETIC,
                "tone": "enthusiastic",
                "formality": 0.2,
                "verbosity": 0.8,
                "encouragement": 0.7,
                "prefix_options": [
                    "I love your enthusiasm!",
                    "That's amazing!",
                    "Your excitement is contagious!",
                    "Let's channel that energy!"
                ]
            },
            
            MoodType.TIRED: {
                "style": ResponseStyle.GENTLE,
                "tone": "calm",
                "formality": 0.6,
                "verbosity": 0.4,
                "encouragement": 0.7,
                "prefix_options": [
                    "You sound tired - let's keep this simple.",
                    "I'll make this as easy as possible.",
                    "Let's take this step by step.",
                    "I'll help you get through this quickly."
                ]
            },
            
            MoodType.STRESSED: {
                "style": ResponseStyle.CALMING,
                "tone": "reassuring",
                "formality": 0.5,
                "verbosity": 0.5,
                "encouragement": 0.9,
                "prefix_options": [
                    "Let's break this down into manageable steps.",
                    "I'm here to help reduce your stress.",
                    "We can tackle this together.",
                    "Let's prioritize what's most important."
                ]
            },
            
            MoodType.CONFUSED: {
                "style": ResponseStyle.DETAILED,
                "tone": "patient",
                "formality": 0.6,
                "verbosity": 0.8,
                "encouragement": 0.7,
                "prefix_options": [
                    "Let me explain this clearly.",
                    "I'll walk you through this step by step.",
                    "Great question - let me clarify.",
                    "I understand the confusion."
                ]
            },
            
            MoodType.CONFIDENT: {
                "style": ResponseStyle.PROFESSIONAL,
                "tone": "direct",
                "formality": 0.7,
                "verbosity": 0.6,
                "encouragement": 0.5,
                "prefix_options": [
                    "I can see you're ready to tackle this.",
                    "Perfect, let's move forward.",
                    "Great approach.",
                    "You're on the right track."
                ]
            }
        }
    
    def analyze_mood(self, text: str, context: Dict[str, any] = None) -> MoodAnalysis:
        """Analyze user mood from text input"""
        try:
            text_lower = text.lower()
            mood_scores = {}
            detected_indicators = []
            
            # Analyze each mood type
            for mood_type, indicators in self.mood_patterns.items():
                score = 0
                mood_indicators = []
                
                for indicator in indicators:
                    for pattern in indicator.patterns:
                        matches = len(re.findall(pattern, text_lower))
                        if matches > 0:
                            # Apply weight and diminishing returns for multiple matches
                            pattern_score = indicator.weight * min(matches, 3) / 3
                            score += pattern_score
                            mood_indicators.append(pattern)
                
                if score > 0:
                    mood_scores[mood_type] = score
                    detected_indicators.extend(mood_indicators)
            
            # Apply contextual adjustments
            if context:
                mood_scores = self._apply_contextual_adjustments(mood_scores, context)
            
            # Determine primary mood
            if not mood_scores:
                primary_mood = MoodType.NEUTRAL
                confidence = 0.5
                secondary_moods = []
            else:
                # Sort by score
                sorted_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
                primary_mood = sorted_moods[0][0]
                confidence = min(sorted_moods[0][1] / 2, 1.0)  # Normalize confidence
                secondary_moods = [(mood, score) for mood, score in sorted_moods[1:3]]
            
            # Determine response style
            style_config = self.response_styles.get(primary_mood, self.response_styles[MoodType.NEUTRAL])
            suggested_style = style_config["style"]
            
            # Generate tone adjustments
            tone_adjustments = {
                "formality": style_config.get("formality", 0.5),
                "verbosity": style_config.get("verbosity", 0.5),
                "encouragement": style_config.get("encouragement", 0.5),
                "tone": style_config.get("tone", "neutral")
            }
            
            return MoodAnalysis(
                primary_mood=primary_mood,
                confidence=confidence,
                secondary_moods=secondary_moods,
                indicators=detected_indicators,
                suggested_style=suggested_style,
                tone_adjustments=tone_adjustments
            )
            
        except Exception as e:
            logger.error(f"Error analyzing mood: {e}")
            return MoodAnalysis(
                primary_mood=MoodType.NEUTRAL,
                confidence=0.5,
                secondary_moods=[],
                indicators=[],
                suggested_style=ResponseStyle.PROFESSIONAL,
                tone_adjustments={"formality": 0.5, "verbosity": 0.5, "encouragement": 0.5, "tone": "neutral"}
            )
    
    def _apply_contextual_adjustments(self, mood_scores: Dict[MoodType, float], 
                                   context: Dict[str, any]) -> Dict[MoodType, float]:
        """Apply contextual adjustments to mood scores"""
        adjusted_scores = mood_scores.copy()
        
        # Time-based adjustments
        if "time" in context:
            hour = context["time"].hour
            if hour < 6 or hour > 22:  # Very early or very late
                adjusted_scores[MoodType.TIRED] = adjusted_scores.get(MoodType.TIRED, 0) + 0.2
            elif 9 <= hour <= 17:  # Work hours
                adjusted_scores[MoodType.STRESSED] = adjusted_scores.get(MoodType.STRESSED, 0) + 0.1
        
        # Previous interaction context
        if "previous_mood" in context:
            previous = context["previous_mood"]
            if previous == MoodType.FRUSTRATED:
                # User might still be frustrated
                adjusted_scores[MoodType.FRUSTRATED] = adjusted_scores.get(MoodType.FRUSTRATED, 0) + 0.1
        
        # Query complexity context
        if "query_complexity" in context:
            complexity = context["query_complexity"]
            if complexity > 0.8:  # High complexity
                adjusted_scores[MoodType.CONFUSED] = adjusted_scores.get(MoodType.CONFUSED, 0) + 0.15
                adjusted_scores[MoodType.STRESSED] = adjusted_scores.get(MoodType.STRESSED, 0) + 0.1
        
        return adjusted_scores
    
    def get_response_prefix(self, mood_analysis: MoodAnalysis) -> str:
        """Get appropriate response prefix based on mood"""
        try:
            style_config = self.response_styles.get(mood_analysis.primary_mood, 
                                                  self.response_styles[MoodType.NEUTRAL])
            
            prefix_options = style_config.get("prefix_options", ["Let me help you with that."])
            
            # Simple selection based on confidence
            if mood_analysis.confidence > 0.7:
                return prefix_options[0]  # Most confident response
            elif mood_analysis.confidence > 0.5:
                return prefix_options[1] if len(prefix_options) > 1 else prefix_options[0]
            else:
                return prefix_options[-1] if len(prefix_options) > 2 else prefix_options[0]
                
        except Exception as e:
            logger.error(f"Error getting response prefix: {e}")
            return "I'm here to help."
    
    def adjust_response_style(self, response: str, mood_analysis: MoodAnalysis) -> str:
        """Adjust response style based on mood analysis"""
        try:
            if mood_analysis.confidence < 0.3:
                return response  # Don't adjust if confidence is too low
            
            adjusted_response = response
            tone_adjustments = mood_analysis.tone_adjustments
            
            # Apply verbosity adjustments
            if tone_adjustments.get("verbosity", 0.5) < 0.4:
                # Make more concise
                adjusted_response = self._make_concise(adjusted_response)
            elif tone_adjustments.get("verbosity", 0.5) > 0.7:
                # Make more detailed
                adjusted_response = self._make_detailed(adjusted_response)
            
            # Apply encouragement
            if tone_adjustments.get("encouragement", 0.5) > 0.7:
                adjusted_response = self._add_encouragement(adjusted_response, mood_analysis.primary_mood)
            
            # Apply formality adjustments
            if tone_adjustments.get("formality", 0.5) < 0.3:
                adjusted_response = self._make_casual(adjusted_response)
            elif tone_adjustments.get("formality", 0.5) > 0.7:
                adjusted_response = self._make_formal(adjusted_response)
            
            return adjusted_response
            
        except Exception as e:
            logger.error(f"Error adjusting response style: {e}")
            return response
    
    def _make_concise(self, text: str) -> str:
        """Make text more concise"""
        # Remove redundant phrases and shorten sentences
        text = re.sub(r'\b(actually|basically|essentially|really|quite|very)\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _make_detailed(self, text: str) -> str:
        """Make text more detailed"""
        # Add explanatory phrases
        if "Here's" in text:
            text = text.replace("Here's", "Let me explain - here's")
        if "You can" in text:
            text = text.replace("You can", "What you can do is")
        return text
    
    def _add_encouragement(self, text: str, mood: MoodType) -> str:
        """Add encouragement based on mood"""
        encouragements = {
            MoodType.SAD: " Remember, you're doing great and I'm here to support you.",
            MoodType.FRUSTRATED: " Don't worry - we'll figure this out together.",
            MoodType.TIRED: " You're doing well despite being tired.",
            MoodType.STRESSED: " Take a deep breath - we'll work through this step by step.",
            MoodType.CONFUSED: " No worries about the confusion - that's completely normal."
        }
        
        encouragement = encouragements.get(mood, " You're doing great!")
        return text + encouragement
    
    def _make_casual(self, text: str) -> str:
        """Make text more casual"""
        # Replace formal phrases with casual ones
        replacements = {
            "I would recommend": "I'd suggest",
            "Please consider": "How about",
            "You may want to": "You might want to",
            "It is important": "It's important",
            "You should": "You could"
        }
        
        for formal, casual in replacements.items():
            text = text.replace(formal, casual)
        
        return text
    
    def _make_formal(self, text: str) -> str:
        """Make text more formal"""
        # Replace casual phrases with formal ones
        replacements = {
            "I'd": "I would",
            "you'll": "you will",
            "can't": "cannot",
            "won't": "will not",
            "How about": "Please consider"
        }
        
        for casual, formal in replacements.items():
            text = text.replace(casual, formal)
        
        return text

# Add neutral mood to response styles
def _add_neutral_mood():
    """Add neutral mood to response styles"""
    system = MoodAwareSystem()
    system.response_styles[MoodType.NEUTRAL] = {
        "style": ResponseStyle.PROFESSIONAL,
        "tone": "neutral",
        "formality": 0.5,
        "verbosity": 0.5,
        "encouragement": 0.5,
        "prefix_options": [
            "I'm here to help.",
            "Let me assist you with that.",
            "I'll help you with this.",
            "Let's work on this together."
        ]
    }
    return system

# Global instance
_mood_aware_system = None

def get_mood_aware_system() -> MoodAwareSystem:
    """Get or create global mood-aware system instance"""
    global _mood_aware_system
    if _mood_aware_system is None:
        _mood_aware_system = _add_neutral_mood()
    return _mood_aware_system