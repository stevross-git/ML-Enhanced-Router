"""
User Profile Builder for Personal AI Assistant
Creates comprehensive user profiles through conversational questioning
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)

class ProfileCategory(Enum):
    """Categories for user profile information"""
    PERSONAL_INFO = "personal_info"
    COMMUNICATION = "communication"
    WORK_LIFE = "work_life"
    INTERESTS = "interests"
    PREFERENCES = "preferences"
    LIFESTYLE = "lifestyle"
    GOALS = "goals"
    CONTEXT = "context"
    RELATIONSHIPS = "relationships"
    TECHNICAL = "technical"

class QuestionType(Enum):
    """Types of questions for profile building"""
    OPEN_ENDED = "open_ended"
    MULTIPLE_CHOICE = "multiple_choice"
    RATING = "rating"
    YES_NO = "yes_no"
    FOLLOW_UP = "follow_up"

@dataclass
class ProfileQuestion:
    """Structure for profile building questions"""
    id: str
    category: ProfileCategory
    question_type: QuestionType
    question: str
    follow_up_questions: List[str] = field(default_factory=list)
    context: Optional[str] = None
    importance: int = 1  # 1-5 scale
    choices: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)  # Keywords that trigger this question
    
@dataclass
class UserResponse:
    """User response to profile questions"""
    question_id: str
    response: str
    timestamp: datetime
    follow_up_responses: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
@dataclass
class UserProfile:
    """Complete user profile"""
    user_id: str
    profile_completion: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    personal_info: Dict[str, Any] = field(default_factory=dict)
    communication_style: Dict[str, Any] = field(default_factory=dict)
    work_context: Dict[str, Any] = field(default_factory=dict)
    interests: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    lifestyle: Dict[str, Any] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    relationships: Dict[str, Any] = field(default_factory=dict)
    technical_preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ProfileQuestionBank:
    """Database of profile building questions"""
    
    def __init__(self):
        self.questions = self._initialize_questions()
        
    def _initialize_questions(self) -> List[ProfileQuestion]:
        """Initialize the question bank with conversational questions"""
        return [
            # Personal Info - Core Identity
            ProfileQuestion(
                id="name_preference",
                category=ProfileCategory.PERSONAL_INFO,
                question_type=QuestionType.OPEN_ENDED,
                question="What would you like me to call you? (First name, nickname, or whatever feels right)",
                importance=5,
                context="This helps me address you personally"
            ),
            
            ProfileQuestion(
                id="personal_background",
                category=ProfileCategory.PERSONAL_INFO,
                question_type=QuestionType.OPEN_ENDED,
                question="Tell me a bit about yourself - what defines you as a person?",
                follow_up_questions=[
                    "What's most important to you in life?",
                    "What are you most proud of?"
                ],
                importance=4
            ),
            
            ProfileQuestion(
                id="location_context",
                category=ProfileCategory.PERSONAL_INFO,
                question_type=QuestionType.OPEN_ENDED,
                question="Where are you based? (This helps me understand your timezone and local context)",
                importance=3
            ),
            
            # Communication Style
            ProfileQuestion(
                id="communication_style",
                category=ProfileCategory.COMMUNICATION,
                question_type=QuestionType.MULTIPLE_CHOICE,
                question="How do you prefer to communicate? Pick what feels most natural:",
                choices=[
                    "Direct and to the point",
                    "Detailed explanations with examples",
                    "Casual and conversational",
                    "Formal and structured",
                    "Mix it up depending on the topic"
                ],
                importance=5
            ),
            
            ProfileQuestion(
                id="explanation_preference",
                category=ProfileCategory.COMMUNICATION,
                question_type=QuestionType.MULTIPLE_CHOICE,
                question="When you need help understanding something, what works best for you?",
                choices=[
                    "Step-by-step instructions",
                    "Real-world examples and analogies",
                    "Visual explanations when possible",
                    "Quick summaries first, details if needed",
                    "Let me figure it out with gentle guidance"
                ],
                importance=4
            ),
            
            ProfileQuestion(
                id="feedback_style",
                category=ProfileCategory.COMMUNICATION,
                question_type=QuestionType.OPEN_ENDED,
                question="How do you like to receive feedback or corrections? What approach motivates you?",
                importance=3
            ),
            
            # Work and Professional Life
            ProfileQuestion(
                id="work_context",
                category=ProfileCategory.WORK_LIFE,
                question_type=QuestionType.OPEN_ENDED,
                question="What do you do for work, or what's your main focus right now?",
                follow_up_questions=[
                    "What are your biggest challenges at work?",
                    "What tools or skills are most important in your role?"
                ],
                importance=4
            ),
            
            ProfileQuestion(
                id="work_schedule",
                category=ProfileCategory.WORK_LIFE,
                question_type=QuestionType.OPEN_ENDED,
                question="What's your typical schedule like? Are you more productive at certain times?",
                importance=3
            ),
            
            ProfileQuestion(
                id="professional_goals",
                category=ProfileCategory.WORK_LIFE,
                question_type=QuestionType.OPEN_ENDED,
                question="What are you working towards professionally? Any specific goals or dreams?",
                importance=3
            ),
            
            # Interests and Hobbies
            ProfileQuestion(
                id="interests_hobbies",
                category=ProfileCategory.INTERESTS,
                question_type=QuestionType.OPEN_ENDED,
                question="What do you enjoy doing in your free time? What are you passionate about?",
                follow_up_questions=[
                    "Is there anything you've always wanted to try?",
                    "What did you love doing as a kid that you still enjoy?"
                ],
                importance=4
            ),
            
            ProfileQuestion(
                id="learning_interests",
                category=ProfileCategory.INTERESTS,
                question_type=QuestionType.OPEN_ENDED,
                question="What topics fascinate you? What would you love to learn more about?",
                importance=3
            ),
            
            ProfileQuestion(
                id="creative_interests",
                category=ProfileCategory.INTERESTS,
                question_type=QuestionType.YES_NO,
                question="Do you enjoy creative activities like writing, art, music, or crafts?",
                follow_up_questions=[
                    "What kind of creative projects appeal to you?",
                    "Have you ever wanted to try a creative hobby?"
                ],
                importance=2
            ),
            
            # Preferences and Style
            ProfileQuestion(
                id="decision_making",
                category=ProfileCategory.PREFERENCES,
                question_type=QuestionType.MULTIPLE_CHOICE,
                question="When making decisions, what's your style?",
                choices=[
                    "I like to research thoroughly before deciding",
                    "I go with my gut feeling",
                    "I prefer discussing options with others first",
                    "I like having a few good options to choose from",
                    "I prefer when someone I trust makes recommendations"
                ],
                importance=4
            ),
            
            ProfileQuestion(
                id="information_processing",
                category=ProfileCategory.PREFERENCES,
                question_type=QuestionType.MULTIPLE_CHOICE,
                question="How do you prefer to receive information?",
                choices=[
                    "Give me the headlines first, details later",
                    "I want comprehensive information upfront",
                    "Break it into digestible chunks",
                    "Use stories and examples to illustrate",
                    "Keep it conversational and interactive"
                ],
                importance=4
            ),
            
            ProfileQuestion(
                id="problem_solving",
                category=ProfileCategory.PREFERENCES,
                question_type=QuestionType.OPEN_ENDED,
                question="When you're stuck on a problem, what usually helps you most?",
                follow_up_questions=[
                    "Do you prefer to work through problems alone or with help?",
                    "What's your biggest frustration when trying to solve problems?"
                ],
                importance=3
            ),
            
            # Lifestyle and Context
            ProfileQuestion(
                id="daily_routine",
                category=ProfileCategory.LIFESTYLE,
                question_type=QuestionType.OPEN_ENDED,
                question="What does a typical day look like for you?",
                importance=3
            ),
            
            ProfileQuestion(
                id="life_priorities",
                category=ProfileCategory.LIFESTYLE,
                question_type=QuestionType.OPEN_ENDED,
                question="What are your main priorities in life right now?",
                importance=4
            ),
            
            ProfileQuestion(
                id="stress_management",
                category=ProfileCategory.LIFESTYLE,
                question_type=QuestionType.OPEN_ENDED,
                question="How do you like to unwind or de-stress?",
                importance=2
            ),
            
            # Goals and Aspirations
            ProfileQuestion(
                id="short_term_goals",
                category=ProfileCategory.GOALS,
                question_type=QuestionType.OPEN_ENDED,
                question="What are you hoping to accomplish in the next few months?",
                importance=4
            ),
            
            ProfileQuestion(
                id="long_term_vision",
                category=ProfileCategory.GOALS,
                question_type=QuestionType.OPEN_ENDED,
                question="Where do you see yourself in a few years? What's your vision for your future?",
                importance=3
            ),
            
            ProfileQuestion(
                id="learning_goals",
                category=ProfileCategory.GOALS,
                question_type=QuestionType.OPEN_ENDED,
                question="Is there anything specific you want to learn or improve at?",
                importance=3
            ),
            
            # Relationships and Social Context
            ProfileQuestion(
                id="important_relationships",
                category=ProfileCategory.RELATIONSHIPS,
                question_type=QuestionType.OPEN_ENDED,
                question="Tell me about the important people in your life (family, friends, colleagues)",
                importance=3,
                context="This helps me understand your social context"
            ),
            
            ProfileQuestion(
                id="collaboration_style",
                category=ProfileCategory.RELATIONSHIPS,
                question_type=QuestionType.MULTIPLE_CHOICE,
                question="How do you prefer to work with others?",
                choices=[
                    "I like to lead and organize",
                    "I prefer to contribute my expertise",
                    "I'm more comfortable following someone else's lead",
                    "I like collaborative brainstorming",
                    "I work best independently, checking in occasionally"
                ],
                importance=3
            ),
            
            # Technical Preferences
            ProfileQuestion(
                id="tech_comfort",
                category=ProfileCategory.TECHNICAL,
                question_type=QuestionType.RATING,
                question="How comfortable are you with technology? (1 = prefer simple, 5 = love complex tech)",
                importance=3
            ),
            
            ProfileQuestion(
                id="device_usage",
                category=ProfileCategory.TECHNICAL,
                question_type=QuestionType.OPEN_ENDED,
                question="What devices do you use most often? (phone, computer, tablet, etc.)",
                importance=2
            ),
            
            ProfileQuestion(
                id="ai_expectations",
                category=ProfileCategory.TECHNICAL,
                question_type=QuestionType.OPEN_ENDED,
                question="What do you hope to get from having a personal AI assistant?",
                follow_up_questions=[
                    "Are there specific tasks you'd like help with?",
                    "What would make this AI assistant most valuable to you?"
                ],
                importance=5
            ),
            
            # Context and Situational
            ProfileQuestion(
                id="current_challenges",
                category=ProfileCategory.CONTEXT,
                question_type=QuestionType.OPEN_ENDED,
                question="What's your biggest challenge or concern right now?",
                importance=4
            ),
            
            ProfileQuestion(
                id="energy_patterns",
                category=ProfileCategory.CONTEXT,
                question_type=QuestionType.OPEN_ENDED,
                question="When do you feel most energetic and focused during the day?",
                importance=2
            ),
            
            ProfileQuestion(
                id="support_needs",
                category=ProfileCategory.CONTEXT,
                question_type=QuestionType.OPEN_ENDED,
                question="What kind of support or help do you need most in your life right now?",
                importance=4
            )
        ]
    
    def get_initial_questions(self, count: int = 5) -> List[ProfileQuestion]:
        """Get initial set of questions for profile building"""
        high_importance = [q for q in self.questions if q.importance >= 4]
        return sorted(high_importance, key=lambda x: x.importance, reverse=True)[:count]
    
    def get_follow_up_questions(self, category: ProfileCategory, count: int = 3) -> List[ProfileQuestion]:
        """Get follow-up questions for a specific category"""
        category_questions = [q for q in self.questions if q.category == category]
        return random.sample(category_questions, min(count, len(category_questions)))
    
    def get_question_by_id(self, question_id: str) -> Optional[ProfileQuestion]:
        """Get a specific question by ID"""
        return next((q for q in self.questions if q.id == question_id), None)

class UserProfileBuilder:
    """Main class for building user profiles through conversational questioning"""
    
    def __init__(self, db_path: str = "user_profiles.db"):
        self.db_path = db_path
        self.question_bank = ProfileQuestionBank()
        self._init_database()
        
    def _init_database(self):
        """Initialize the user profiles database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT,
                    completion_percentage REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS profile_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    question_id TEXT,
                    response TEXT,
                    follow_up_responses TEXT,
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS profile_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    session_data TEXT,
                    current_question_id TEXT,
                    questions_asked INTEGER DEFAULT 0,
                    completion_status TEXT DEFAULT 'in_progress',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("User profiles database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing user profiles database: {e}")
    
    def start_profile_building(self, user_id: str) -> Dict[str, Any]:
        """Start the profile building process for a new user"""
        try:
            # Check if user already has a profile
            existing_profile = self.get_user_profile(user_id)
            if existing_profile and existing_profile.profile_completion > 0.8:
                return {
                    "status": "existing_profile",
                    "message": "Welcome back! I already know quite a bit about you. Would you like to update your profile?",
                    "profile_completion": existing_profile.profile_completion
                }
            
            # Get initial questions
            initial_questions = self.question_bank.get_initial_questions(5)
            first_question = initial_questions[0]
            
            # Create new profile session
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT OR REPLACE INTO profile_sessions 
                (user_id, session_data, current_question_id, questions_asked, completion_status)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, json.dumps({"questions_queue": [q.id for q in initial_questions]}), 
                  first_question.id, 0, "in_progress"))
            conn.commit()
            conn.close()
            
            return {
                "status": "started",
                "message": "Hi! I'm your Personal AI Assistant. To help you better, I'd love to get to know you through some friendly questions. This will help me understand your preferences and provide more personalized assistance.",
                "question": {
                    "id": first_question.id,
                    "text": first_question.question,
                    "type": first_question.question_type.value,
                    "choices": first_question.choices,
                    "context": first_question.context
                },
                "progress": {
                    "current": 1,
                    "total": len(initial_questions),
                    "category": first_question.category.value
                }
            }
            
        except Exception as e:
            logger.error(f"Error starting profile building: {e}")
            return {"status": "error", "message": "Sorry, I couldn't start the profile building process."}
    
    def process_response(self, user_id: str, question_id: str, response: str) -> Dict[str, Any]:
        """Process user response and return next question or completion"""
        try:
            # Store the response
            self._store_response(user_id, question_id, response)
            
            # Get current session
            session = self._get_current_session(user_id)
            if not session:
                return {"status": "error", "message": "No active profile session found."}
            
            # Update session progress
            questions_asked = session.get("questions_asked", 0) + 1
            questions_queue = session.get("questions_queue", [])
            
            # Remove current question from queue
            if question_id in questions_queue:
                questions_queue.remove(question_id)
            
            # Check if we should ask follow-up questions
            current_question = self.question_bank.get_question_by_id(question_id)
            if current_question and current_question.follow_up_questions:
                # Ask a follow-up question based on the response
                follow_up = self._select_follow_up_question(current_question, response)
                if follow_up:
                    return {
                        "status": "follow_up",
                        "message": "That's interesting! Let me ask you a bit more about that:",
                        "question": {
                            "id": f"{question_id}_followup",
                            "text": follow_up,
                            "type": "open_ended",
                            "context": "Follow-up question"
                        },
                        "progress": {
                            "current": questions_asked,
                            "total": len(session.get("original_questions", [])),
                            "category": current_question.category.value
                        }
                    }
            
            # Get next question
            if questions_queue:
                next_question_id = questions_queue[0]
                next_question = self.question_bank.get_question_by_id(next_question_id)
                
                if next_question:
                    # Update session
                    self._update_session(user_id, {
                        "questions_queue": questions_queue,
                        "questions_asked": questions_asked,
                        "current_question_id": next_question_id
                    })
                    
                    return {
                        "status": "next_question",
                        "message": self._get_transition_message(current_question, next_question),
                        "question": {
                            "id": next_question.id,
                            "text": next_question.question,
                            "type": next_question.question_type.value,
                            "choices": next_question.choices,
                            "context": next_question.context
                        },
                        "progress": {
                            "current": questions_asked + 1,
                            "total": len(session.get("original_questions", [])),
                            "category": next_question.category.value
                        }
                    }
            
            # Profile building complete
            self._complete_profile_building(user_id)
            profile = self.get_user_profile(user_id)
            
            return {
                "status": "completed",
                "message": f"Thank you for sharing! I feel like I know you much better now. Your profile is {int(profile.profile_completion * 100)}% complete. I'm here whenever you need help, and I'll keep learning about your preferences as we work together.",
                "profile_summary": self._generate_profile_summary(profile),
                "completion": profile.profile_completion
            }
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return {"status": "error", "message": "Sorry, I couldn't process your response."}
    
    def _store_response(self, user_id: str, question_id: str, response: str):
        """Store user response in database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO profile_responses 
            (user_id, question_id, response, confidence)
            VALUES (?, ?, ?, ?)
        ''', (user_id, question_id, response, 1.0))
        conn.commit()
        conn.close()
    
    def _get_current_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get current profile building session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT session_data, questions_asked FROM profile_sessions 
            WHERE user_id = ? AND completion_status = 'in_progress'
            ORDER BY updated_at DESC LIMIT 1
        ''', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            session_data = json.loads(result[0])
            session_data["questions_asked"] = result[1]
            return session_data
        return None
    
    def _update_session(self, user_id: str, updates: Dict[str, Any]):
        """Update profile building session"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            UPDATE profile_sessions 
            SET session_data = ?, questions_asked = ?, current_question_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ? AND completion_status = 'in_progress'
        ''', (json.dumps(updates), updates.get("questions_asked", 0), 
              updates.get("current_question_id", ""), user_id))
        conn.commit()
        conn.close()
    
    def _select_follow_up_question(self, question: ProfileQuestion, response: str) -> Optional[str]:
        """Select appropriate follow-up question based on response"""
        if not question.follow_up_questions:
            return None
        
        # Simple logic to select follow-up - can be enhanced with NLP
        if len(response) > 50:  # Detailed response
            return random.choice(question.follow_up_questions)
        elif "yes" in response.lower() or "love" in response.lower():
            return question.follow_up_questions[0] if question.follow_up_questions else None
        
        return None
    
    def _get_transition_message(self, from_question: Optional[ProfileQuestion], 
                               to_question: ProfileQuestion) -> str:
        """Generate smooth transition message between questions"""
        transitions = [
            "Thanks for sharing that!",
            "That's really helpful to know.",
            "I appreciate you telling me that.",
            "Got it, that gives me good insight.",
            "Thank you for being so open.",
            "That's great information."
        ]
        
        base_message = random.choice(transitions)
        
        if to_question.category != from_question.category if from_question else None:
            category_intros = {
                ProfileCategory.COMMUNICATION: "Now, let's talk about how you like to communicate:",
                ProfileCategory.WORK_LIFE: "I'd love to understand more about your work:",
                ProfileCategory.INTERESTS: "Tell me about your interests:",
                ProfileCategory.PREFERENCES: "Let's explore your preferences:",
                ProfileCategory.LIFESTYLE: "Now about your lifestyle:",
                ProfileCategory.GOALS: "What about your goals and aspirations?",
                ProfileCategory.RELATIONSHIPS: "Let's talk about relationships:",
                ProfileCategory.TECHNICAL: "A quick question about technology:"
            }
            
            if to_question.category in category_intros:
                return f"{base_message} {category_intros[to_question.category]}"
        
        return base_message
    
    def _complete_profile_building(self, user_id: str):
        """Complete the profile building process"""
        # Build profile from responses
        profile = self._build_profile_from_responses(user_id)
        
        # Store complete profile
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO user_profiles 
            (user_id, profile_data, completion_percentage, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (user_id, json.dumps(profile.__dict__, default=str), profile.profile_completion))
        
        # Mark session as complete
        conn.execute('''
            UPDATE profile_sessions 
            SET completion_status = 'completed', updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ? AND completion_status = 'in_progress'
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def _build_profile_from_responses(self, user_id: str) -> UserProfile:
        """Build user profile from stored responses"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT question_id, response FROM profile_responses 
            WHERE user_id = ? ORDER BY timestamp
        ''', (user_id,))
        responses = cursor.fetchall()
        conn.close()
        
        profile = UserProfile(user_id=user_id)
        
        for question_id, response in responses:
            question = self.question_bank.get_question_by_id(question_id)
            if not question:
                continue
                
            # Categorize response into profile sections
            if question.category == ProfileCategory.PERSONAL_INFO:
                if question.id == "name_preference":
                    profile.personal_info["preferred_name"] = response
                elif question.id == "personal_background":
                    profile.personal_info["background"] = response
                elif question.id == "location_context":
                    profile.personal_info["location"] = response
                    
            elif question.category == ProfileCategory.COMMUNICATION:
                if question.id == "communication_style":
                    profile.communication_style["preferred_style"] = response
                elif question.id == "explanation_preference":
                    profile.communication_style["explanation_style"] = response
                elif question.id == "feedback_style":
                    profile.communication_style["feedback_preference"] = response
                    
            elif question.category == ProfileCategory.WORK_LIFE:
                if question.id == "work_context":
                    profile.work_context["role"] = response
                elif question.id == "work_schedule":
                    profile.work_context["schedule"] = response
                elif question.id == "professional_goals":
                    profile.work_context["goals"] = response
                    
            elif question.category == ProfileCategory.INTERESTS:
                if question.id == "interests_hobbies":
                    profile.interests = response.split(",") if "," in response else [response]
                elif question.id == "learning_interests":
                    profile.interests.extend(response.split(",") if "," in response else [response])
                    
            elif question.category == ProfileCategory.PREFERENCES:
                if question.id == "decision_making":
                    profile.preferences["decision_style"] = response
                elif question.id == "information_processing":
                    profile.preferences["info_processing"] = response
                elif question.id == "problem_solving":
                    profile.preferences["problem_solving"] = response
                    
            elif question.category == ProfileCategory.LIFESTYLE:
                if question.id == "daily_routine":
                    profile.lifestyle["routine"] = response
                elif question.id == "life_priorities":
                    profile.lifestyle["priorities"] = response
                elif question.id == "stress_management":
                    profile.lifestyle["stress_management"] = response
                    
            elif question.category == ProfileCategory.GOALS:
                if question.id == "short_term_goals":
                    profile.goals.append(f"Short-term: {response}")
                elif question.id == "long_term_vision":
                    profile.goals.append(f"Long-term: {response}")
                elif question.id == "learning_goals":
                    profile.goals.append(f"Learning: {response}")
                    
            elif question.category == ProfileCategory.RELATIONSHIPS:
                if question.id == "important_relationships":
                    profile.relationships["important_people"] = response
                elif question.id == "collaboration_style":
                    profile.relationships["collaboration_style"] = response
                    
            elif question.category == ProfileCategory.TECHNICAL:
                if question.id == "tech_comfort":
                    profile.technical_preferences["comfort_level"] = response
                elif question.id == "device_usage":
                    profile.technical_preferences["devices"] = response
                elif question.id == "ai_expectations":
                    profile.technical_preferences["ai_expectations"] = response
        
        # Calculate completion percentage
        total_categories = len(ProfileCategory)
        completed_categories = sum(1 for category in ProfileCategory if self._has_category_data(profile, category))
        profile.profile_completion = completed_categories / total_categories
        
        return profile
    
    def _has_category_data(self, profile: UserProfile, category: ProfileCategory) -> bool:
        """Check if profile has data for a specific category"""
        category_mappings = {
            ProfileCategory.PERSONAL_INFO: bool(profile.personal_info),
            ProfileCategory.COMMUNICATION: bool(profile.communication_style),
            ProfileCategory.WORK_LIFE: bool(profile.work_context),
            ProfileCategory.INTERESTS: bool(profile.interests),
            ProfileCategory.PREFERENCES: bool(profile.preferences),
            ProfileCategory.LIFESTYLE: bool(profile.lifestyle),
            ProfileCategory.GOALS: bool(profile.goals),
            ProfileCategory.RELATIONSHIPS: bool(profile.relationships),
            ProfileCategory.TECHNICAL: bool(profile.technical_preferences)
        }
        
        return category_mappings.get(category, False)
    
    def _generate_profile_summary(self, profile: UserProfile) -> Dict[str, Any]:
        """Generate a summary of the user's profile"""
        summary = {
            "name": profile.personal_info.get("preferred_name", "User"),
            "completion": f"{int(profile.profile_completion * 100)}%",
            "key_traits": [],
            "communication_style": profile.communication_style.get("preferred_style", "Not specified"),
            "main_interests": profile.interests[:3] if profile.interests else ["None specified"],
            "current_goals": profile.goals[:2] if profile.goals else ["None specified"]
        }
        
        # Add key traits based on responses
        if profile.preferences.get("decision_style"):
            summary["key_traits"].append(f"Decision maker: {profile.preferences['decision_style']}")
        if profile.work_context.get("role"):
            summary["key_traits"].append(f"Role: {profile.work_context['role']}")
        if profile.lifestyle.get("priorities"):
            summary["key_traits"].append(f"Priorities: {profile.lifestyle['priorities']}")
        
        return summary
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get complete user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT profile_data, completion_percentage FROM user_profiles 
            WHERE user_id = ?
        ''', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            profile_data = json.loads(result[0])
            profile = UserProfile(**profile_data)
            profile.profile_completion = result[1]
            return profile
        return None
    
    def is_profile_complete(self, user_id: str) -> bool:
        """Check if user profile is sufficiently complete"""
        profile = self.get_user_profile(user_id)
        return profile and profile.profile_completion >= 0.6  # 60% completion threshold

# Global instance
_user_profile_builder = None

def get_user_profile_builder() -> UserProfileBuilder:
    """Get or create global user profile builder instance"""
    global _user_profile_builder
    if _user_profile_builder is None:
        _user_profile_builder = UserProfileBuilder()
    return _user_profile_builder