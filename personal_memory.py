"""
Enhanced Personal Memory System with Dynamic Identity Shaping and Life Timeline
Implements advanced personalization features for the Personal AI Assistant
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)

class PersonaType(Enum):
    """Different persona types for the AI assistant"""
    WORK = "work"
    FAMILY = "family"
    CREATIVE = "creative"
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    LEARNING = "learning"

class MemoryCategory(Enum):
    """Categories for memory storage"""
    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    RELATIONSHIPS = "relationships"
    PREFERENCES = "preferences"
    SKILLS = "skills"
    TIMELINE = "timeline"
    HABITS = "habits"
    GOALS = "goals"

class ConfidenceLevel(Enum):
    """Confidence levels for memory entries"""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

@dataclass
class Persona:
    """User persona definition"""
    id: str
    name: str
    type: PersonaType
    description: str
    tone: str = "friendly"
    formality: float = 0.5  # 0-1 scale
    verbosity: float = 0.5  # 0-1 scale
    active_hours: List[int] = field(default_factory=list)  # Hours when this persona is active
    context_keywords: List[str] = field(default_factory=list)
    memory_filters: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

@dataclass
class MemoryEntry:
    """Enhanced memory entry with confidence and metadata"""
    id: str
    user_id: str
    content: str
    category: MemoryCategory
    confidence: float
    timestamp: datetime
    source: str  # How this memory was acquired
    personas: List[str] = field(default_factory=list)  # Which personas this applies to
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    is_frozen: bool = False  # Frozen memories cannot be updated

@dataclass
class TimelineEvent:
    """Life timeline event"""
    id: str
    user_id: str
    title: str
    description: str
    date: datetime
    category: str
    importance: float  # 0-1 scale
    tags: List[str] = field(default_factory=list)
    related_memories: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class UserTrait:
    """Dynamically learned user traits"""
    id: str
    user_id: str
    trait_name: str
    trait_value: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 1

class PersonalMemorySystem:
    """Enhanced personal memory system with dynamic identity shaping"""
    
    def __init__(self, db_path: str = "personal_memory.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize the enhanced personal memory database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Personas table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS personas (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    description TEXT,
                    tone TEXT DEFAULT 'friendly',
                    formality REAL DEFAULT 0.5,
                    verbosity REAL DEFAULT 0.5,
                    active_hours TEXT,
                    context_keywords TEXT,
                    memory_filters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Enhanced memories table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS enhanced_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT NOT NULL,
                    personas TEXT,
                    tags TEXT,
                    metadata TEXT,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    is_frozen BOOLEAN DEFAULT 0
                )
            ''')
            
            # Timeline events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS timeline_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    date TIMESTAMP NOT NULL,
                    category TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    tags TEXT,
                    related_memories TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User traits table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_traits (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    trait_name TEXT NOT NULL,
                    trait_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_count INTEGER DEFAULT 1
                )
            ''')
            
            # Reflection logs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS reflection_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    reflection_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    insights TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Enhanced personal memory database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing enhanced personal memory database: {e}")
    
    def create_persona(self, user_id: str, name: str, persona_type: PersonaType, 
                      description: str = "", **kwargs) -> Persona:
        """Create a new persona for the user"""
        try:
            persona_id = f"persona_{user_id}_{persona_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            persona = Persona(
                id=persona_id,
                name=name,
                type=persona_type,
                description=description,
                tone=kwargs.get('tone', 'friendly'),
                formality=kwargs.get('formality', 0.5),
                verbosity=kwargs.get('verbosity', 0.5),
                active_hours=kwargs.get('active_hours', []),
                context_keywords=kwargs.get('context_keywords', []),
                memory_filters=kwargs.get('memory_filters', [])
            )
            
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO personas 
                (id, user_id, name, type, description, tone, formality, verbosity, 
                 active_hours, context_keywords, memory_filters, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (persona.id, user_id, persona.name, persona.type.value, persona.description,
                  persona.tone, persona.formality, persona.verbosity,
                  json.dumps(persona.active_hours), json.dumps(persona.context_keywords),
                  json.dumps(persona.memory_filters), persona.created_at, persona.is_active))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created persona '{name}' for user {user_id}")
            return persona
            
        except Exception as e:
            logger.error(f"Error creating persona: {e}")
            return None
    
    def get_active_persona(self, user_id: str, context: str = "", 
                          current_time: datetime = None) -> Optional[Persona]:
        """Determine the active persona based on context and time"""
        try:
            if current_time is None:
                current_time = datetime.now()
            
            current_hour = current_time.hour
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT * FROM personas 
                WHERE user_id = ? AND is_active = 1
                ORDER BY created_at DESC
            ''', (user_id,))
            
            personas = []
            for row in cursor.fetchall():
                active_hours = json.loads(row[8]) if row[8] else []
                context_keywords = json.loads(row[9]) if row[9] else []
                memory_filters = json.loads(row[10]) if row[10] else []
                
                persona = Persona(
                    id=row[0], name=row[2], type=PersonaType(row[3]),
                    description=row[4], tone=row[5], formality=row[6],
                    verbosity=row[7], active_hours=active_hours,
                    context_keywords=context_keywords, memory_filters=memory_filters,
                    created_at=datetime.fromisoformat(row[11]), is_active=bool(row[12])
                )
                personas.append(persona)
            
            conn.close()
            
            if not personas:
                return None
            
            # Score personas based on context and time
            best_persona = None
            best_score = 0
            
            for persona in personas:
                score = 0
                
                # Time-based scoring
                if persona.active_hours and current_hour in persona.active_hours:
                    score += 3
                
                # Context-based scoring
                if persona.context_keywords and context:
                    context_lower = context.lower()
                    for keyword in persona.context_keywords:
                        if keyword.lower() in context_lower:
                            score += 2
                
                # Default persona gets base score
                if persona.type == PersonaType.CASUAL:
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_persona = persona
            
            return best_persona or personas[0]  # Return first persona if no clear winner
            
        except Exception as e:
            logger.error(f"Error getting active persona: {e}")
            return None
    
    def store_memory(self, user_id: str, content: str, category: MemoryCategory,
                    confidence: float = 0.6, source: str = "user_input",
                    personas: List[str] = None, tags: List[str] = None,
                    metadata: Dict[str, Any] = None) -> bool:
        """Store an enhanced memory entry"""
        try:
            memory_id = f"memory_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            memory = MemoryEntry(
                id=memory_id,
                user_id=user_id,
                content=content,
                category=category,
                confidence=confidence,
                timestamp=datetime.now(),
                source=source,
                personas=personas or [],
                tags=tags or [],
                metadata=metadata or {}
            )
            
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO enhanced_memories 
                (id, user_id, content, category, confidence, timestamp, source, 
                 personas, tags, metadata, last_accessed, access_count, is_frozen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (memory.id, memory.user_id, memory.content, memory.category.value,
                  memory.confidence, memory.timestamp, memory.source,
                  json.dumps(memory.personas), json.dumps(memory.tags),
                  json.dumps(memory.metadata), memory.last_accessed,
                  memory.access_count, memory.is_frozen))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored memory for user {user_id}: {content[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False
    
    def add_timeline_event(self, user_id: str, title: str, description: str,
                          date: datetime, category: str, importance: float = 0.5,
                          tags: List[str] = None) -> bool:
        """Add an event to the user's life timeline"""
        try:
            event_id = f"timeline_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            event = TimelineEvent(
                id=event_id,
                user_id=user_id,
                title=title,
                description=description,
                date=date,
                category=category,
                importance=importance,
                tags=tags or []
            )
            
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO timeline_events 
                (id, user_id, title, description, date, category, importance, tags, related_memories, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (event.id, event.user_id, event.title, event.description,
                  event.date, event.category, event.importance,
                  json.dumps(event.tags), json.dumps(event.related_memories),
                  event.created_at))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added timeline event for user {user_id}: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding timeline event: {e}")
            return False
    
    def update_user_trait(self, user_id: str, trait_name: str, trait_value: str,
                         confidence: float, evidence: str = "") -> bool:
        """Update or create a user trait with confidence weighting"""
        try:
            # Check if trait already exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT * FROM user_traits WHERE user_id = ? AND trait_name = ?
            ''', (user_id, trait_name))
            
            existing_trait = cursor.fetchone()
            
            if existing_trait:
                # Update existing trait with weighted confidence
                old_confidence = existing_trait[4]
                old_evidence = json.loads(existing_trait[5]) if existing_trait[5] else []
                update_count = existing_trait[7] + 1
                
                # Calculate new confidence using weighted average
                new_confidence = (old_confidence * existing_trait[7] + confidence) / update_count
                new_evidence = old_evidence + [evidence] if evidence else old_evidence
                
                conn.execute('''
                    UPDATE user_traits 
                    SET trait_value = ?, confidence = ?, evidence = ?, 
                        last_updated = ?, update_count = ?
                    WHERE user_id = ? AND trait_name = ?
                ''', (trait_value, new_confidence, json.dumps(new_evidence),
                      datetime.now(), update_count, user_id, trait_name))
            else:
                # Create new trait
                trait_id = f"trait_{user_id}_{trait_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                evidence_list = [evidence] if evidence else []
                
                conn.execute('''
                    INSERT INTO user_traits 
                    (id, user_id, trait_name, trait_value, confidence, evidence, last_updated, update_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (trait_id, user_id, trait_name, trait_value, confidence,
                      json.dumps(evidence_list), datetime.now(), 1))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated trait '{trait_name}' for user {user_id}: {trait_value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating user trait: {e}")
            return False
    
    def get_timeline_events(self, user_id: str, start_date: datetime = None,
                           end_date: datetime = None, category: str = None) -> List[TimelineEvent]:
        """Retrieve timeline events for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM timeline_events WHERE user_id = ?"
            params = [user_id]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            if category:
                query += " AND category = ?"
                params.append(category)
            
            query += " ORDER BY date DESC"
            
            cursor = conn.execute(query, params)
            events = []
            
            for row in cursor.fetchall():
                event = TimelineEvent(
                    id=row[0],
                    user_id=row[1],
                    title=row[2],
                    description=row[3],
                    date=datetime.fromisoformat(row[4]),
                    category=row[5],
                    importance=row[6],
                    tags=json.loads(row[7]) if row[7] else [],
                    related_memories=json.loads(row[8]) if row[8] else [],
                    created_at=datetime.fromisoformat(row[9])
                )
                events.append(event)
            
            conn.close()
            return events
            
        except Exception as e:
            logger.error(f"Error retrieving timeline events: {e}")
            return []
    
    def get_user_traits(self, user_id: str, min_confidence: float = 0.5) -> List[UserTrait]:
        """Get user traits above a confidence threshold"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT * FROM user_traits 
                WHERE user_id = ? AND confidence >= ?
                ORDER BY confidence DESC, last_updated DESC
            ''', (user_id, min_confidence))
            
            traits = []
            for row in cursor.fetchall():
                trait = UserTrait(
                    id=row[0],
                    user_id=row[1],
                    trait_name=row[2],
                    trait_value=row[3],
                    confidence=row[4],
                    evidence=json.loads(row[5]) if row[5] else [],
                    last_updated=datetime.fromisoformat(row[6]),
                    update_count=row[7]
                )
                traits.append(trait)
            
            conn.close()
            return traits
            
        except Exception as e:
            logger.error(f"Error retrieving user traits: {e}")
            return []
    
    def store_reflection(self, user_id: str, session_id: str, reflection_type: str,
                        content: str, insights: str = "") -> bool:
        """Store contextual reflection for learning"""
        try:
            reflection_id = f"reflection_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO reflection_logs 
                (id, user_id, session_id, reflection_type, content, insights, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (reflection_id, user_id, session_id, reflection_type, content, insights, datetime.now()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored reflection for user {user_id}: {reflection_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing reflection: {e}")
            return False
    
    def analyze_interaction_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user interaction patterns for autonomous profile refinement"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Analyze communication patterns
            cursor = conn.execute('''
                SELECT content, timestamp FROM enhanced_memories 
                WHERE user_id = ? AND source = 'user_input'
                ORDER BY timestamp DESC LIMIT 50
            ''', (user_id,))
            
            recent_messages = cursor.fetchall()
            
            # Analyze traits
            cursor = conn.execute('''
                SELECT trait_name, trait_value, confidence FROM user_traits 
                WHERE user_id = ? AND confidence > 0.6
                ORDER BY confidence DESC
            ''', (user_id,))
            
            traits = cursor.fetchall()
            
            # Analyze timeline patterns
            cursor = conn.execute('''
                SELECT category, COUNT(*) as count FROM timeline_events 
                WHERE user_id = ? GROUP BY category
                ORDER BY count DESC
            ''', (user_id,))
            
            timeline_categories = cursor.fetchall()
            
            conn.close()
            
            # Generate analysis
            analysis = {
                "message_count": len(recent_messages),
                "high_confidence_traits": len(traits),
                "dominant_timeline_categories": [cat[0] for cat in timeline_categories[:3]],
                "communication_style": self._analyze_communication_style(recent_messages),
                "interaction_frequency": self._analyze_interaction_frequency(recent_messages),
                "trait_stability": self._analyze_trait_stability(traits)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing interaction patterns: {e}")
            return {}
    
    def _analyze_communication_style(self, messages: List[Tuple]) -> Dict[str, Any]:
        """Analyze communication style from messages"""
        if not messages:
            return {}
        
        total_chars = sum(len(msg[0]) for msg in messages)
        avg_length = total_chars / len(messages)
        
        # Count question marks and exclamation points
        questions = sum(msg[0].count('?') for msg in messages)
        exclamations = sum(msg[0].count('!') for msg in messages)
        
        return {
            "average_message_length": avg_length,
            "question_frequency": questions / len(messages),
            "exclamation_frequency": exclamations / len(messages),
            "preferred_length": "concise" if avg_length < 50 else "detailed" if avg_length > 150 else "moderate"
        }
    
    def _analyze_interaction_frequency(self, messages: List[Tuple]) -> Dict[str, Any]:
        """Analyze when user typically interacts"""
        if not messages:
            return {}
        
        hours = [datetime.fromisoformat(msg[1]).hour for msg in messages]
        hour_counts = {}
        
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "peak_hours": [hour for hour, count in peak_hours],
            "interaction_pattern": self._classify_interaction_pattern(peak_hours)
        }
    
    def _classify_interaction_pattern(self, peak_hours: List[Tuple]) -> str:
        """Classify interaction pattern based on peak hours"""
        if not peak_hours:
            return "unknown"
        
        top_hour = peak_hours[0][0]
        
        if 6 <= top_hour <= 9:
            return "morning_person"
        elif 9 <= top_hour <= 12:
            return "morning_worker"
        elif 12 <= top_hour <= 14:
            return "lunch_break"
        elif 14 <= top_hour <= 17:
            return "afternoon_worker"
        elif 17 <= top_hour <= 20:
            return "evening_person"
        elif 20 <= top_hour <= 23:
            return "night_owl"
        else:
            return "late_night"
    
    def _analyze_trait_stability(self, traits: List[Tuple]) -> Dict[str, Any]:
        """Analyze stability of user traits"""
        if not traits:
            return {}
        
        high_confidence = sum(1 for trait in traits if trait[2] > 0.8)
        medium_confidence = sum(1 for trait in traits if 0.6 <= trait[2] <= 0.8)
        
        return {
            "high_confidence_traits": high_confidence,
            "medium_confidence_traits": medium_confidence,
            "stability_score": high_confidence / len(traits) if traits else 0
        }

# Global instance
_personal_memory_system = None

def get_personal_memory_system() -> PersonalMemorySystem:
    """Get or create global personal memory system instance"""
    global _personal_memory_system
    if _personal_memory_system is None:
        _personal_memory_system = PersonalMemorySystem()
    return _personal_memory_system