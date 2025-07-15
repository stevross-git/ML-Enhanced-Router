"""
Cognitive Loop Debugging System
Provides transparency into AI decision-making processes
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of AI decisions"""
    PERSONA_SWITCH = "persona_switch"
    TONE_ADJUSTMENT = "tone_adjustment"
    MEMORY_RETRIEVAL = "memory_retrieval"
    ROUTING_DECISION = "routing_decision"
    CONTEXT_SELECTION = "context_selection"
    RESPONSE_STYLE = "response_style"
    MOOD_DETECTION = "mood_detection"
    KNOWLEDGE_BRIDGE = "knowledge_bridge"

@dataclass
class DecisionFactor:
    """Represents a factor that influenced a decision"""
    factor_type: str
    description: str
    confidence: float
    weight: float
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CognitiveDecision:
    """Represents a cognitive decision made by the AI"""
    decision_id: str
    decision_type: DecisionType
    decision_made: str
    alternatives_considered: List[str]
    confidence: float
    reasoning: str
    influencing_factors: List[DecisionFactor]
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = "default"
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "decision_made": self.decision_made,
            "alternatives_considered": self.alternatives_considered,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "influencing_factors": [
                {
                    "factor_type": factor.factor_type,
                    "description": factor.description,
                    "confidence": factor.confidence,
                    "weight": factor.weight,
                    "evidence": factor.evidence,
                    "metadata": factor.metadata
                }
                for factor in self.influencing_factors
            ],
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id
        }

class CognitiveLoopDebugger:
    """Tracks and analyzes AI decision-making processes"""
    
    def __init__(self, db_path: str = "cognitive_decisions.db"):
        self.db_path = db_path
        self.decisions: Dict[str, CognitiveDecision] = {}
        self.lock = threading.Lock()
        self.current_session_id = None
        
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cognitive_decisions (
                decision_id TEXT PRIMARY KEY,
                decision_type TEXT NOT NULL,
                decision_made TEXT NOT NULL,
                alternatives_considered TEXT,
                confidence REAL NOT NULL,
                reasoning TEXT,
                influencing_factors TEXT,
                context TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'default',
                session_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decision_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                decision_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_session(self, user_id: str = "default") -> str:
        """Start a new decision tracking session"""
        session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session_id = session_id
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO decision_sessions (session_id, user_id, metadata)
            VALUES (?, ?, ?)
        ''', (session_id, user_id, "{}"))
        conn.commit()
        conn.close()
        
        return session_id
    
    def end_session(self, session_id: str = None):
        """End a decision tracking session"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE decision_sessions 
                SET ended_at = CURRENT_TIMESTAMP,
                    decision_count = (
                        SELECT COUNT(*) FROM cognitive_decisions 
                        WHERE session_id = ?
                    )
                WHERE session_id = ?
            ''', (session_id, session_id))
            conn.commit()
            conn.close()
        
        if session_id == self.current_session_id:
            self.current_session_id = None
    
    def log_persona_switch(self, from_persona: str, to_persona: str, 
                          reason: str, context: Dict[str, Any],
                          influencing_memories: List[str] = None,
                          confidence: float = 0.8,
                          user_id: str = "default") -> str:
        """Log a persona switching decision"""
        
        factors = [
            DecisionFactor(
                factor_type="context_analysis",
                description="Context suggested different persona needed",
                confidence=confidence,
                weight=0.4,
                evidence=[f"Context: {context.get('query', 'N/A')}"]
            ),
            DecisionFactor(
                factor_type="persona_compatibility",
                description=f"Target persona '{to_persona}' better suited for task",
                confidence=confidence,
                weight=0.3,
                evidence=[reason]
            )
        ]
        
        if influencing_memories:
            factors.append(DecisionFactor(
                factor_type="memory_influence",
                description="Historical interactions influenced persona choice",
                confidence=0.6,
                weight=0.3,
                evidence=influencing_memories
            ))
        
        decision = CognitiveDecision(
            decision_id=f"persona_switch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            decision_type=DecisionType.PERSONA_SWITCH,
            decision_made=f"Switch from '{from_persona}' to '{to_persona}'",
            alternatives_considered=[from_persona, to_persona],
            confidence=confidence,
            reasoning=reason,
            influencing_factors=factors,
            context=context,
            user_id=user_id,
            session_id=self.current_session_id
        )
        
        return self._store_decision(decision)
    
    def log_tone_adjustment(self, original_tone: str, adjusted_tone: str,
                           mood_detected: str, confidence: float,
                           context: Dict[str, Any],
                           user_id: str = "default") -> str:
        """Log a tone adjustment decision"""
        
        factors = [
            DecisionFactor(
                factor_type="mood_detection",
                description=f"Detected user mood: {mood_detected}",
                confidence=confidence,
                weight=0.5,
                evidence=[f"Mood confidence: {confidence}"]
            ),
            DecisionFactor(
                factor_type="tone_adaptation",
                description=f"Adjusted tone from {original_tone} to {adjusted_tone}",
                confidence=0.7,
                weight=0.5,
                evidence=[f"Query: {context.get('query', 'N/A')[:100]}"]
            )
        ]
        
        decision = CognitiveDecision(
            decision_id=f"tone_adjust_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            decision_type=DecisionType.TONE_ADJUSTMENT,
            decision_made=f"Adjust tone to {adjusted_tone}",
            alternatives_considered=[original_tone, adjusted_tone],
            confidence=confidence,
            reasoning=f"User mood '{mood_detected}' suggests {adjusted_tone} tone",
            influencing_factors=factors,
            context=context,
            user_id=user_id,
            session_id=self.current_session_id
        )
        
        return self._store_decision(decision)
    
    def log_memory_retrieval(self, query: str, retrieved_memories: List[Dict],
                            selection_criteria: str, confidence: float,
                            context: Dict[str, Any],
                            user_id: str = "default") -> str:
        """Log a memory retrieval decision"""
        
        factors = [
            DecisionFactor(
                factor_type="relevance_matching",
                description="Selected memories based on semantic relevance",
                confidence=confidence,
                weight=0.4,
                evidence=[f"Query: {query}"]
            ),
            DecisionFactor(
                factor_type="recency_weighting",
                description="Weighted memories by recency",
                confidence=0.6,
                weight=0.3,
                evidence=[f"Retrieved {len(retrieved_memories)} memories"]
            ),
            DecisionFactor(
                factor_type="context_filtering",
                description="Filtered memories by context relevance",
                confidence=0.7,
                weight=0.3,
                evidence=[selection_criteria]
            )
        ]
        
        decision = CognitiveDecision(
            decision_id=f"memory_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            decision_type=DecisionType.MEMORY_RETRIEVAL,
            decision_made=f"Retrieved {len(retrieved_memories)} relevant memories",
            alternatives_considered=["all_memories", "recent_memories", "relevant_memories"],
            confidence=confidence,
            reasoning=selection_criteria,
            influencing_factors=factors,
            context=context,
            user_id=user_id,
            session_id=self.current_session_id
        )
        
        return self._store_decision(decision)
    
    def log_routing_decision(self, query: str, chosen_route: str,
                           complexity_analysis: Dict[str, Any],
                           available_routes: List[str],
                           confidence: float,
                           context: Dict[str, Any],
                           user_id: str = "default") -> str:
        """Log a routing decision"""
        
        factors = [
            DecisionFactor(
                factor_type="complexity_analysis",
                description="Analyzed query complexity",
                confidence=confidence,
                weight=0.4,
                evidence=[f"Complexity: {complexity_analysis.get('level', 'unknown')}"]
            ),
            DecisionFactor(
                factor_type="capability_matching",
                description=f"Matched query to {chosen_route} capabilities",
                confidence=0.7,
                weight=0.3,
                evidence=[f"Available routes: {', '.join(available_routes)}"]
            ),
            DecisionFactor(
                factor_type="performance_optimization",
                description="Optimized for best performance/cost ratio",
                confidence=0.6,
                weight=0.3,
                evidence=[f"Chosen route: {chosen_route}"]
            )
        ]
        
        decision = CognitiveDecision(
            decision_id=f"routing_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            decision_type=DecisionType.ROUTING_DECISION,
            decision_made=f"Route to {chosen_route}",
            alternatives_considered=available_routes,
            confidence=confidence,
            reasoning=f"Query complexity and capabilities best match {chosen_route}",
            influencing_factors=factors,
            context=context,
            user_id=user_id,
            session_id=self.current_session_id
        )
        
        return self._store_decision(decision)
    
    def _store_decision(self, decision: CognitiveDecision) -> str:
        """Store a decision in the database"""
        with self.lock:
            self.decisions[decision.decision_id] = decision
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO cognitive_decisions 
                (decision_id, decision_type, decision_made, alternatives_considered,
                 confidence, reasoning, influencing_factors, context, user_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                decision.decision_id,
                decision.decision_type.value,
                decision.decision_made,
                json.dumps(decision.alternatives_considered),
                decision.confidence,
                decision.reasoning,
                json.dumps([factor.__dict__ for factor in decision.influencing_factors]),
                json.dumps(decision.context),
                decision.user_id,
                decision.session_id
            ))
            conn.commit()
            conn.close()
        
        return decision.decision_id
    
    def get_decision(self, decision_id: str) -> Optional[CognitiveDecision]:
        """Get a specific decision by ID"""
        return self.decisions.get(decision_id)
    
    def get_session_decisions(self, session_id: str) -> List[CognitiveDecision]:
        """Get all decisions from a session"""
        return [
            decision for decision in self.decisions.values()
            if decision.session_id == session_id
        ]
    
    def get_user_decisions(self, user_id: str, limit: int = 50) -> List[CognitiveDecision]:
        """Get recent decisions for a user"""
        user_decisions = [
            decision for decision in self.decisions.values()
            if decision.user_id == user_id
        ]
        
        # Sort by timestamp (most recent first)
        user_decisions.sort(key=lambda x: x.timestamp, reverse=True)
        return user_decisions[:limit]
    
    def get_decision_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze decision patterns for a user"""
        user_decisions = self.get_user_decisions(user_id, limit=100)
        
        if not user_decisions:
            return {"total_decisions": 0}
        
        # Analyze decision types
        decision_types = {}
        confidence_scores = []
        
        for decision in user_decisions:
            dtype = decision.decision_type.value
            decision_types[dtype] = decision_types.get(dtype, 0) + 1
            confidence_scores.append(decision.confidence)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Find most common decision type
        most_common_type = max(decision_types.items(), key=lambda x: x[1])
        
        return {
            "total_decisions": len(user_decisions),
            "decision_types": decision_types,
            "average_confidence": avg_confidence,
            "most_common_decision": most_common_type[0],
            "most_common_count": most_common_type[1],
            "time_range": {
                "earliest": user_decisions[-1].timestamp.isoformat(),
                "latest": user_decisions[0].timestamp.isoformat()
            }
        }
    
    def explain_decision(self, decision_id: str) -> Dict[str, Any]:
        """Get detailed explanation of a decision"""
        decision = self.get_decision(decision_id)
        
        if not decision:
            return {"error": "Decision not found"}
        
        explanation = {
            "decision_overview": {
                "type": decision.decision_type.value,
                "made": decision.decision_made,
                "confidence": decision.confidence,
                "timestamp": decision.timestamp.isoformat()
            },
            "reasoning": decision.reasoning,
            "alternatives_considered": decision.alternatives_considered,
            "influencing_factors": [
                {
                    "factor": factor.factor_type,
                    "description": factor.description,
                    "confidence": factor.confidence,
                    "weight": factor.weight,
                    "evidence": factor.evidence
                }
                for factor in decision.influencing_factors
            ],
            "context": decision.context
        }
        
        return explanation
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        total_decisions = len(self.decisions)
        
        if total_decisions == 0:
            return {"total_decisions": 0}
        
        # Count by decision type
        type_counts = {}
        confidence_scores = []
        
        for decision in self.decisions.values():
            dtype = decision.decision_type.value
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
            confidence_scores.append(decision.confidence)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            "total_decisions": total_decisions,
            "decision_types": type_counts,
            "average_confidence": avg_confidence,
            "high_confidence_decisions": len([c for c in confidence_scores if c > 0.8]),
            "low_confidence_decisions": len([c for c in confidence_scores if c < 0.5])
        }

# Global instance
cognitive_debugger = None

def get_cognitive_debugger():
    """Get or create global cognitive debugger instance"""
    global cognitive_debugger
    if cognitive_debugger is None:
        cognitive_debugger = CognitiveLoopDebugger()
    return cognitive_debugger