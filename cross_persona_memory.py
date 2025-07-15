"""
Cross-Persona Memory Inference System
Bridges knowledge between personas while preserving boundaries
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
import threading
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryVisibility(Enum):
    """Memory visibility rules"""
    SHARED = "shared"          # Available to all personas
    ISOLATED = "isolated"      # Confined to specific persona
    SUGGESTIVE = "suggestive"  # Suggest linking to other personas

class LinkageType(Enum):
    """Types of persona linkages"""
    SKILL_TRANSFER = "skill_transfer"        # Skills applicable across personas
    INTEREST_OVERLAP = "interest_overlap"    # Shared interests
    CONTEXT_BRIDGE = "context_bridge"        # Contextual connections
    PREFERENCE_SYNC = "preference_sync"      # Shared preferences
    KNOWLEDGE_SHARE = "knowledge_share"      # Factual knowledge sharing

@dataclass
class PersonaLinkage:
    """Represents a link between personas"""
    persona_1: str
    persona_2: str
    linkage_type: LinkageType
    confidence: float
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    user_approved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryBridge:
    """Represents a memory bridge between personas"""
    source_persona: str
    target_persona: str
    memory_id: str
    bridge_reason: str
    confidence: float
    visibility: MemoryVisibility
    suggested_at: datetime = field(default_factory=datetime.now)
    user_action: Optional[str] = None  # accepted, rejected, pending
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrossPersonaInsight:
    """Insight generated from cross-persona analysis"""
    insight_type: str
    description: str
    involved_personas: List[str]
    confidence: float
    evidence: List[str]
    suggestion: str
    timestamp: datetime = field(default_factory=datetime.now)

class CrossPersonaMemorySystem:
    """Manages memory inference across personas"""
    
    def __init__(self, db_path: str = "cross_persona_memory.db"):
        self.db_path = db_path
        self.linkages: Dict[str, PersonaLinkage] = {}
        self.bridges: Dict[str, MemoryBridge] = {}
        self.insights: List[CrossPersonaInsight] = []
        self.lock = threading.Lock()
        
        self._init_database()
        self._load_from_database()
    
    def _init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Persona linkages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persona_linkages (
                id TEXT PRIMARY KEY,
                persona_1 TEXT NOT NULL,
                persona_2 TEXT NOT NULL,
                linkage_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                user_approved BOOLEAN DEFAULT FALSE,
                metadata TEXT
            )
        ''')
        
        # Memory bridges table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_bridges (
                id TEXT PRIMARY KEY,
                source_persona TEXT NOT NULL,
                target_persona TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                bridge_reason TEXT,
                confidence REAL NOT NULL,
                visibility TEXT NOT NULL,
                suggested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_action TEXT,
                metadata TEXT
            )
        ''')
        
        # Cross-persona insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_persona_insights (
                id TEXT PRIMARY KEY,
                insight_type TEXT NOT NULL,
                description TEXT,
                involved_personas TEXT,
                confidence REAL NOT NULL,
                evidence TEXT,
                suggestion TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_from_database(self):
        """Load existing data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load linkages
        cursor.execute('SELECT * FROM persona_linkages')
        for row in cursor.fetchall():
            linkage = PersonaLinkage(
                persona_1=row[1],
                persona_2=row[2],
                linkage_type=LinkageType(row[3]),
                confidence=row[4],
                description=row[5],
                created_at=datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
                last_used=datetime.fromisoformat(row[7]) if row[7] else None,
                usage_count=row[8] or 0,
                user_approved=bool(row[9]),
                metadata=json.loads(row[10]) if row[10] else {}
            )
            self.linkages[row[0]] = linkage
        
        # Load bridges
        cursor.execute('SELECT * FROM memory_bridges')
        for row in cursor.fetchall():
            bridge = MemoryBridge(
                source_persona=row[1],
                target_persona=row[2],
                memory_id=row[3],
                bridge_reason=row[4],
                confidence=row[5],
                visibility=MemoryVisibility(row[6]),
                suggested_at=datetime.fromisoformat(row[7]) if row[7] else datetime.now(),
                user_action=row[8],
                metadata=json.loads(row[9]) if row[9] else {}
            )
            self.bridges[row[0]] = bridge
        
        conn.close()
    
    def analyze_persona_compatibility(self, persona_1: str, persona_2: str, 
                                    persona_1_data: Dict, persona_2_data: Dict) -> List[PersonaLinkage]:
        """Analyze compatibility between two personas"""
        linkages = []
        
        # Check for skill overlaps
        skills_1 = set(persona_1_data.get('skills', []))
        skills_2 = set(persona_2_data.get('skills', []))
        skill_overlap = skills_1.intersection(skills_2)
        
        if skill_overlap:
            linkages.append(PersonaLinkage(
                persona_1=persona_1,
                persona_2=persona_2,
                linkage_type=LinkageType.SKILL_TRANSFER,
                confidence=min(0.9, len(skill_overlap) * 0.2),
                description=f"Shared skills: {', '.join(skill_overlap)}",
                metadata={"shared_skills": list(skill_overlap)}
            ))
        
        # Check for interest overlaps
        interests_1 = set(persona_1_data.get('interests', []))
        interests_2 = set(persona_2_data.get('interests', []))
        interest_overlap = interests_1.intersection(interests_2)
        
        if interest_overlap:
            linkages.append(PersonaLinkage(
                persona_1=persona_1,
                persona_2=persona_2,
                linkage_type=LinkageType.INTEREST_OVERLAP,
                confidence=min(0.8, len(interest_overlap) * 0.15),
                description=f"Shared interests: {', '.join(interest_overlap)}",
                metadata={"shared_interests": list(interest_overlap)}
            ))
        
        # Check for preference similarities
        prefs_1 = persona_1_data.get('preferences', {})
        prefs_2 = persona_2_data.get('preferences', {})
        
        compatible_prefs = []
        for pref_key in prefs_1:
            if pref_key in prefs_2:
                val_1 = prefs_1[pref_key]
                val_2 = prefs_2[pref_key]
                
                # Check if values are similar (for numeric values)
                if isinstance(val_1, (int, float)) and isinstance(val_2, (int, float)):
                    if abs(val_1 - val_2) < 0.3:
                        compatible_prefs.append(pref_key)
                elif val_1 == val_2:
                    compatible_prefs.append(pref_key)
        
        if compatible_prefs:
            linkages.append(PersonaLinkage(
                persona_1=persona_1,
                persona_2=persona_2,
                linkage_type=LinkageType.PREFERENCE_SYNC,
                confidence=min(0.7, len(compatible_prefs) * 0.2),
                description=f"Compatible preferences: {', '.join(compatible_prefs)}",
                metadata={"compatible_preferences": compatible_prefs}
            ))
        
        return linkages
    
    def suggest_memory_bridge(self, memory_id: str, source_persona: str, 
                            memory_content: str, available_personas: List[str]) -> List[MemoryBridge]:
        """Suggest bridging a memory to other personas"""
        suggestions = []
        
        # Simple keyword-based analysis for now
        knowledge_keywords = ["learned", "discovered", "found out", "realized", "technique", "method"]
        skill_keywords = ["practice", "exercise", "training", "skill", "ability"]
        preference_keywords = ["prefer", "like", "enjoy", "love", "favorite"]
        
        content_lower = memory_content.lower()
        
        for target_persona in available_personas:
            if target_persona == source_persona:
                continue
            
            confidence = 0.0
            bridge_reason = ""
            visibility = MemoryVisibility.SUGGESTIVE
            
            # Check for knowledge sharing potential
            if any(keyword in content_lower for keyword in knowledge_keywords):
                confidence += 0.3
                bridge_reason = "Contains transferable knowledge"
                visibility = MemoryVisibility.SUGGESTIVE
            
            # Check for skill transfer potential
            if any(keyword in content_lower for keyword in skill_keywords):
                confidence += 0.4
                bridge_reason = "Contains skill-related information"
                visibility = MemoryVisibility.SUGGESTIVE
            
            # Check for preference sharing
            if any(keyword in content_lower for keyword in preference_keywords):
                confidence += 0.2
                bridge_reason = "Contains preference information"
                visibility = MemoryVisibility.SHARED
            
            # Check if personas are already linked
            linkage_key = f"{source_persona}_{target_persona}"
            reverse_key = f"{target_persona}_{source_persona}"
            
            if linkage_key in self.linkages or reverse_key in self.linkages:
                confidence += 0.2
                bridge_reason += " (personas are linked)"
            
            if confidence > 0.4:
                suggestions.append(MemoryBridge(
                    source_persona=source_persona,
                    target_persona=target_persona,
                    memory_id=memory_id,
                    bridge_reason=bridge_reason,
                    confidence=confidence,
                    visibility=visibility,
                    metadata={"content_preview": memory_content[:100]}
                ))
        
        return suggestions
    
    def generate_cross_persona_insights(self, user_id: str, persona_data: Dict[str, Dict]) -> List[CrossPersonaInsight]:
        """Generate insights from cross-persona analysis"""
        insights = []
        
        personas = list(persona_data.keys())
        
        # Find personas with complementary skills
        for i, persona_1 in enumerate(personas):
            for persona_2 in personas[i+1:]:
                data_1 = persona_data[persona_1]
                data_2 = persona_data[persona_2]
                
                # Check for skill complementarity
                skills_1 = set(data_1.get('skills', []))
                skills_2 = set(data_2.get('skills', []))
                
                if skills_1 and skills_2 and not skills_1.intersection(skills_2):
                    insights.append(CrossPersonaInsight(
                        insight_type="skill_complementarity",
                        description=f"{persona_1} and {persona_2} have complementary skills",
                        involved_personas=[persona_1, persona_2],
                        confidence=0.7,
                        evidence=[
                            f"{persona_1} skills: {', '.join(list(skills_1)[:3])}",
                            f"{persona_2} skills: {', '.join(list(skills_2)[:3])}"
                        ],
                        suggestion=f"Consider combining {persona_1} and {persona_2} for complex tasks"
                    ))
        
        # Find personas with similar communication styles
        style_groups = defaultdict(list)
        for persona, data in persona_data.items():
            style = data.get('communication_style', 'default')
            style_groups[style].append(persona)
        
        for style, personas_list in style_groups.items():
            if len(personas_list) > 1:
                insights.append(CrossPersonaInsight(
                    insight_type="communication_style_similarity",
                    description=f"Multiple personas share {style} communication style",
                    involved_personas=personas_list,
                    confidence=0.6,
                    evidence=[f"All use {style} communication style"],
                    suggestion="Consider consolidating similar personas or differentiating their purposes"
                ))
        
        return insights
    
    def get_persona_linkage_graph(self, user_id: str) -> Dict[str, Any]:
        """Generate a persona linkage graph"""
        graph = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "total_linkages": len(self.linkages),
                "generated_at": datetime.now().isoformat()
            }
        }
        
        # Get all personas involved in linkages
        personas = set()
        for linkage in self.linkages.values():
            personas.add(linkage.persona_1)
            personas.add(linkage.persona_2)
        
        # Add nodes
        for persona in personas:
            graph["nodes"].append({
                "id": persona,
                "name": persona,
                "type": "persona"
            })
        
        # Add edges
        for linkage_id, linkage in self.linkages.items():
            graph["edges"].append({
                "id": linkage_id,
                "source": linkage.persona_1,
                "target": linkage.persona_2,
                "type": linkage.linkage_type.value,
                "confidence": linkage.confidence,
                "description": linkage.description,
                "user_approved": linkage.user_approved
            })
        
        return graph
    
    def approve_memory_bridge(self, bridge_id: str, user_decision: str) -> bool:
        """Approve or reject a memory bridge"""
        if bridge_id not in self.bridges:
            return False
        
        with self.lock:
            self.bridges[bridge_id].user_action = user_decision
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE memory_bridges 
                SET user_action = ? 
                WHERE id = ?
            ''', (user_decision, bridge_id))
            conn.commit()
            conn.close()
        
        return True
    
    def get_memory_bridge_suggestions(self, persona: str, limit: int = 5) -> List[MemoryBridge]:
        """Get pending memory bridge suggestions for a persona"""
        suggestions = []
        
        for bridge in self.bridges.values():
            if (bridge.target_persona == persona and 
                bridge.user_action is None and
                len(suggestions) < limit):
                suggestions.append(bridge)
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cross-persona memory statistics"""
        approved_bridges = sum(1 for bridge in self.bridges.values() if bridge.user_action == "accepted")
        rejected_bridges = sum(1 for bridge in self.bridges.values() if bridge.user_action == "rejected")
        pending_bridges = sum(1 for bridge in self.bridges.values() if bridge.user_action is None)
        
        return {
            "total_linkages": len(self.linkages),
            "approved_linkages": sum(1 for linkage in self.linkages.values() if linkage.user_approved),
            "total_bridges": len(self.bridges),
            "approved_bridges": approved_bridges,
            "rejected_bridges": rejected_bridges,
            "pending_bridges": pending_bridges,
            "total_insights": len(self.insights),
            "linkage_types": {
                linkage_type.value: sum(1 for linkage in self.linkages.values() 
                                      if linkage.linkage_type == linkage_type)
                for linkage_type in LinkageType
            }
        }

# Global instance
cross_persona_system = None

def get_cross_persona_system():
    """Get or create global cross-persona system instance"""
    global cross_persona_system
    if cross_persona_system is None:
        cross_persona_system = CrossPersonaMemorySystem()
    return cross_persona_system