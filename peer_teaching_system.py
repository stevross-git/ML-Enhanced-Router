"""
Peer-Teaching & Collaborative Agents System
Agent communities that share strategies, federated knowledge exchange, and multi-agent debate/critique
"""

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import hashlib
from collections import defaultdict
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LessonType(Enum):
    """Types of lessons agents can share"""
    STRATEGY = "strategy"
    PATTERN = "pattern"
    SOLUTION = "solution"
    OPTIMIZATION = "optimization"
    ERROR_CORRECTION = "error_correction"
    DOMAIN_INSIGHT = "domain_insight"
    TOOLKIT_USAGE = "toolkit_usage"
    COLLABORATIVE_APPROACH = "collaborative_approach"

class AgentSpecialization(Enum):
    """Agent specializations for task-specific agents"""
    MATH_SOLVER = "math_solver"
    WRITING_ASSISTANT = "writing_assistant"
    CODE_HELPER = "code_helper"
    GRAMMAR_CHECKER = "grammar_checker"
    RESEARCH_ASSISTANT = "research_assistant"
    CREATIVE_WRITER = "creative_writer"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"
    TRANSLATOR = "translator"
    DEBUGGER = "debugger"

class ConsensusMethod(Enum):
    """Methods for multi-agent consensus"""
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    RANKED_CHOICE = "ranked_choice"
    EXPERT_WEIGHTED = "expert_weighted"
    DEBATE_RESOLUTION = "debate_resolution"

@dataclass
class AgentLesson:
    """A lesson learned by an agent that can be shared with peers"""
    lesson_id: str
    agent_id: str
    lesson_type: LessonType
    domain: str
    title: str
    content: str
    strategy_steps: List[str]
    effectiveness_score: float
    usage_context: str
    success_metrics: Dict[str, float]
    created_at: datetime
    anonymized_data: Dict[str, Any] = field(default_factory=dict)
    peer_validations: List[Dict[str, Any]] = field(default_factory=list)
    adoption_count: int = 0
    improvement_suggestions: List[str] = field(default_factory=list)

@dataclass
class KnowledgeContribution:
    """Anonymized knowledge contribution for federated learning"""
    contribution_id: str
    agent_specialization: AgentSpecialization
    query_type: str
    toolkit_used: str
    approach_summary: str
    performance_metrics: Dict[str, float]
    lessons_learned: List[str]
    optimization_tips: List[str]
    error_patterns: List[str]
    timestamp: datetime
    anonymized_hash: str

@dataclass
class AgentDebatePosition:
    """Position taken by an agent in a multi-agent debate"""
    agent_id: str
    agent_specialization: AgentSpecialization
    position: str
    reasoning: str
    confidence: float
    supporting_evidence: List[str]
    counter_arguments: List[str]
    peer_critiques: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class CollaborativeSession:
    """A collaborative session between multiple agents"""
    session_id: str
    initiator_agent: str
    participating_agents: List[str]
    task_description: str
    session_type: str
    agent_contributions: List[Dict[str, Any]]
    consensus_result: Optional[str] = None
    consensus_method: ConsensusMethod = ConsensusMethod.MAJORITY_VOTE
    session_start: datetime = field(default_factory=datetime.now)
    session_end: Optional[datetime] = None
    lessons_generated: List[str] = field(default_factory=list)
    cross_corrections: List[Dict[str, Any]] = field(default_factory=list)

class PeerTeachingSystem:
    """Main system for managing peer teaching and collaborative agents"""
    
    def __init__(self, db_path: str = "peer_teaching.db"):
        self.db_path = db_path
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Dict[str, CollaborativeSession] = {}
        self.knowledge_compendium: Dict[str, List[KnowledgeContribution]] = defaultdict(list)
        self.lesson_library: Dict[str, AgentLesson] = {}
        self.specialist_agents: Dict[AgentSpecialization, List[str]] = defaultdict(list)
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for peer teaching data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Agent lessons table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_lessons (
                    lesson_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    lesson_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    strategy_steps TEXT NOT NULL,
                    effectiveness_score REAL NOT NULL,
                    usage_context TEXT NOT NULL,
                    success_metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    anonymized_data TEXT,
                    adoption_count INTEGER DEFAULT 0
                )
            ''')
            
            # Knowledge contributions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_contributions (
                    contribution_id TEXT PRIMARY KEY,
                    agent_specialization TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    toolkit_used TEXT NOT NULL,
                    approach_summary TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    lessons_learned TEXT NOT NULL,
                    optimization_tips TEXT NOT NULL,
                    error_patterns TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    anonymized_hash TEXT NOT NULL
                )
            ''')
            
            # Collaborative sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS collaborative_sessions (
                    session_id TEXT PRIMARY KEY,
                    initiator_agent TEXT NOT NULL,
                    participating_agents TEXT NOT NULL,
                    task_description TEXT NOT NULL,
                    session_type TEXT NOT NULL,
                    agent_contributions TEXT NOT NULL,
                    consensus_result TEXT,
                    consensus_method TEXT NOT NULL,
                    session_start TEXT NOT NULL,
                    session_end TEXT,
                    lessons_generated TEXT,
                    cross_corrections TEXT
                )
            ''')
            
            # Agent registry table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_registry (
                    agent_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    specialization TEXT NOT NULL,
                    capabilities TEXT NOT NULL,
                    learning_stats TEXT NOT NULL,
                    peer_connections TEXT NOT NULL,
                    registered_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Peer teaching database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def register_agent(self, agent_id: str, agent_name: str, 
                      specialization: AgentSpecialization, 
                      capabilities: List[str]) -> bool:
        """Register a new agent in the peer teaching system"""
        try:
            agent_data = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "specialization": specialization,
                "capabilities": capabilities,
                "learning_stats": {
                    "lessons_contributed": 0,
                    "lessons_adopted": 0,
                    "collaborations_participated": 0,
                    "peer_rating": 0.0
                },
                "peer_connections": [],
                "registered_at": datetime.now()
            }
            
            self.agent_registry[agent_id] = agent_data
            self.specialist_agents[specialization].append(agent_id)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO agent_registry 
                (agent_id, agent_name, specialization, capabilities, learning_stats, peer_connections, registered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                agent_id, agent_name, specialization.value,
                json.dumps(capabilities), json.dumps(agent_data["learning_stats"]),
                json.dumps(agent_data["peer_connections"]), agent_data["registered_at"].isoformat()
            ))
            conn.commit()
            conn.close()
            
            logger.info(f"Agent {agent_name} registered with specialization {specialization.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering agent {agent_id}: {e}")
            return False
    
    def contribute_lesson(self, agent_id: str, lesson_type: LessonType, 
                         domain: str, title: str, content: str,
                         strategy_steps: List[str], effectiveness_score: float,
                         usage_context: str, success_metrics: Dict[str, float]) -> str:
        """Agent contributes a lesson to the peer teaching system"""
        try:
            lesson_id = str(uuid.uuid4())
            lesson = AgentLesson(
                lesson_id=lesson_id,
                agent_id=agent_id,
                lesson_type=lesson_type,
                domain=domain,
                title=title,
                content=content,
                strategy_steps=strategy_steps,
                effectiveness_score=effectiveness_score,
                usage_context=usage_context,
                success_metrics=success_metrics,
                created_at=datetime.now(),
                anonymized_data=self._anonymize_lesson_data(agent_id, content)
            )
            
            self.lesson_library[lesson_id] = lesson
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO agent_lessons 
                (lesson_id, agent_id, lesson_type, domain, title, content, strategy_steps, 
                 effectiveness_score, usage_context, success_metrics, created_at, anonymized_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                lesson_id, agent_id, lesson_type.value, domain, title, content,
                json.dumps(strategy_steps), effectiveness_score, usage_context,
                json.dumps(success_metrics), lesson.created_at.isoformat(),
                json.dumps(lesson.anonymized_data)
            ))
            conn.commit()
            conn.close()
            
            # Update agent stats
            if agent_id in self.agent_registry:
                self.agent_registry[agent_id]["learning_stats"]["lessons_contributed"] += 1
            
            logger.info(f"Lesson '{title}' contributed by agent {agent_id}")
            return lesson_id
            
        except Exception as e:
            logger.error(f"Error contributing lesson: {e}")
            return ""
    
    def _anonymize_lesson_data(self, agent_id: str, content: str) -> Dict[str, Any]:
        """Anonymize lesson data for federated learning"""
        # Create anonymized hash
        anonymized_hash = hashlib.sha256(f"{agent_id}_{content}".encode()).hexdigest()[:16]
        
        return {
            "anonymized_hash": anonymized_hash,
            "content_length": len(content),
            "word_count": len(content.split()),
            "complexity_score": min(len(content.split()) / 100, 1.0),
            "domain_keywords": self._extract_domain_keywords(content)
        }
    
    def _extract_domain_keywords(self, content: str) -> List[str]:
        """Extract domain-specific keywords from content"""
        # Simple keyword extraction (could be enhanced with NLP)
        common_programming_keywords = [
            "function", "variable", "loop", "condition", "algorithm", "data structure",
            "optimization", "debugging", "testing", "refactoring"
        ]
        
        common_math_keywords = [
            "equation", "formula", "theorem", "proof", "calculation", "geometry",
            "algebra", "calculus", "statistics", "probability"
        ]
        
        common_writing_keywords = [
            "structure", "paragraph", "thesis", "argument", "evidence", "style",
            "grammar", "vocabulary", "narrative", "persuasion"
        ]
        
        all_keywords = common_programming_keywords + common_math_keywords + common_writing_keywords
        
        found_keywords = []
        content_lower = content.lower()
        for keyword in all_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords[:10]  # Return top 10 keywords
    
    def find_relevant_lessons(self, agent_id: str, query_domain: str, 
                             lesson_type: Optional[LessonType] = None) -> List[AgentLesson]:
        """Find relevant lessons for an agent based on domain and type"""
        try:
            relevant_lessons = []
            
            for lesson in self.lesson_library.values():
                # Skip own lessons
                if lesson.agent_id == agent_id:
                    continue
                
                # Check domain match
                if lesson.domain.lower() == query_domain.lower():
                    if lesson_type is None or lesson.lesson_type == lesson_type:
                        relevant_lessons.append(lesson)
            
            # Sort by effectiveness score
            relevant_lessons.sort(key=lambda x: x.effectiveness_score, reverse=True)
            
            return relevant_lessons[:10]  # Return top 10 lessons
            
        except Exception as e:
            logger.error(f"Error finding relevant lessons: {e}")
            return []
    
    def adopt_lesson(self, agent_id: str, lesson_id: str) -> bool:
        """Agent adopts a lesson from a peer"""
        try:
            if lesson_id not in self.lesson_library:
                return False
            
            lesson = self.lesson_library[lesson_id]
            lesson.adoption_count += 1
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE agent_lessons SET adoption_count = adoption_count + 1 
                WHERE lesson_id = ?
            ''', (lesson_id,))
            conn.commit()
            conn.close()
            
            # Update agent stats
            if agent_id in self.agent_registry:
                self.agent_registry[agent_id]["learning_stats"]["lessons_adopted"] += 1
            
            logger.info(f"Agent {agent_id} adopted lesson {lesson_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adopting lesson: {e}")
            return False
    
    def contribute_federated_knowledge(self, agent_id: str, 
                                     specialization: AgentSpecialization,
                                     query_type: str, toolkit_used: str,
                                     approach_summary: str, 
                                     performance_metrics: Dict[str, float],
                                     lessons_learned: List[str],
                                     optimization_tips: List[str],
                                     error_patterns: List[str]) -> str:
        """Contribute anonymized knowledge for federated learning"""
        try:
            contribution_id = str(uuid.uuid4())
            
            # Create anonymized hash
            anonymized_hash = hashlib.sha256(f"{agent_id}_{query_type}_{toolkit_used}".encode()).hexdigest()[:16]
            
            contribution = KnowledgeContribution(
                contribution_id=contribution_id,
                agent_specialization=specialization,
                query_type=query_type,
                toolkit_used=toolkit_used,
                approach_summary=approach_summary,
                performance_metrics=performance_metrics,
                lessons_learned=lessons_learned,
                optimization_tips=optimization_tips,
                error_patterns=error_patterns,
                timestamp=datetime.now(),
                anonymized_hash=anonymized_hash
            )
            
            self.knowledge_compendium[specialization.value].append(contribution)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO knowledge_contributions 
                (contribution_id, agent_specialization, query_type, toolkit_used, 
                 approach_summary, performance_metrics, lessons_learned, optimization_tips, 
                 error_patterns, timestamp, anonymized_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                contribution_id, specialization.value, query_type, toolkit_used,
                approach_summary, json.dumps(performance_metrics), 
                json.dumps(lessons_learned), json.dumps(optimization_tips),
                json.dumps(error_patterns), contribution.timestamp.isoformat(),
                anonymized_hash
            ))
            conn.commit()
            conn.close()
            
            logger.info(f"Federated knowledge contributed by agent {agent_id}")
            return contribution_id
            
        except Exception as e:
            logger.error(f"Error contributing federated knowledge: {e}")
            return ""
    
    def get_federated_knowledge(self, specialization: AgentSpecialization, 
                              query_type: str) -> List[KnowledgeContribution]:
        """Get relevant federated knowledge for a specialization and query type"""
        try:
            relevant_contributions = []
            
            for contribution in self.knowledge_compendium[specialization.value]:
                if contribution.query_type.lower() == query_type.lower():
                    relevant_contributions.append(contribution)
            
            # Sort by performance metrics (assuming higher is better)
            relevant_contributions.sort(
                key=lambda x: sum(x.performance_metrics.values()) / len(x.performance_metrics),
                reverse=True
            )
            
            return relevant_contributions[:5]  # Return top 5 contributions
            
        except Exception as e:
            logger.error(f"Error getting federated knowledge: {e}")
            return []
    
    async def start_collaborative_session(self, initiator_agent: str, 
                                        task_description: str, 
                                        session_type: str,
                                        required_specializations: List[AgentSpecialization]) -> str:
        """Start a collaborative session with multiple agents"""
        try:
            session_id = str(uuid.uuid4())
            
            # Find available agents with required specializations
            participating_agents = [initiator_agent]
            for specialization in required_specializations:
                available_agents = self.specialist_agents[specialization]
                if available_agents:
                    # Select best agent based on peer rating
                    best_agent = max(available_agents, 
                                   key=lambda x: self.agent_registry.get(x, {}).get("learning_stats", {}).get("peer_rating", 0))
                    if best_agent not in participating_agents:
                        participating_agents.append(best_agent)
            
            session = CollaborativeSession(
                session_id=session_id,
                initiator_agent=initiator_agent,
                participating_agents=participating_agents,
                task_description=task_description,
                session_type=session_type,
                agent_contributions=[]
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Collaborative session {session_id} started with {len(participating_agents)} agents")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting collaborative session: {e}")
            return ""
    
    async def multi_agent_debate(self, session_id: str, question: str, 
                               consensus_method: ConsensusMethod = ConsensusMethod.MAJORITY_VOTE) -> Dict[str, Any]:
        """Conduct multi-agent debate and reach consensus"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError("Session not found")
            
            session = self.active_sessions[session_id]
            agent_positions = []
            
            # Collect positions from all agents
            for agent_id in session.participating_agents:
                agent_data = self.agent_registry.get(agent_id, {})
                specialization = agent_data.get("specialization", AgentSpecialization.ANALYZER)
                
                # Simulate agent analysis (in real implementation, would call actual agent)
                position = self._simulate_agent_position(agent_id, specialization, question)
                agent_positions.append(position)
            
            # Apply consensus method
            consensus_result = self._apply_consensus_method(agent_positions, consensus_method)
            
            session.consensus_result = consensus_result["final_answer"]
            session.consensus_method = consensus_method
            session.agent_contributions.extend([
                {
                    "agent_id": pos.agent_id,
                    "position": pos.position,
                    "confidence": pos.confidence,
                    "reasoning": pos.reasoning
                }
                for pos in agent_positions
            ])
            
            # Generate lessons from debate
            lessons = self._generate_lessons_from_debate(agent_positions, consensus_result)
            session.lessons_generated.extend(lessons)
            
            return {
                "session_id": session_id,
                "consensus_result": consensus_result,
                "agent_positions": [
                    {
                        "agent_id": pos.agent_id,
                        "specialization": pos.agent_specialization.value,
                        "position": pos.position,
                        "confidence": pos.confidence,
                        "reasoning": pos.reasoning
                    }
                    for pos in agent_positions
                ],
                "lessons_generated": lessons
            }
            
        except Exception as e:
            logger.error(f"Error in multi-agent debate: {e}")
            return {"error": str(e)}
    
    def _simulate_agent_position(self, agent_id: str, specialization: AgentSpecialization, 
                               question: str) -> AgentDebatePosition:
        """Simulate an agent's position in a debate (placeholder for actual agent call)"""
        # This would be replaced with actual agent inference
        positions = {
            AgentSpecialization.MATH_SOLVER: f"Mathematical approach: {question[:50]}...",
            AgentSpecialization.WRITING_ASSISTANT: f"Writing perspective: {question[:50]}...",
            AgentSpecialization.CODE_HELPER: f"Programming solution: {question[:50]}...",
            AgentSpecialization.ANALYZER: f"Analytical view: {question[:50]}..."
        }
        
        return AgentDebatePosition(
            agent_id=agent_id,
            agent_specialization=specialization,
            position=positions.get(specialization, f"General approach: {question[:50]}..."),
            reasoning=f"Based on {specialization.value} expertise",
            confidence=0.8,
            supporting_evidence=[f"Evidence from {specialization.value} domain"],
            counter_arguments=[f"Potential issues from {specialization.value} perspective"]
        )
    
    def _apply_consensus_method(self, positions: List[AgentDebatePosition], 
                              method: ConsensusMethod) -> Dict[str, Any]:
        """Apply consensus method to reach final decision"""
        try:
            if method == ConsensusMethod.MAJORITY_VOTE:
                # Simple majority vote based on position similarity
                position_groups = defaultdict(list)
                for pos in positions:
                    position_groups[pos.position[:20]].append(pos)
                
                majority_group = max(position_groups.values(), key=len)
                return {
                    "final_answer": majority_group[0].position,
                    "confidence": sum(pos.confidence for pos in majority_group) / len(majority_group),
                    "method": "majority_vote",
                    "supporting_agents": [pos.agent_id for pos in majority_group]
                }
            
            elif method == ConsensusMethod.CONFIDENCE_WEIGHTED:
                # Weight by confidence scores
                total_weighted_confidence = sum(pos.confidence for pos in positions)
                if total_weighted_confidence > 0:
                    best_position = max(positions, key=lambda x: x.confidence)
                    return {
                        "final_answer": best_position.position,
                        "confidence": best_position.confidence,
                        "method": "confidence_weighted",
                        "supporting_agents": [best_position.agent_id]
                    }
            
            # Default fallback
            return {
                "final_answer": positions[0].position if positions else "No consensus reached",
                "confidence": 0.5,
                "method": method.value,
                "supporting_agents": [pos.agent_id for pos in positions]
            }
            
        except Exception as e:
            logger.error(f"Error applying consensus method: {e}")
            return {"final_answer": "Error in consensus", "confidence": 0.0, "method": method.value}
    
    def _generate_lessons_from_debate(self, positions: List[AgentDebatePosition], 
                                    consensus_result: Dict[str, Any]) -> List[str]:
        """Generate lessons from multi-agent debate"""
        lessons = []
        
        # Lesson about consensus building
        if len(positions) > 1:
            lessons.append(f"Consensus Method: {consensus_result['method']} achieved {consensus_result['confidence']:.2f} confidence")
        
        # Lessons about different perspectives
        specializations = [pos.agent_specialization.value for pos in positions]
        if len(set(specializations)) > 1:
            lessons.append(f"Multi-perspective analysis: {', '.join(set(specializations))} provided diverse viewpoints")
        
        # Lesson about confidence patterns
        avg_confidence = sum(pos.confidence for pos in positions) / len(positions)
        lessons.append(f"Average agent confidence: {avg_confidence:.2f} - {'High' if avg_confidence > 0.7 else 'Moderate' if avg_confidence > 0.5 else 'Low'} certainty")
        
        return lessons
    
    async def cross_correct_agents(self, session_id: str, primary_agent: str, 
                                 secondary_agent: str, content: str) -> Dict[str, Any]:
        """Enable cross-correction between specialized agents"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError("Session not found")
            
            session = self.active_sessions[session_id]
            
            primary_data = self.agent_registry.get(primary_agent, {})
            secondary_data = self.agent_registry.get(secondary_agent, {})
            
            primary_spec = primary_data.get("specialization", AgentSpecialization.ANALYZER)
            secondary_spec = secondary_data.get("specialization", AgentSpecialization.ANALYZER)
            
            # Simulate cross-correction (would call actual agents)
            corrections = {
                "primary_agent": primary_agent,
                "secondary_agent": secondary_agent,
                "primary_specialization": primary_spec.value,
                "secondary_specialization": secondary_spec.value,
                "original_content": content,
                "corrections": self._simulate_cross_corrections(primary_spec, secondary_spec, content),
                "improvement_score": 0.85,
                "timestamp": datetime.now().isoformat()
            }
            
            session.cross_corrections.append(corrections)
            
            return corrections
            
        except Exception as e:
            logger.error(f"Error in cross-correction: {e}")
            return {"error": str(e)}
    
    def _simulate_cross_corrections(self, primary_spec: AgentSpecialization, 
                                  secondary_spec: AgentSpecialization, 
                                  content: str) -> List[Dict[str, str]]:
        """Simulate cross-corrections between agents"""
        corrections = []
        
        if primary_spec == AgentSpecialization.CODE_HELPER and secondary_spec == AgentSpecialization.GRAMMAR_CHECKER:
            corrections.append({
                "type": "grammar_check",
                "suggestion": "Improved variable naming and comment clarity",
                "from_agent": secondary_spec.value
            })
        
        elif primary_spec == AgentSpecialization.WRITING_ASSISTANT and secondary_spec == AgentSpecialization.MATH_SOLVER:
            corrections.append({
                "type": "mathematical_accuracy",
                "suggestion": "Verified mathematical calculations and formulas",
                "from_agent": secondary_spec.value
            })
        
        corrections.append({
            "type": "general_improvement",
            "suggestion": f"Cross-validated content from {secondary_spec.value} perspective",
            "from_agent": secondary_spec.value
        })
        
        return corrections
    
    def get_peer_teaching_stats(self) -> Dict[str, Any]:
        """Get comprehensive peer teaching system statistics"""
        try:
            stats = {
                "total_agents": len(self.agent_registry),
                "total_lessons": len(self.lesson_library),
                "active_sessions": len(self.active_sessions),
                "knowledge_contributions": sum(len(contributions) for contributions in self.knowledge_compendium.values()),
                "specialization_distribution": {
                    spec.value: len(agents) for spec, agents in self.specialist_agents.items()
                },
                "lesson_adoption_rate": self._calculate_lesson_adoption_rate(),
                "collaboration_success_rate": self._calculate_collaboration_success_rate(),
                "top_lesson_contributors": self._get_top_lesson_contributors(),
                "federated_knowledge_summary": self._get_federated_knowledge_summary()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting peer teaching stats: {e}")
            return {"error": str(e)}
    
    def _calculate_lesson_adoption_rate(self) -> float:
        """Calculate lesson adoption rate"""
        if not self.lesson_library:
            return 0.0
        
        total_lessons = len(self.lesson_library)
        adopted_lessons = sum(1 for lesson in self.lesson_library.values() if lesson.adoption_count > 0)
        
        return adopted_lessons / total_lessons if total_lessons > 0 else 0.0
    
    def _calculate_collaboration_success_rate(self) -> float:
        """Calculate collaboration success rate"""
        if not self.active_sessions:
            return 0.0
        
        successful_sessions = sum(1 for session in self.active_sessions.values() if session.consensus_result)
        
        return successful_sessions / len(self.active_sessions) if self.active_sessions else 0.0
    
    def _get_top_lesson_contributors(self) -> List[Dict[str, Any]]:
        """Get top lesson contributors"""
        contributor_stats = defaultdict(int)
        
        for lesson in self.lesson_library.values():
            contributor_stats[lesson.agent_id] += 1
        
        top_contributors = sorted(contributor_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [
            {
                "agent_id": agent_id,
                "agent_name": self.agent_registry.get(agent_id, {}).get("agent_name", "Unknown"),
                "lessons_contributed": count
            }
            for agent_id, count in top_contributors
        ]
    
    def _get_federated_knowledge_summary(self) -> Dict[str, Any]:
        """Get federated knowledge summary"""
        summary = {}
        
        for specialization, contributions in self.knowledge_compendium.items():
            summary[specialization] = {
                "total_contributions": len(contributions),
                "avg_performance": sum(
                    sum(contrib.performance_metrics.values()) / len(contrib.performance_metrics)
                    for contrib in contributions
                ) / len(contributions) if contributions else 0.0,
                "common_toolkits": list(set(contrib.toolkit_used for contrib in contributions))[:5]
            }
        
        return summary

# Global instance
peer_teaching_system = None

def get_peer_teaching_system():
    """Get or create global peer teaching system instance"""
    global peer_teaching_system
    if peer_teaching_system is None:
        peer_teaching_system = PeerTeachingSystem()
    return peer_teaching_system

async def demo_peer_teaching():
    """Demo function to show peer teaching capabilities"""
    system = get_peer_teaching_system()
    
    # Register demo agents
    await system.register_agent("math_agent_1", "MathSolver Pro", AgentSpecialization.MATH_SOLVER, 
                               ["algebra", "calculus", "statistics"])
    await system.register_agent("code_agent_1", "CodeHelper AI", AgentSpecialization.CODE_HELPER, 
                               ["python", "javascript", "debugging"])
    await system.register_agent("writing_agent_1", "WritingAssistant", AgentSpecialization.WRITING_ASSISTANT, 
                               ["grammar", "style", "structure"])
    
    # Demonstrate lesson sharing
    lesson_id = system.contribute_lesson(
        "math_agent_1", LessonType.STRATEGY, "algebra", 
        "Quadratic Equation Solving", "Use factoring first, then quadratic formula",
        ["Factor if possible", "Apply quadratic formula", "Check solutions"],
        0.92, "High school algebra problems", {"success_rate": 0.95, "time_efficiency": 0.88}
    )
    
    # Demonstrate lesson adoption
    relevant_lessons = system.find_relevant_lessons("code_agent_1", "algebra", LessonType.STRATEGY)
    print(f"Found {len(relevant_lessons)} relevant lessons")
    
    # Demonstrate collaborative session
    session_id = await system.start_collaborative_session(
        "math_agent_1", "Solve complex optimization problem",
        "problem_solving", [AgentSpecialization.MATH_SOLVER, AgentSpecialization.CODE_HELPER]
    )
    
    # Demonstrate multi-agent debate
    debate_result = await system.multi_agent_debate(
        session_id, "What's the best approach to solve this optimization problem?",
        ConsensusMethod.CONFIDENCE_WEIGHTED
    )
    
    print(f"Debate result: {debate_result}")
    
    # Get system stats
    stats = system.get_peer_teaching_stats()
    print(f"Peer teaching stats: {stats}")

if __name__ == "__main__":
    asyncio.run(demo_peer_teaching())