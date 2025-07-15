"""
Personal AI Router with Hybrid Edge-Cloud Routing
Implements intelligent routing between local Ollama models and cloud LLMs
Enhanced with decentralized P2P AI routing capabilities for web4ai network
"""

import os
import json
import logging
import asyncio
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
import sqlite3
import threading
from contextlib import contextmanager
import socket
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import P2P networking capabilities
try:
    from p2p_ai_router import (
        P2PNetworkManager, AgentCapability, NodeStatus, MessageType,
        get_p2p_network_manager
    )
    P2P_AVAILABLE = True
    logger.info("P2P AI Router available - enabling web4ai network integration")
except ImportError:
    P2P_AVAILABLE = False
    logger.warning("P2P AI Router not available, running in standalone mode")

class QueryComplexity(Enum):
    """Query complexity levels"""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class QueryIntent(Enum):
    """Query intent categories"""
    PERSONAL_MEMO = "personal_memo"
    QUICK_FACT = "quick_fact"
    CASUAL_CHAT = "casual_chat"
    CREATIVE_WRITING = "creative_writing"
    TECHNICAL_HELP = "technical_help"
    MATH_PROBLEM = "math_problem"
    RESEARCH = "research"
    CODING = "coding"
    ANALYSIS = "analysis"
    WORKFLOW = "workflow"

class RoutingDecision(Enum):
    """Routing decision types"""
    LOCAL_ONLY = "local_only"
    CLOUD_ONLY = "cloud_only"
    HYBRID = "hybrid"
    CACHED = "cached"
    FAILED_OVER = "failed_over"
    P2P_NETWORK = "p2p_network"
    DISTRIBUTED_MESH = "distributed_mesh"

@dataclass
class QueryAnalysis:
    """Query analysis result"""
    text: str
    intent: QueryIntent
    complexity: QueryComplexity
    privacy_score: float  # 0-1, higher = more private
    confidence: float
    estimated_tokens: int
    requires_real_time: bool = False
    requires_creativity: bool = False
    requires_logic: bool = False
    personal_context: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RoutingResult:
    """Routing decision result"""
    decision: RoutingDecision
    model_used: str
    response: str
    confidence: float
    latency: float
    cost_estimate: float
    reasoning: str
    cached: bool = False
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class OllamaClient:
    """Local Ollama client for on-device inference"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama is running and get available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                self.available_models = [model["name"] for model in models]
                logger.info(f"Ollama connected. Available models: {self.available_models}")
            else:
                logger.warning("Ollama not accessible")
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {e}")
    
    async def generate(self, model: str, prompt: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Generate response using local Ollama model"""
        try:
            if model not in self.available_models:
                logger.error(f"Model {model} not available in Ollama")
                return {"error": f"Model {model} not available"}
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            }
            
            if context:
                payload["context"] = context
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result.get("response", ""),
                    "model": model,
                    "done": result.get("done", False),
                    "context": result.get("context", []),
                    "eval_count": result.get("eval_count", 0),
                    "eval_duration": result.get("eval_duration", 0)
                }
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            return {"error": str(e)}

class PersonalMemoryStore:
    """Personal memory and preferences store"""
    
    def __init__(self, db_path: str = "personal_memory.db"):
        self.db_path = db_path
        self._init_database()
        self.lock = threading.Lock()
    
    def _init_database(self):
        """Initialize personal memory database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    context TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    preference TEXT NOT NULL,
                    value TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model_used TEXT,
                    routing_decision TEXT,
                    satisfaction_score REAL,
                    feedback TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper locking"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def store_memory(self, category: str, key: str, value: str, context: Optional[str] = None):
        """Store a personal memory"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO memories 
                (category, key, value, context, created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
            ''', (category, key, value, context))
            conn.commit()
    
    def get_memory(self, category: str, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a personal memory"""
        with self.get_connection() as conn:
            # Update access count
            conn.execute('''
                UPDATE memories 
                SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE category = ? AND key = ?
            ''', (category, key))
            
            cursor = conn.execute('''
                SELECT * FROM memories 
                WHERE category = ? AND key = ?
            ''', (category, key))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def search_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by query"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM memories 
                WHERE key LIKE ? OR value LIKE ? OR context LIKE ?
                ORDER BY access_count DESC, confidence DESC
                LIMIT ?
            ''', (f"%{query}%", f"%{query}%", f"%{query}%", limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def store_preference(self, category: str, preference: str, value: str, strength: float = 1.0):
        """Store a user preference"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO preferences 
                (category, preference, value, strength, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (category, preference, value, strength))
            conn.commit()
    
    def get_preferences(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get user preferences"""
        with self.get_connection() as conn:
            if category:
                cursor = conn.execute('''
                    SELECT * FROM preferences 
                    WHERE category = ?
                    ORDER BY strength DESC
                ''', (category,))
            else:
                cursor = conn.execute('''
                    SELECT * FROM preferences 
                    ORDER BY category, strength DESC
                ''')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def record_interaction(self, query: str, response: str, model_used: str, 
                          routing_decision: str, satisfaction_score: Optional[float] = None,
                          feedback: Optional[str] = None):
        """Record an interaction for learning"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO interactions 
                (query, response, model_used, routing_decision, satisfaction_score, feedback)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (query, response, model_used, routing_decision, satisfaction_score, feedback))
            conn.commit()

class QueryComplexityAnalyzer:
    """Analyzes query complexity and intent"""
    
    def __init__(self):
        self.complexity_keywords = {
            QueryComplexity.TRIVIAL: ["hi", "hello", "thanks", "yes", "no", "ok"],
            QueryComplexity.SIMPLE: ["what", "when", "where", "who", "weather", "time"],
            QueryComplexity.MODERATE: ["how", "why", "explain", "compare", "difference"],
            QueryComplexity.COMPLEX: ["analyze", "evaluate", "synthesize", "create", "design"],
            QueryComplexity.EXPERT: ["optimize", "architecture", "algorithm", "implement", "research"]
        }
        
        self.intent_patterns = {
            QueryIntent.PERSONAL_MEMO: ["remember", "note", "memo", "remind me"],
            QueryIntent.QUICK_FACT: ["what is", "define", "fact", "lookup"],
            QueryIntent.CASUAL_CHAT: ["how are you", "chat", "talk", "conversation"],
            QueryIntent.CREATIVE_WRITING: ["write", "story", "poem", "creative"],
            QueryIntent.TECHNICAL_HELP: ["help", "troubleshoot", "fix", "error"],
            QueryIntent.MATH_PROBLEM: ["calculate", "solve", "math", "equation"],
            QueryIntent.RESEARCH: ["research", "study", "investigate", "explore"],
            QueryIntent.CODING: ["code", "program", "debug", "function"],
            QueryIntent.ANALYSIS: ["analyze", "examine", "review", "assess"],
            QueryIntent.WORKFLOW: ["schedule", "plan", "organize", "workflow"]
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query complexity and intent"""
        query_lower = query.lower()
        
        # Analyze complexity
        complexity_scores = {}
        for complexity, keywords in self.complexity_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            complexity_scores[complexity] = score
        
        # Determine primary complexity
        primary_complexity = max(complexity_scores, key=complexity_scores.get)
        if complexity_scores[primary_complexity] == 0:
            primary_complexity = QueryComplexity.MODERATE  # Default
        
        # Analyze intent
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            intent_scores[intent] = score
        
        # Determine primary intent
        primary_intent = max(intent_scores, key=intent_scores.get)
        if intent_scores[primary_intent] == 0:
            primary_intent = QueryIntent.QUICK_FACT  # Default
        
        # Calculate privacy score
        privacy_indicators = ["personal", "private", "my", "i", "me", "secret"]
        privacy_score = min(1.0, sum(1 for indicator in privacy_indicators if indicator in query_lower) / 3)
        
        # Estimate tokens
        estimated_tokens = len(query.split()) * 1.3  # Rough estimate
        
        # Determine special requirements
        requires_real_time = any(word in query_lower for word in ["current", "now", "today", "latest"])
        requires_creativity = any(word in query_lower for word in ["create", "write", "story", "poem"])
        requires_logic = any(word in query_lower for word in ["solve", "calculate", "logic", "reason"])
        
        return QueryAnalysis(
            text=query,
            intent=primary_intent,
            complexity=primary_complexity,
            privacy_score=privacy_score,
            confidence=0.8,  # Base confidence
            estimated_tokens=int(estimated_tokens),
            requires_real_time=requires_real_time,
            requires_creativity=requires_creativity,
            requires_logic=requires_logic,
            personal_context=[],
            metadata={}
        )

class AdaptiveCache:
    """Adaptive caching system with learning capabilities"""
    
    def __init__(self, db_path: str = "adaptive_cache.db"):
        self.db_path = db_path
        self._init_database()
        self.lock = threading.Lock()
    
    def _init_database(self):
        """Initialize adaptive cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE NOT NULL,
                    query_text TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model_used TEXT,
                    confidence REAL,
                    hit_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            ''')
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper locking"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response for query"""
        import hashlib
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM cache_entries 
                WHERE query_hash = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            ''', (query_hash,))
            
            row = cursor.fetchone()
            if row:
                # Update access statistics
                conn.execute('''
                    UPDATE cache_entries 
                    SET hit_count = hit_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (row['id'],))
                conn.commit()
                
                return dict(row)
            return None
    
    def cache_response(self, query: str, response: str, model_used: str, 
                      confidence: float, ttl_hours: int = 24):
        """Cache a response"""
        import hashlib
        from datetime import timedelta
        
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO cache_entries 
                (query_hash, query_text, response, model_used, confidence, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (query_hash, query, response, model_used, confidence, expires_at))
            conn.commit()

class PersonalAIRouter:
    """Main Personal AI Router with hybrid edge-cloud capabilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ollama_client = OllamaClient()
        self.memory_store = PersonalMemoryStore()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.adaptive_cache = AdaptiveCache()
        
        # Initialize P2P network capabilities
        self.p2p_network = None
        self.web4ai_node_id = None
        self.p2p_capabilities = [
            AgentCapability.TEXT_GENERATION,
            AgentCapability.QUESTION_ANSWERING,
            AgentCapability.SUMMARIZATION,
            AgentCapability.CREATIVE_WRITING
        ]
        
        # Initialize P2P network if available
        if P2P_AVAILABLE:
            self._init_p2p_network()
        
        # Default models
        self.local_models = {
            "simple": "llama3.2:3b",
            "moderate": "llama3.1:8b", 
            "complex": "llama3.1:70b"
        }
        
        # Routing thresholds (enhanced with P2P options)
        self.complexity_thresholds = {
            "local_only": [QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE],
            "hybrid": [QueryComplexity.MODERATE],
            "cloud_preferred": [QueryComplexity.COMPLEX, QueryComplexity.EXPERT],
            "p2p_preferred": [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
        }
        
        # Privacy thresholds
        self.privacy_threshold = 0.7  # Above this, prefer local
        
        logger.info("Personal AI Router initialized")
    
    def _init_p2p_network(self):
        """Initialize P2P network for web4ai"""
        try:
            # Generate unique node ID for web4ai network
            self.web4ai_node_id = f"web4ai_personal_{uuid.uuid4().hex[:8]}"
            
            # Find available port for P2P network
            p2p_port = self._find_available_port(9000)
            
            # Initialize P2P network manager
            self.p2p_network = P2PNetworkManager(
                node_id=self.web4ai_node_id,
                port=p2p_port,
                capabilities=self.p2p_capabilities
            )
            
            # Start P2P network
            self.p2p_network.start_network()
            
            # Try to join web4ai network with default bootstrap nodes
            bootstrap_nodes = [
                ("127.0.0.1", 9001),
                ("127.0.0.1", 9002),
                ("127.0.0.1", 9003)
            ]
            
            # Join the network
            self.p2p_network.join_network(bootstrap_nodes)
            
            logger.info(f"P2P network initialized for web4ai with node ID: {self.web4ai_node_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize P2P network: {e}")
            self.p2p_network = None
    
    def _find_available_port(self, start_port: int) -> int:
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + 100):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', port))
                sock.close()
                return port
            except OSError:
                continue
        
        raise Exception("No available ports found")
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> RoutingResult:
        """Process a query with hybrid routing"""
        start_time = datetime.now()
        
        # Step 1: Analyze query
        analysis = self.complexity_analyzer.analyze_query(query)
        
        # Step 2: Check cache first
        cached_response = self.adaptive_cache.get_cached_response(query)
        if cached_response:
            return RoutingResult(
                decision=RoutingDecision.CACHED,
                model_used=cached_response['model_used'],
                response=cached_response['response'],
                confidence=cached_response['confidence'],
                latency=0.1,
                cost_estimate=0.0,
                reasoning="Retrieved from adaptive cache",
                cached=True
            )
        
        # Step 3: Enhance with personal context
        personal_context = self._get_personal_context(query, analysis)
        enhanced_query = self._enhance_with_context(query, personal_context)
        
        # Step 4: Make routing decision
        routing_decision = self._decide_routing(analysis)
        
        # Step 5: Execute based on routing decision
        if routing_decision == RoutingDecision.LOCAL_ONLY:
            result = await self._process_local(enhanced_query, analysis)
        elif routing_decision == RoutingDecision.CLOUD_ONLY:
            result = await self._process_cloud(enhanced_query, analysis)
        elif routing_decision == RoutingDecision.HYBRID:
            result = await self._process_hybrid(enhanced_query, analysis)
        elif routing_decision == RoutingDecision.P2P_NETWORK:
            result = await self._process_p2p_network(enhanced_query, analysis)
        elif routing_decision == RoutingDecision.DISTRIBUTED_MESH:
            result = await self._process_distributed_mesh(enhanced_query, analysis)
        else:
            result = await self._process_local(enhanced_query, analysis)  # Fallback
        
        # Step 6: Calculate metrics
        end_time = datetime.now()
        result.latency = (end_time - start_time).total_seconds()
        
        # Step 7: Cache successful responses
        if result.confidence > 0.7:
            self.adaptive_cache.cache_response(
                query, result.response, result.model_used, result.confidence
            )
        
        # Step 8: Record interaction
        self.memory_store.record_interaction(
            query, result.response, result.model_used, result.decision.value
        )
        
        return result
    
    def _get_personal_context(self, query: str, analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """Get relevant personal context for the query"""
        # Search memories for relevant context
        memories = self.memory_store.search_memories(query, limit=5)
        
        # Get relevant preferences
        preferences = self.memory_store.get_preferences()
        
        return {
            "memories": memories,
            "preferences": preferences,
            "analysis": analysis
        }
    
    def _enhance_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """Enhance query with personal context"""
        if not context.get("memories") and not context.get("preferences"):
            return query
        
        enhanced = query
        
        # Add relevant memories
        if context.get("memories"):
            memory_context = "\n".join([
                f"Memory: {mem['key']} - {mem['value']}"
                for mem in context["memories"][:3]
            ])
            enhanced = f"Personal Context:\n{memory_context}\n\nQuery: {query}"
        
        # Add preferences
        if context.get("preferences"):
            prefs = [p for p in context["preferences"] if p["strength"] > 0.7][:2]
            if prefs:
                pref_context = "\n".join([
                    f"Preference: {pref['preference']} - {pref['value']}"
                    for pref in prefs
                ])
                enhanced = f"{enhanced}\n\nPreferences:\n{pref_context}"
        
        return enhanced
    
    def _decide_routing(self, analysis: QueryAnalysis) -> RoutingDecision:
        """Decide routing strategy based on analysis"""
        # Privacy-first routing
        if analysis.privacy_score > self.privacy_threshold:
            return RoutingDecision.LOCAL_ONLY
        
        # Check P2P network availability first
        if P2P_AVAILABLE and self.p2p_network and analysis.complexity in self.complexity_thresholds["p2p_preferred"]:
            # Check if P2P network has suitable agents
            if self._has_p2p_agents_for_intent(analysis.intent):
                return RoutingDecision.P2P_NETWORK
        
        # Complexity-based routing
        if analysis.complexity in self.complexity_thresholds["local_only"]:
            return RoutingDecision.LOCAL_ONLY
        elif analysis.complexity in self.complexity_thresholds["cloud_preferred"]:
            return RoutingDecision.CLOUD_ONLY
        else:
            return RoutingDecision.HYBRID
    
    async def _process_local(self, query: str, analysis: QueryAnalysis) -> RoutingResult:
        """Process query using local Ollama model"""
        # Select appropriate local model
        model = self.local_models.get("simple", "llama3.2:3b")
        if analysis.complexity == QueryComplexity.MODERATE:
            model = self.local_models.get("moderate", "llama3.1:8b")
        elif analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            model = self.local_models.get("complex", "llama3.1:70b")
        
        # Generate response
        result = await self.ollama_client.generate(model, query)
        
        if "error" in result:
            return RoutingResult(
                decision=RoutingDecision.FAILED_OVER,
                model_used=model,
                response=f"Local model error: {result['error']}",
                confidence=0.0,
                latency=0.0,
                cost_estimate=0.0,
                reasoning="Local model failed",
                fallback_used=True
            )
        
        return RoutingResult(
            decision=RoutingDecision.LOCAL_ONLY,
            model_used=model,
            response=result["response"],
            confidence=0.8,
            latency=0.0,
            cost_estimate=0.0,
            reasoning="Processed locally for privacy/simplicity"
        )
    
    async def _process_cloud(self, query: str, analysis: QueryAnalysis) -> RoutingResult:
        """Process query using cloud LLM (via ML router)"""
        # This would integrate with the existing ML router
        # For now, return a placeholder
        return RoutingResult(
            decision=RoutingDecision.CLOUD_ONLY,
            model_used="gpt-4o",
            response="Cloud processing would be implemented here",
            confidence=0.9,
            latency=1.5,
            cost_estimate=0.01,
            reasoning="Complex query requires cloud processing"
        )
    
    async def _process_hybrid(self, query: str, analysis: QueryAnalysis) -> RoutingResult:
        """Process query using hybrid approach"""
        # Try local first
        local_result = await self._process_local(query, analysis)
        
        # If local succeeds with high confidence, use it
        if local_result.confidence > 0.7:
            return local_result
        
        # Otherwise, fall back to cloud
        cloud_result = await self._process_cloud(query, analysis)
        cloud_result.decision = RoutingDecision.HYBRID
        cloud_result.reasoning = "Local attempt failed, used cloud fallback"
        cloud_result.fallback_used = True
        
        return cloud_result
    
    def _has_p2p_agents_for_intent(self, intent: QueryIntent) -> bool:
        """Check if P2P network has agents suitable for the query intent"""
        if not self.p2p_network:
            return False
        
        # Map query intents to agent capabilities
        intent_capability_map = {
            QueryIntent.CREATIVE_WRITING: AgentCapability.CREATIVE_WRITING,
            QueryIntent.TECHNICAL_HELP: AgentCapability.TECHNICAL_ANALYSIS,
            QueryIntent.CODING: AgentCapability.CODE_GENERATION,
            QueryIntent.RESEARCH: AgentCapability.RESEARCH,
            QueryIntent.ANALYSIS: AgentCapability.TECHNICAL_ANALYSIS,
            QueryIntent.QUICK_FACT: AgentCapability.QUESTION_ANSWERING,
            QueryIntent.CASUAL_CHAT: AgentCapability.TEXT_GENERATION,
            QueryIntent.MATH_PROBLEM: AgentCapability.TECHNICAL_ANALYSIS
        }
        
        required_capability = intent_capability_map.get(intent, AgentCapability.TEXT_GENERATION)
        
        # Check if discovery system has agents with required capability
        suitable_agents = self.p2p_network.discovery_system.get_agents_by_capability(required_capability)
        return len(suitable_agents) > 0
    
    async def _process_p2p_network(self, query: str, analysis: QueryAnalysis) -> RoutingResult:
        """Process query using P2P network"""
        try:
            # Map query intent to capability
            intent_capability_map = {
                QueryIntent.CREATIVE_WRITING: AgentCapability.CREATIVE_WRITING,
                QueryIntent.TECHNICAL_HELP: AgentCapability.TECHNICAL_ANALYSIS,
                QueryIntent.CODING: AgentCapability.CODE_GENERATION,
                QueryIntent.RESEARCH: AgentCapability.RESEARCH,
                QueryIntent.ANALYSIS: AgentCapability.TECHNICAL_ANALYSIS,
                QueryIntent.QUICK_FACT: AgentCapability.QUESTION_ANSWERING,
                QueryIntent.CASUAL_CHAT: AgentCapability.TEXT_GENERATION,
                QueryIntent.MATH_PROBLEM: AgentCapability.TECHNICAL_ANALYSIS
            }
            
            required_capability = intent_capability_map.get(analysis.intent, AgentCapability.TEXT_GENERATION)
            
            # Route query through P2P network
            response = self.p2p_network.route_query(query, required_capability)
            
            if response:
                return RoutingResult(
                    decision=RoutingDecision.P2P_NETWORK,
                    model_used=f"web4ai_peer_agent",
                    response=response,
                    confidence=0.85,
                    latency=0.0,
                    cost_estimate=0.0,
                    reasoning="Processed by web4ai P2P network agent"
                )
            else:
                # Fallback to local processing
                return await self._process_local(query, analysis)
                
        except Exception as e:
            logger.error(f"P2P network processing failed: {e}")
            # Fallback to local processing
            return await self._process_local(query, analysis)
    
    async def _process_distributed_mesh(self, query: str, analysis: QueryAnalysis) -> RoutingResult:
        """Process query using distributed mesh network"""
        try:
            # For complex queries, use distributed mesh approach
            # This would involve multiple P2P agents collaborating
            
            # Get network statistics
            network_stats = self.p2p_network.get_network_stats()
            
            if network_stats["peer_count"] < 2:
                # Not enough peers for distributed processing
                return await self._process_p2p_network(query, analysis)
            
            # Route to multiple agents for consensus
            responses = []
            capabilities = [AgentCapability.TEXT_GENERATION, AgentCapability.TECHNICAL_ANALYSIS]
            
            for capability in capabilities:
                response = self.p2p_network.route_query(query, capability)
                if response:
                    responses.append(response)
            
            if responses:
                # Combine responses (simple approach)
                combined_response = f"Distributed mesh response:\n" + "\n".join(responses)
                
                return RoutingResult(
                    decision=RoutingDecision.DISTRIBUTED_MESH,
                    model_used="web4ai_distributed_mesh",
                    response=combined_response,
                    confidence=0.90,
                    latency=0.0,
                    cost_estimate=0.0,
                    reasoning="Processed by web4ai distributed mesh network"
                )
            else:
                # Fallback to single P2P agent
                return await self._process_p2p_network(query, analysis)
                
        except Exception as e:
            logger.error(f"Distributed mesh processing failed: {e}")
            # Fallback to P2P network
            return await self._process_p2p_network(query, analysis)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        with self.memory_store.get_connection() as conn:
            cursor = conn.execute('''
                SELECT 
                    routing_decision,
                    COUNT(*) as count,
                    AVG(satisfaction_score) as avg_satisfaction
                FROM interactions 
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY routing_decision
            ''')
            
            routing_stats = {row['routing_decision']: {
                'count': row['count'],
                'avg_satisfaction': row['avg_satisfaction']
            } for row in cursor.fetchall()}
        
        # Get P2P network stats if available
        p2p_stats = {}
        if self.p2p_network:
            p2p_stats = self.p2p_network.get_network_stats()
        
        return {
            "routing_stats": routing_stats,
            "available_local_models": self.ollama_client.available_models,
            "total_memories": len(self.memory_store.search_memories("", limit=1000)),
            "cache_hit_rate": self._get_cache_hit_rate(),
            "p2p_network": {
                "enabled": P2P_AVAILABLE and self.p2p_network is not None,
                "node_id": self.web4ai_node_id,
                "network_stats": p2p_stats
            }
        }
    
    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        with self.adaptive_cache.get_connection() as conn:
            cursor = conn.execute('''
                SELECT AVG(hit_count) as avg_hits FROM cache_entries
            ''')
            result = cursor.fetchone()
            return result['avg_hits'] if result and result['avg_hits'] else 0.0

# Global instance
_personal_ai_router = None

def get_personal_ai_router(config: Dict[str, Any] = None) -> PersonalAIRouter:
    """Get or create global Personal AI Router instance"""
    global _personal_ai_router
    if _personal_ai_router is None:
        _personal_ai_router = PersonalAIRouter(config)
    return _personal_ai_router