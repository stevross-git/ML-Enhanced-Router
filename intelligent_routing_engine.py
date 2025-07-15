#!/usr/bin/env python3
"""
Intelligent Routing Engine with Advanced ML-based Agent Selection
Provides sophisticated routing decisions with performance optimization
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib

from advanced_ml_classifier import AdvancedMLClassifier, QueryAnalysis, QueryCategory, QueryIntent

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy options"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    PERFORMANCE_BASED = "performance_based"
    ML_OPTIMIZED = "ml_optimized"
    CONSENSUS_BASED = "consensus_based"
    HYBRID = "hybrid"


class AgentStatus(Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class AgentCapability:
    """Agent capability specification"""
    name: str
    proficiency: float  # 0.0 to 1.0
    max_concurrent: int
    response_time_avg: float
    success_rate: float
    specializations: List[str] = field(default_factory=list)
    
    
@dataclass
class AgentPerformanceMetrics:
    """Agent performance tracking"""
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_count: int = 0
    failure_count: int = 0
    total_requests: int = 0
    current_load: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    quality_score: float = 1.0
    user_satisfaction: float = 1.0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0
        
    @property
    def average_response_time(self) -> float:
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
        
    @property
    def load_factor(self) -> float:
        return min(self.current_load / 10.0, 1.0)  # Normalized load


@dataclass
class EnhancedAgent:
    """Enhanced agent with advanced capabilities"""
    id: str
    name: str
    endpoint: str
    capabilities: List[AgentCapability]
    max_concurrent: int = 10
    timeout: float = 30.0
    status: AgentStatus = AgentStatus.ACTIVE
    performance: AgentPerformanceMetrics = field(default_factory=AgentPerformanceMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced features
    specializations: List[QueryCategory] = field(default_factory=list)
    preferred_intents: List[QueryIntent] = field(default_factory=list)
    technical_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    cost_per_request: float = 0.0
    priority_level: int = 1
    geographic_region: str = "global"
    
    def can_handle_query(self, analysis: QueryAnalysis) -> Tuple[bool, float]:
        """Check if agent can handle query and return confidence score"""
        if self.status != AgentStatus.ACTIVE:
            return False, 0.0
            
        # Check specializations
        specialization_match = 0.0
        if self.specializations:
            if analysis.primary_category in self.specializations:
                specialization_match = 1.0
            elif any(cat in self.specializations for cat in analysis.categories):
                specialization_match = 0.7
                
        # Check intent compatibility
        intent_match = 0.0
        if self.preferred_intents:
            if analysis.intent in self.preferred_intents:
                intent_match = 1.0
            else:
                intent_match = 0.3
        else:
            intent_match = 0.8  # Default if no specific intents
            
        # Check technical level
        technical_match = 0.0
        if self.technical_levels:
            if analysis.technical_level in self.technical_levels:
                technical_match = 1.0
            else:
                technical_match = 0.5
        else:
            technical_match = 0.8
            
        # Check language support
        language_match = 0.0
        if self.languages:
            if analysis.language in self.languages:
                language_match = 1.0
            elif "en" in self.languages:  # English fallback
                language_match = 0.7
        else:
            language_match = 0.9  # Default if no language restrictions
            
        # Check capability requirements
        capability_match = 1.0
        for required_capability in analysis.required_capabilities:
            has_capability = any(
                required_capability in cap.name or 
                required_capability in cap.specializations 
                for cap in self.capabilities
            )
            if not has_capability:
                capability_match *= 0.5
                
        # Calculate overall confidence
        weights = {
            "specialization": 0.3,
            "intent": 0.2,
            "technical": 0.2,
            "language": 0.1,
            "capability": 0.2
        }
        
        confidence = (
            specialization_match * weights["specialization"] +
            intent_match * weights["intent"] +
            technical_match * weights["technical"] +
            language_match * weights["language"] +
            capability_match * weights["capability"]
        )
        
        # Apply performance penalties
        confidence *= self.performance.success_rate
        confidence *= (1.0 - self.performance.load_factor * 0.3)
        
        can_handle = confidence > 0.3 and self.performance.current_load < self.max_concurrent
        
        return can_handle, confidence


@dataclass
class RoutingDecision:
    """Routing decision with metadata"""
    selected_agents: List[EnhancedAgent]
    primary_agent: EnhancedAgent
    confidence: float
    strategy_used: RoutingStrategy
    reasoning: str
    fallback_agents: List[EnhancedAgent]
    estimated_response_time: float
    estimated_cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentRoutingEngine:
    """Advanced intelligent routing engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ml_classifier = AdvancedMLClassifier(config)
        self.agents: Dict[str, EnhancedAgent] = {}
        self.routing_history: deque = deque(maxlen=1000)
        self.performance_cache: Dict[str, Any] = {}
        self.load_balancer_state = {}
        self.circuit_breakers: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Advanced routing configuration
        self.routing_strategies = {
            RoutingStrategy.ROUND_ROBIN: self._route_round_robin,
            RoutingStrategy.WEIGHTED_ROUND_ROBIN: self._route_weighted_round_robin,
            RoutingStrategy.LEAST_CONNECTIONS: self._route_least_connections,
            RoutingStrategy.PERFORMANCE_BASED: self._route_performance_based,
            RoutingStrategy.ML_OPTIMIZED: self._route_ml_optimized,
            RoutingStrategy.CONSENSUS_BASED: self._route_consensus_based,
            RoutingStrategy.HYBRID: self._route_hybrid
        }
        
        self.default_strategy = RoutingStrategy.HYBRID
        
    async def initialize(self):
        """Initialize routing engine"""
        await self.ml_classifier.initialize()
        
        # Initialize circuit breakers
        for agent_id in self.agents:
            self.circuit_breakers[agent_id] = {
                "state": "closed",  # closed, open, half-open
                "failure_count": 0,
                "last_failure": None,
                "success_count": 0
            }
            
        logger.info("Intelligent routing engine initialized")
        
    async def register_agent(self, agent: EnhancedAgent):
        """Register a new agent"""
        self.agents[agent.id] = agent
        
        # Initialize circuit breaker
        self.circuit_breakers[agent.id] = {
            "state": "closed",
            "failure_count": 0,
            "last_failure": None,
            "success_count": 0
        }
        
        logger.info(f"Agent {agent.id} registered successfully")
        
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if agent_id in self.circuit_breakers:
                del self.circuit_breakers[agent_id]
            logger.info(f"Agent {agent_id} unregistered")
            
    async def route_query(self, query: str, context: Optional[Dict] = None, 
                         strategy: Optional[RoutingStrategy] = None) -> RoutingDecision:
        """Route query to appropriate agent(s)"""
        start_time = time.time()
        
        # Analyze query
        analysis = await self.ml_classifier.analyze(query, context)
        
        # Select routing strategy
        selected_strategy = strategy or self._select_optimal_strategy(analysis)
        
        # Apply routing strategy
        routing_function = self.routing_strategies[selected_strategy]
        decision = await routing_function(analysis, context)
        
        # Record routing decision
        routing_record = {
            "timestamp": datetime.now(),
            "query": query,
            "analysis": analysis,
            "decision": decision,
            "processing_time": time.time() - start_time
        }
        self.routing_history.append(routing_record)
        
        # Update performance cache
        self._update_performance_cache(decision, analysis)
        
        return decision
        
    def _select_optimal_strategy(self, analysis: QueryAnalysis) -> RoutingStrategy:
        """Select optimal routing strategy based on query analysis"""
        # Complex queries benefit from consensus
        if analysis.complexity > 0.8:
            return RoutingStrategy.CONSENSUS_BASED
            
        # High-priority queries need performance-based routing
        if analysis.priority > 7:
            return RoutingStrategy.PERFORMANCE_BASED
            
        # Multi-step queries benefit from ML optimization
        if analysis.multi_step:
            return RoutingStrategy.ML_OPTIMIZED
            
        # Creative queries can use round-robin
        if analysis.primary_category == QueryCategory.CREATIVE:
            return RoutingStrategy.WEIGHTED_ROUND_ROBIN
            
        # Default to hybrid approach
        return RoutingStrategy.HYBRID
        
    async def _route_round_robin(self, analysis: QueryAnalysis, context: Optional[Dict]) -> RoutingDecision:
        """Simple round-robin routing"""
        available_agents = [agent for agent in self.agents.values() 
                          if agent.status == AgentStatus.ACTIVE]
        
        if not available_agents:
            raise Exception("No available agents")
            
        # Get next agent in round-robin
        if "round_robin_index" not in self.load_balancer_state:
            self.load_balancer_state["round_robin_index"] = 0
            
        index = self.load_balancer_state["round_robin_index"]
        selected_agent = available_agents[index % len(available_agents)]
        self.load_balancer_state["round_robin_index"] = (index + 1) % len(available_agents)
        
        return RoutingDecision(
            selected_agents=[selected_agent],
            primary_agent=selected_agent,
            confidence=0.5,
            strategy_used=RoutingStrategy.ROUND_ROBIN,
            reasoning="Round-robin selection",
            fallback_agents=available_agents[1:],
            estimated_response_time=selected_agent.performance.average_response_time,
            estimated_cost=selected_agent.cost_per_request
        )
        
    async def _route_weighted_round_robin(self, analysis: QueryAnalysis, context: Optional[Dict]) -> RoutingDecision:
        """Weighted round-robin based on agent performance"""
        available_agents = [agent for agent in self.agents.values() 
                          if agent.status == AgentStatus.ACTIVE]
        
        if not available_agents:
            raise Exception("No available agents")
            
        # Calculate weights based on performance
        weights = []
        for agent in available_agents:
            weight = (
                agent.performance.success_rate * 0.4 +
                (1.0 - agent.performance.load_factor) * 0.3 +
                agent.performance.quality_score * 0.3
            )
            weights.append(weight)
            
        # Select agent based on weights
        total_weight = sum(weights)
        if total_weight == 0:
            return await self._route_round_robin(analysis, context)
            
        normalized_weights = [w / total_weight for w in weights]
        selected_index = np.random.choice(len(available_agents), p=normalized_weights)
        selected_agent = available_agents[selected_index]
        
        return RoutingDecision(
            selected_agents=[selected_agent],
            primary_agent=selected_agent,
            confidence=normalized_weights[selected_index],
            strategy_used=RoutingStrategy.WEIGHTED_ROUND_ROBIN,
            reasoning=f"Weighted selection based on performance (weight: {normalized_weights[selected_index]:.2f})",
            fallback_agents=[a for i, a in enumerate(available_agents) if i != selected_index],
            estimated_response_time=selected_agent.performance.average_response_time,
            estimated_cost=selected_agent.cost_per_request
        )
        
    async def _route_least_connections(self, analysis: QueryAnalysis, context: Optional[Dict]) -> RoutingDecision:
        """Route to agent with least connections"""
        available_agents = [agent for agent in self.agents.values() 
                          if agent.status == AgentStatus.ACTIVE]
        
        if not available_agents:
            raise Exception("No available agents")
            
        # Find agent with lowest current load
        selected_agent = min(available_agents, key=lambda a: a.performance.current_load)
        
        return RoutingDecision(
            selected_agents=[selected_agent],
            primary_agent=selected_agent,
            confidence=0.7,
            strategy_used=RoutingStrategy.LEAST_CONNECTIONS,
            reasoning=f"Least connections: {selected_agent.performance.current_load}",
            fallback_agents=[a for a in available_agents if a != selected_agent],
            estimated_response_time=selected_agent.performance.average_response_time,
            estimated_cost=selected_agent.cost_per_request
        )
        
    async def _route_performance_based(self, analysis: QueryAnalysis, context: Optional[Dict]) -> RoutingDecision:
        """Route based on agent performance metrics"""
        available_agents = [agent for agent in self.agents.values() 
                          if agent.status == AgentStatus.ACTIVE]
        
        if not available_agents:
            raise Exception("No available agents")
            
        # Calculate performance scores
        scored_agents = []
        for agent in available_agents:
            can_handle, confidence = agent.can_handle_query(analysis)
            if can_handle:
                performance_score = (
                    agent.performance.success_rate * 0.3 +
                    (1.0 - agent.performance.load_factor) * 0.2 +
                    agent.performance.quality_score * 0.2 +
                    confidence * 0.3
                )
                scored_agents.append((agent, performance_score))
                
        if not scored_agents:
            # Fallback to any available agent
            return await self._route_round_robin(analysis, context)
            
        # Select best performing agent
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        selected_agent, score = scored_agents[0]
        
        return RoutingDecision(
            selected_agents=[selected_agent],
            primary_agent=selected_agent,
            confidence=score,
            strategy_used=RoutingStrategy.PERFORMANCE_BASED,
            reasoning=f"Performance-based selection (score: {score:.2f})",
            fallback_agents=[agent for agent, _ in scored_agents[1:]],
            estimated_response_time=selected_agent.performance.average_response_time,
            estimated_cost=selected_agent.cost_per_request
        )
        
    async def _route_ml_optimized(self, analysis: QueryAnalysis, context: Optional[Dict]) -> RoutingDecision:
        """ML-optimized routing based on query analysis"""
        available_agents = [agent for agent in self.agents.values() 
                          if agent.status == AgentStatus.ACTIVE]
        
        if not available_agents:
            raise Exception("No available agents")
            
        # Find best matching agents
        agent_scores = []
        for agent in available_agents:
            can_handle, confidence = agent.can_handle_query(analysis)
            if can_handle:
                # Enhanced scoring with ML features
                ml_score = self._calculate_ml_score(agent, analysis)
                combined_score = confidence * 0.6 + ml_score * 0.4
                agent_scores.append((agent, combined_score, confidence))
                
        if not agent_scores:
            return await self._route_round_robin(analysis, context)
            
        # Select top agent
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        selected_agent, score, confidence = agent_scores[0]
        
        return RoutingDecision(
            selected_agents=[selected_agent],
            primary_agent=selected_agent,
            confidence=confidence,
            strategy_used=RoutingStrategy.ML_OPTIMIZED,
            reasoning=f"ML-optimized selection (score: {score:.2f})",
            fallback_agents=[agent for agent, _, _ in agent_scores[1:]],
            estimated_response_time=selected_agent.performance.average_response_time,
            estimated_cost=selected_agent.cost_per_request
        )
        
    async def _route_consensus_based(self, analysis: QueryAnalysis, context: Optional[Dict]) -> RoutingDecision:
        """Consensus-based routing for complex queries"""
        available_agents = [agent for agent in self.agents.values() 
                          if agent.status == AgentStatus.ACTIVE]
        
        if not available_agents:
            raise Exception("No available agents")
            
        # Select multiple agents for consensus
        capable_agents = []
        for agent in available_agents:
            can_handle, confidence = agent.can_handle_query(analysis)
            if can_handle and confidence > 0.4:
                capable_agents.append((agent, confidence))
                
        if not capable_agents:
            return await self._route_round_robin(analysis, context)
            
        # Sort by confidence and select top agents
        capable_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Select 2-3 agents for consensus
        consensus_count = min(3, len(capable_agents), max(2, len(capable_agents) // 2))
        selected_agents = [agent for agent, _ in capable_agents[:consensus_count]]
        primary_agent = selected_agents[0]
        
        avg_confidence = sum(conf for _, conf in capable_agents[:consensus_count]) / consensus_count
        
        return RoutingDecision(
            selected_agents=selected_agents,
            primary_agent=primary_agent,
            confidence=avg_confidence,
            strategy_used=RoutingStrategy.CONSENSUS_BASED,
            reasoning=f"Consensus-based selection with {len(selected_agents)} agents",
            fallback_agents=[agent for agent, _ in capable_agents[consensus_count:]],
            estimated_response_time=max(a.performance.average_response_time for a in selected_agents),
            estimated_cost=sum(a.cost_per_request for a in selected_agents)
        )
        
    async def _route_hybrid(self, analysis: QueryAnalysis, context: Optional[Dict]) -> RoutingDecision:
        """Hybrid routing combining multiple strategies"""
        available_agents = [agent for agent in self.agents.values() 
                          if agent.status == AgentStatus.ACTIVE]
        
        if not available_agents:
            raise Exception("No available agents")
            
        # Combine multiple scoring methods
        agent_scores = []
        for agent in available_agents:
            can_handle, confidence = agent.can_handle_query(analysis)
            if can_handle:
                # Performance score
                perf_score = (
                    agent.performance.success_rate * 0.3 +
                    (1.0 - agent.performance.load_factor) * 0.2 +
                    agent.performance.quality_score * 0.2 +
                    confidence * 0.3
                )
                
                # ML score
                ml_score = self._calculate_ml_score(agent, analysis)
                
                # Cost efficiency
                cost_score = 1.0 / (1.0 + agent.cost_per_request)
                
                # Combined score
                combined_score = (
                    perf_score * 0.4 +
                    ml_score * 0.3 +
                    cost_score * 0.2 +
                    confidence * 0.1
                )
                
                agent_scores.append((agent, combined_score, confidence))
                
        if not agent_scores:
            return await self._route_round_robin(analysis, context)
            
        # Select best agent
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        selected_agent, score, confidence = agent_scores[0]
        
        return RoutingDecision(
            selected_agents=[selected_agent],
            primary_agent=selected_agent,
            confidence=confidence,
            strategy_used=RoutingStrategy.HYBRID,
            reasoning=f"Hybrid selection (score: {score:.2f})",
            fallback_agents=[agent for agent, _, _ in agent_scores[1:]],
            estimated_response_time=selected_agent.performance.average_response_time,
            estimated_cost=selected_agent.cost_per_request
        )
        
    def _calculate_ml_score(self, agent: EnhancedAgent, analysis: QueryAnalysis) -> float:
        """Calculate ML-based score for agent-query matching"""
        score = 0.0
        
        # Category matching
        if agent.specializations:
            if analysis.primary_category in agent.specializations:
                score += 0.3
            if any(cat in agent.specializations for cat in analysis.categories):
                score += 0.2
                
        # Complexity matching
        if analysis.complexity > 0.7 and agent.priority_level > 2:
            score += 0.2
        elif analysis.complexity < 0.3 and agent.priority_level == 1:
            score += 0.1
            
        # Domain expertise
        if analysis.domain_expertise:
            expertise_overlap = len(set(analysis.domain_expertise) & set(agent.metadata.get('domains', [])))
            score += min(expertise_overlap * 0.1, 0.2)
            
        # Historical performance
        if hasattr(agent, 'category_performance'):
            category_perf = agent.category_performance.get(analysis.primary_category.value, 0.5)
            score += category_perf * 0.2
            
        return min(score, 1.0)
        
    def _update_performance_cache(self, decision: RoutingDecision, analysis: QueryAnalysis):
        """Update performance cache with routing decision"""
        cache_key = f"{decision.primary_agent.id}_{analysis.primary_category.value}"
        
        if cache_key not in self.performance_cache:
            self.performance_cache[cache_key] = {
                "total_routed": 0,
                "success_rate": 1.0,
                "avg_response_time": 0.0,
                "complexity_handled": []
            }
            
        cache_entry = self.performance_cache[cache_key]
        cache_entry["total_routed"] += 1
        cache_entry["complexity_handled"].append(analysis.complexity)
        
        # Keep only recent complexity values
        if len(cache_entry["complexity_handled"]) > 50:
            cache_entry["complexity_handled"] = cache_entry["complexity_handled"][-50:]
            
    async def update_agent_performance(self, agent_id: str, response_time: float, 
                                     success: bool, quality_score: float = None):
        """Update agent performance metrics"""
        if agent_id not in self.agents:
            return
            
        agent = self.agents[agent_id]
        agent.performance.response_times.append(response_time)
        agent.performance.total_requests += 1
        
        if success:
            agent.performance.success_count += 1
            self.circuit_breakers[agent_id]["success_count"] += 1
            self.circuit_breakers[agent_id]["failure_count"] = 0
        else:
            agent.performance.failure_count += 1
            self.circuit_breakers[agent_id]["failure_count"] += 1
            self.circuit_breakers[agent_id]["last_failure"] = datetime.now()
            
        if quality_score is not None:
            agent.performance.quality_score = quality_score
            
        # Update circuit breaker state
        await self._update_circuit_breaker(agent_id)
        
        agent.performance.last_updated = datetime.now()
        
    async def _update_circuit_breaker(self, agent_id: str):
        """Update circuit breaker state"""
        breaker = self.circuit_breakers[agent_id]
        agent = self.agents[agent_id]
        
        if breaker["state"] == "closed":
            if breaker["failure_count"] >= 5:  # Threshold
                breaker["state"] = "open"
                agent.status = AgentStatus.OFFLINE
                logger.warning(f"Circuit breaker opened for agent {agent_id}")
                
        elif breaker["state"] == "open":
            if breaker["last_failure"] and (datetime.now() - breaker["last_failure"]).seconds > 60:
                breaker["state"] = "half-open"
                agent.status = AgentStatus.ACTIVE
                logger.info(f"Circuit breaker half-opened for agent {agent_id}")
                
        elif breaker["state"] == "half-open":
            if breaker["success_count"] >= 3:
                breaker["state"] = "closed"
                breaker["failure_count"] = 0
                agent.status = AgentStatus.ACTIVE
                logger.info(f"Circuit breaker closed for agent {agent_id}")
                
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        total_routes = len(self.routing_history)
        
        if total_routes == 0:
            return {"total_routes": 0, "message": "No routing history"}
            
        # Strategy usage
        strategy_usage = defaultdict(int)
        for record in self.routing_history:
            strategy_usage[record["decision"].strategy_used.value] += 1
            
        # Category distribution
        category_dist = defaultdict(int)
        for record in self.routing_history:
            category_dist[record["analysis"].primary_category.value] += 1
            
        # Agent utilization
        agent_usage = defaultdict(int)
        for record in self.routing_history:
            agent_usage[record["decision"].primary_agent.id] += 1
            
        # Performance metrics
        response_times = [r["processing_time"] for r in self.routing_history]
        
        return {
            "total_routes": total_routes,
            "strategy_usage": dict(strategy_usage),
            "category_distribution": dict(category_dist),
            "agent_utilization": dict(agent_usage),
            "performance": {
                "avg_routing_time": sum(response_times) / len(response_times),
                "min_routing_time": min(response_times),
                "max_routing_time": max(response_times)
            },
            "circuit_breakers": {
                agent_id: breaker["state"] 
                for agent_id, breaker in self.circuit_breakers.items()
            }
        }