"""
Agent Service
Handles agent registration, management, and routing decisions
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from flask import current_app
from sqlalchemy import func, and_

from app.extensions import db
from app.utils.exceptions import AgentError, ValidationError, ServiceError


class AgentStatus(Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentService:
    """Service for handling agent operations"""
    
    def __init__(self):
        self.registered_agents = {}
        self.agent_capabilities = {}
        self.load_balancing_strategy = "round_robin"
        self.health_check_interval = 300  # 5 minutes
        self.max_concurrent_sessions = 10
        self.initialized = False
    
    def initialize(self, app):
        """Initialize the agent service with app configuration"""
        try:
            self.load_balancing_strategy = app.config.get('AGENT_LOAD_BALANCING', 'round_robin')
            self.health_check_interval = app.config.get('AGENT_HEALTH_CHECK_INTERVAL', 300)
            self.max_concurrent_sessions = app.config.get('AGENT_MAX_CONCURRENT_SESSIONS', 10)
            
            self._load_agents_from_db()
            
            self.initialized = True
            current_app.logger.info("Agent service initialized successfully")
            
        except Exception as e:
            current_app.logger.error(f"Agent service initialization failed: {e}")
            self.initialized = False
    
    def get_all_agents(self) -> List:
        """
        Get all active agents
        
        Returns:
            List of active Agent objects
            
        Raises:
            AgentError: If service not initialized
        """
        if not self.initialized:
            raise AgentError("Agent service not initialized")
        
        try:
            # Import within method to avoid circular imports
            from app.models.agent import Agent
            agents = Agent.query.filter_by(is_active=True).all()
            return agents
            
        except Exception as e:
            current_app.logger.error(f"Error getting all agents: {e}")
            return []

    def get_agent(self, agent_id: str) -> Optional:
        """
        Get a specific agent by ID
        
        Args:
            agent_id: The agent ID to look up
            
        Returns:
            Agent object or None if not found
            
        Raises:
            AgentError: If service not initialized
        """
        if not self.initialized:
            raise AgentError("Agent service not initialized")
        
        try:
            from app.models.agent import Agent
            agent = Agent.query.filter_by(id=agent_id, is_active=True).first()
            return agent
            
        except Exception as e:
            current_app.logger.error(f"Error getting agent {agent_id}: {e}")
            return None

    async def check_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """
        Check the health status of a specific agent
        
        Args:
            agent_id: The agent ID to check
            
        Returns:
            Dict containing health status information
            
        Raises:
            AgentError: If service not initialized
        """
        if not self.initialized:
            raise AgentError("Agent service not initialized")
        
        try:
            from app.models.agent import Agent, AgentMetrics
            agent = Agent.query.filter_by(id=agent_id, is_active=True).first()
            if not agent:
                return {
                    'status': 'not_found',
                    'healthy': False,
                    'message': 'Agent not found'
                }
            
            # Basic health check
            health_status = {
                'status': agent.status,
                'healthy': agent.status == AgentStatus.ACTIVE.value and agent.is_healthy,
                'last_seen': agent.last_seen.isoformat() if agent.last_seen else None,
                'active_sessions': agent.active_sessions,
                'max_sessions': agent.max_concurrent_sessions,
                'response_time': agent.avg_response_time,
                'success_rate': agent.success_rate
            }
            
            # Get recent metrics if available
            latest_metrics = AgentMetrics.query.filter_by(agent_id=agent_id).order_by(
                AgentMetrics.period_end.desc()
            ).first()
            
            if latest_metrics:
                health_status['response_time'] = latest_metrics.average_response_time
                health_status['success_rate'] = (
                    latest_metrics.successful_requests / 
                    max(latest_metrics.total_requests, 1) * 100
                )
            
            return health_status
            
        except Exception as e:
            current_app.logger.error(f"Error checking agent health {agent_id}: {e}")
            return {
                'status': 'error',
                'healthy': False,
                'message': str(e)
            }
    
    def register_agent(self, agent_data: Dict[str, Any]) -> str:
        """
        Register a new agent
        
        Args:
            agent_data: Dict containing agent information
            
        Returns:
            Agent ID
            
        Raises:
            ValidationError: If agent data is invalid
            AgentError: If registration fails
        """
        if not self.initialized:
            raise AgentError("Agent service not initialized")
        
        try:
            from app.models.agent import Agent, AgentCapability, AgentMetrics
            
            required_fields = ['name', 'type', 'endpoint', 'capabilities']
            for field in required_fields:
                if not agent_data.get(field):
                    raise ValidationError(f"Missing required field: {field}")
            
            agent_id = str(uuid.uuid4())
            
            agent = Agent(
                id=agent_id,
                name=agent_data['name'],
                type=agent_data['type'],
                endpoint=agent_data['endpoint'],
                description=agent_data.get('description', ''),
                version=agent_data.get('version', '1.0.0'),
                status=AgentStatus.ACTIVE.value,
                agent_metadata=agent_data.get('metadata', {}),
                max_concurrent_sessions=agent_data.get('max_concurrent_sessions', self.max_concurrent_sessions),
                last_seen=datetime.utcnow(),
                is_active=True
            )
            
            db.session.add(agent)
            db.session.flush()
            
            capabilities = agent_data['capabilities']
            if isinstance(capabilities, str):
                capabilities = [capabilities]
            
            for capability_name in capabilities:
                capability = AgentCapability(
                    agent_id=agent_id,
                    capability=capability_name,
                    confidence_score=agent_data.get('confidence_scores', {}).get(capability_name, 1.0),
                    is_active=True
                )
                db.session.add(capability)
            
            metrics = AgentMetrics(
                agent_id=agent_id,
                agent_name=agent_data['name'],
                period_start=datetime.utcnow(),
                period_end=datetime.utcnow(),
                granularity='day',
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time=0.0
            )
            db.session.add(metrics)
            
            db.session.commit()
            
            self.registered_agents[agent_id] = agent
            self.agent_capabilities[agent_id] = capabilities
            
            current_app.logger.info(f"Agent registered: {agent_id} ({agent_data['name']})")
            return agent_id
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Agent registration error: {e}")
            if isinstance(e, (ValidationError, AgentError)):
                raise
            raise AgentError(f"Failed to register agent: {str(e)}")
    
    def get_agent_for_capability(self, capability: str, exclude_agents: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the best agent for a specific capability
        
        Args:
            capability: Required capability
            exclude_agents: List of agent IDs to exclude
            
        Returns:
            Agent info dict or None if no suitable agent found
        """
        if not self.initialized:
            raise AgentError("Agent service not initialized")
        
        try:
            from app.models.agent import Agent, AgentCapability, AgentMetrics
            
            exclude_agents = exclude_agents or []
            
            query = db.session.query(Agent, AgentCapability, AgentMetrics).join(
                AgentCapability, Agent.id == AgentCapability.agent_id
            ).join(
                AgentMetrics, Agent.id == AgentMetrics.agent_id
            ).filter(
                and_(
                    Agent.is_active == True,
                    Agent.status == AgentStatus.ACTIVE.value,
                    AgentCapability.capability == capability,
                    AgentCapability.is_active == True,
                    ~Agent.id.in_(exclude_agents)
                )
            )
            
            if self.load_balancing_strategy == "round_robin":
                query = query.order_by(Agent.last_used.asc().nullsfirst())
            elif self.load_balancing_strategy == "least_loaded":
                query = query.order_by(Agent.active_sessions.asc())
            elif self.load_balancing_strategy == "best_performance":
                query = query.order_by(
                    AgentMetrics.average_response_time.asc(),
                    (AgentMetrics.successful_requests / func.greatest(AgentMetrics.total_requests, 1)).desc()
                )
            elif self.load_balancing_strategy == "highest_confidence":
                query = query.order_by(AgentCapability.confidence_score.desc())
            
            result = query.first()
            
            if not result:
                return None
            
            agent, capability_record, metrics = result
            
            if agent.active_sessions >= agent.max_concurrent_sessions:
                current_app.logger.warning(f"Agent {agent.id} at max capacity")
                return None
            
            agent.last_used = datetime.utcnow()
            db.session.commit()
            
            return {
                'id': agent.id,
                'name': agent.name,
                'type': agent.type,
                'endpoint': agent.endpoint,
                'capability': capability,
                'confidence_score': capability_record.confidence_score,
                'active_sessions': agent.active_sessions,
                'max_sessions': agent.max_concurrent_sessions,
                'average_response_time': metrics.average_response_time,
                'success_rate': (metrics.successful_requests / max(metrics.total_requests, 1)) * 100
            }
            
        except Exception as e:
            current_app.logger.error(f"Agent selection error: {e}")
            return None
    
    def route_query(self, query: str, required_capabilities: List[str] = None, 
                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route a query to the most appropriate agent
        
        Args:
            query: User query
            required_capabilities: List of required capabilities
            context: Additional context for routing decision
            
        Returns:
            Dict containing routing decision and agent info
        """
        if not self.initialized:
            raise AgentError("Agent service not initialized")
        
        try:
            if not required_capabilities:
                required_capabilities = self._analyze_query_capabilities(query, context)
            
            selected_agents = {}
            routing_scores = {}
            
            for capability in required_capabilities:
                agent = self.get_agent_for_capability(capability)
                if agent:
                    selected_agents[capability] = agent
                    routing_scores[capability] = self._calculate_routing_score(agent, query, context)
            
            if not selected_agents:
                return {
                    'success': False,
                    'error': 'No suitable agents available',
                    'required_capabilities': required_capabilities
                }
            
            primary_capability = max(routing_scores.keys(), key=lambda k: routing_scores[k])
            primary_agent = selected_agents[primary_capability]
            
            session_id = self._create_agent_session(primary_agent['id'], query, context)
            
            return {
                'success': True,
                'session_id': session_id,
                'primary_agent': primary_agent,
                'backup_agents': {k: v for k, v in selected_agents.items() if k != primary_capability},
                'routing_score': routing_scores[primary_capability],
                'required_capabilities': required_capabilities
            }
            
        except Exception as e:
            current_app.logger.error(f"Query routing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'required_capabilities': required_capabilities or []
            }
    
    def update_agent_status(self, agent_id: str, status: str, metadata: Dict[str, Any] = None) -> bool:
        """Update agent status"""
        try:
            from app.models.agent import Agent
            
            agent = Agent.query.get(agent_id)
            if not agent:
                return False
            
            agent.status = status
            agent.last_seen = datetime.utcnow()
            
            if metadata:
                if agent.agent_metadata is None:
                    agent.agent_metadata = {}
                agent.agent_metadata.update(metadata)
            
            db.session.commit()
            
            if agent_id in self.registered_agents:
                self.registered_agents[agent_id] = agent
            
            current_app.logger.info(f"Agent status updated: {agent_id} -> {status}")
            return True
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Agent status update error: {e}")
            return False
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        try:
            from app.models.agent import Agent, AgentCapability, AgentMetrics, AgentSession
            
            stats = {
                'total_agents': Agent.query.filter_by(is_active=True).count(),
                'active_agents': Agent.query.filter_by(
                    is_active=True, 
                    status=AgentStatus.ACTIVE.value
                ).count(),
                'total_capabilities': AgentCapability.query.filter_by(is_active=True).count(),
                'active_sessions': AgentSession.query.filter_by(is_active=True).count(),
                'agents_by_type': {},
                'capabilities_distribution': {},
                'performance_summary': {}
            }
            
            agent_types = db.session.query(
                Agent.type, func.count(Agent.id)
            ).filter_by(is_active=True).group_by(Agent.type).all()
            
            stats['agents_by_type'] = {agent_type: count for agent_type, count in agent_types}
            
            capabilities = db.session.query(
                AgentCapability.capability, func.count(AgentCapability.id)
            ).filter_by(is_active=True).group_by(AgentCapability.capability).all()
            
            stats['capabilities_distribution'] = {cap: count for cap, count in capabilities}
            
            performance = db.session.query(
                func.avg(AgentMetrics.average_response_time),
                func.avg(AgentMetrics.successful_requests / func.greatest(AgentMetrics.total_requests, 1) * 100),
                func.sum(AgentMetrics.total_requests)
            ).first()
            
            if performance and performance[0] is not None:
                stats['performance_summary'] = {
                    'avg_response_time': float(performance[0]),
                    'avg_success_rate': float(performance[1]),
                    'total_requests': int(performance[2])
                }
            
            return stats
            
        except Exception as e:
            current_app.logger.error(f"Agent stats error: {e}")
            return {'error': str(e)}
    
    def _load_agents_from_db(self):
        """Load existing agents from database into memory"""
        try:
            # Import within method to avoid circular import issues
            from app.models.agent import Agent, AgentCapability
            
            agents = Agent.query.filter_by(is_active=True).all()
            for agent in agents:
                self.registered_agents[agent.id] = agent
                
                capabilities = AgentCapability.query.filter_by(
                    agent_id=agent.id, 
                    is_active=True
                ).all()
                self.agent_capabilities[agent.id] = [cap.capability for cap in capabilities]
            
            current_app.logger.info(f"Loaded {len(agents)} agents from database")
            
        except Exception as e:
            current_app.logger.error(f"Agent loading error: {e}")
    
    def _analyze_query_capabilities(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """Analyze query to determine required capabilities"""
        capabilities = []
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['search', 'find', 'lookup', 'retrieve']):
            capabilities.append('search')
        
        if any(word in query_lower for word in ['generate', 'create', 'write', 'compose']):
            capabilities.append('generation')
        
        if any(word in query_lower for word in ['analyze', 'summarize', 'explain', 'understand']):
            capabilities.append('analysis')
        
        if any(word in query_lower for word in ['translate', 'language']):
            capabilities.append('translation')
        
        if any(word in query_lower for word in ['code', 'program', 'script', 'function']):
            capabilities.append('coding')
        
        if not capabilities:
            capabilities.append('general')
        
        return capabilities
    
    def _calculate_routing_score(self, agent: Dict[str, Any], query: str, 
                               context: Dict[str, Any] = None) -> float:
        """Calculate routing score for agent"""
        score = 0.0
        
        score += agent['confidence_score'] * 0.4
        
        if agent['average_response_time'] > 0:
            response_score = max(0, 1 - (agent['average_response_time'] / 10.0))
            score += response_score * 0.3
        
        score += (agent['success_rate'] / 100.0) * 0.2
        
        load_factor = agent['active_sessions'] / agent['max_sessions']
        load_score = max(0, 1 - load_factor)
        score += load_score * 0.1
        
        return min(score, 1.0)
    
    def _create_agent_session(self, agent_id: str, query: str, context: Dict[str, Any] = None) -> str:
        """Create new agent session"""
        try:
            from app.models.agent import Agent, AgentSession
            
            session_id = str(uuid.uuid4())
            
            session = AgentSession(
                id=session_id,
                agent_id=agent_id,
                session_type='query',
                user_id=context.get('user_id') if context else None,
                context=context or {},
                is_active=True
            )
            
            db.session.add(session)
            
            agent = Agent.query.get(agent_id)
            if agent:
                agent.active_sessions += 1
            
            db.session.commit()
            
            return session_id
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Session creation error: {e}")
            return None


_agent_service = None

def get_agent_service() -> AgentService:
    """Get singleton agent service instance"""
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
        if current_app:
            _agent_service.initialize(current_app)
    return _agent_service

def init_agent_service(app):
    """Initialize agent service with Flask app"""
    service = get_agent_service()
    service.initialize(app)
    return service