"""
ML Router Service
Service wrapper for the ML-Enhanced Query Router
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from ..extensions import db
from ..models.query import QueryLog, QueryMetrics
from ..models.agent import AgentRegistration
from ..utils.exceptions import ServiceError, ValidationError
from ..utils.async_helpers import run_async_in_thread

# Import the core ML router
try:
    import sys
    import os
    # Add the project root to Python path to import ml_router
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from ml_router import MLEnhancedQueryRouter, QueryCategory, Agent
    ML_ROUTER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML Router not available: {e}")
    ML_ROUTER_AVAILABLE = False
    MLEnhancedQueryRouter = None
    QueryCategory = None
    Agent = None

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result of a query processing operation"""
    success: bool
    response: str
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    category: Optional[str] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    cached: bool = False
    metadata: Dict[str, Any] = None

@dataclass
class RouterStats:
    """Router statistics"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    active_agents: int = 0
    category_distribution: Dict[str, int] = None

class MLRouterService:
    """
    Service wrapper for ML-Enhanced Query Router
    Provides async interface and database integration
    """
    
    def __init__(self):
        self.is_initialized = False
        self.core_router = None
        self.config = self._get_default_config()
        self._stats = RouterStats()
        
    def _get_default_config(self):
        """Get default configuration for the router"""
        from types import SimpleNamespace
        return SimpleNamespace(
            use_ml_classification=True,
            cache_ttl_seconds=3600,
            cache_max_size=10000,
            rate_limit_window_size=60,
            rate_limit_per_minute=100,
            min_confidence_threshold=0.6
        )
    
    async def initialize(self):
        """Initialize the ML router service"""
        try:
            logger.info("🔄 Initializing ML Router Service...")
            
            if not ML_ROUTER_AVAILABLE:
                logger.warning("⚠️ ML Router core not available, using mock implementation")
                self.is_initialized = True
                return
            
            # Get model manager if available
            model_manager = None
            try:
                from .ai_models import get_ai_model_manager
                model_manager = get_ai_model_manager()
            except Exception as e:
                logger.warning(f"Model manager not available: {e}")
            
            # Initialize the core router
            self.core_router = MLEnhancedQueryRouter(self.config, model_manager)
            await self.core_router.initialize()
            
            self.is_initialized = True
            logger.info("✅ ML Router Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ ML Router Service initialization failed: {e}")
            raise ServiceError(f"Router initialization failed: {e}")
    
    async def process_query(
        self,
        query: str,
        model_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> QueryResult:
        """
        Process a query through the ML router
        
        Args:
            query: The query text to process
            model_id: Optional specific model to use
            parameters: Optional query parameters
            session_id: Optional session identifier
            user_id: Optional user identifier
            
        Returns:
            QueryResult object with processing results
        """
        if not self.is_initialized:
            raise ServiceError("ML Router not initialized")
        
        start_time = datetime.now()
        
        try:
            # Validate query
            if not query or len(query.strip()) == 0:
                raise ValidationError("Query cannot be empty")
            
            if len(query) > 10000:  # Max query length
                raise ValidationError("Query too long")
            
            # Log query to database
            query_log = QueryLog(
                query_text=query,
                session_id=session_id,
                user_id=user_id,
                status='processing'
            )
            db.session.add(query_log)
            db.session.commit()
            
            # Process query through core router or mock
            if self.core_router:
                result_dict = await self.core_router.route_query(query, user_id)
            else:
                # Mock implementation for when core router is not available
                result_dict = await self._mock_process_query(query, user_id)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result object
            result = QueryResult(
                success=result_dict.get('status') == 'success',
                response=result_dict.get('response', result_dict.get('error', 'Unknown error')),
                agent_id=result_dict.get('agent_id'),
                agent_name=result_dict.get('agent_name'),
                category=result_dict.get('category'),
                confidence=result_dict.get('confidence', 0.0),
                processing_time=processing_time,
                cached=result_dict.get('cached', False),
                metadata=result_dict.get('metadata', {})
            )
            
            # Update query log
            query_log.status = 'completed' if result.success else 'failed'
            query_log.response_time = processing_time
            query_log.agent_id = result.agent_id
            query_log.agent_name = result.agent_name
            query_log.category = result.category or 'unknown'
            query_log.confidence = result.confidence
            if not result.success:
                query_log.error_message = result.response
            db.session.commit()
            
            # Update statistics
            self._update_stats(result, processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            
            # Update failed query log
            if 'query_log' in locals():
                query_log.status = 'failed'
                query_log.error_message = str(e)
                query_log.response_time = (datetime.now() - start_time).total_seconds()
                db.session.commit()
            
            return QueryResult(
                success=False,
                response=f"Query processing failed: {e}",
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _mock_process_query(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """Mock query processing when core router is not available"""
        # Simple keyword-based categorization for demo
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['code', 'program', 'function', 'debug']):
            category = 'coding'
            response = "This would be handled by the coding agent."
        elif any(word in query_lower for word in ['analyze', 'data', 'pattern']):
            category = 'analysis'
            response = "This would be handled by the analysis agent."
        elif any(word in query_lower for word in ['create', 'write', 'story']):
            category = 'creative'
            response = "This would be handled by the creative agent."
        else:
            category = 'conversational'
            response = "This would be handled by the general conversational agent."
        
        return {
            'status': 'success',
            'agent_id': f'mock-{category}-agent',
            'agent_name': f'Mock {category.title()} Agent',
            'category': category,
            'confidence': 0.8,
            'query': query,
            'response': response,
            'response_time': 0.5,
            'timestamp': datetime.now().isoformat(),
            'cached': False
        }
    
    async def stream_query(
        self,
        query: str,
        model_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Process a query with streaming response
        
        Args:
            query: The query text to process
            model_id: Optional specific model to use
            parameters: Optional query parameters
            session_id: Optional session identifier
            user_id: Optional user identifier
            
        Yields:
            Response chunks as they become available
        """
        if not self.is_initialized:
            raise ServiceError("ML Router not initialized")
        
        try:
            # For now, process normally and yield the result
            # In the future, this could be enhanced for true streaming
            result = await self.process_query(query, model_id, parameters, session_id, user_id)
            
            # Simulate streaming by breaking response into chunks
            response_text = result.response
            chunk_size = 50  # Characters per chunk
            
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                yield f"data: {{'chunk': '{chunk}', 'type': 'content'}}\n\n"
            
            # Send completion signal
            yield f"data: {{'type': 'done', 'metadata': {{'agent': '{result.agent_name}', 'category': '{result.category}'}}}}\n\n"
                
        except Exception as e:
            logger.error(f"Streaming query error: {e}")
            yield f"data: {{'error': '{str(e)}', 'type': 'error'}}\n\n"
    
    async def register_agent(
        self,
        name: str,
        description: str,
        categories: List[str],
        endpoint: str,
        capabilities: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Register a new agent"""
        try:
            if self.core_router:
                agent_id = await self.core_router.register_agent(
                    name, description, categories, endpoint, capabilities, metadata
                )
            else:
                # Mock agent registration
                import uuid
                agent_id = str(uuid.uuid4())
                
                # Save to database
                agent_reg = AgentRegistration(
                    id=agent_id,
                    name=name,
                    description=description,
                    endpoint=endpoint,
                    categories=categories,
                    capabilities=capabilities or {},
                    agent_metadata=metadata or {},
                    is_active=True
                )
                db.session.add(agent_reg)
                db.session.commit()
            
            logger.info(f"Agent registered: {name} ({agent_id})")
            return agent_id
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            raise ServiceError(f"Failed to register agent: {e}")
    
    def get_agents(self) -> List[Dict[str, Any]]:
        """Get all registered agents"""
        try:
            agents = db.session.query(AgentRegistration).filter_by(is_active=True).all()
            return [agent.to_dict() for agent in agents]
            
        except Exception as e:
            logger.error(f"Failed to get agents: {e}")
            return []
    
    def get_stats(self) -> RouterStats:
        """Get router statistics"""
        try:
            # Update stats from database
            total_queries = db.session.query(QueryLog).count()
            successful_queries = db.session.query(QueryLog).filter_by(status='completed').count()
            failed_queries = db.session.query(QueryLog).filter_by(status='failed').count()
            
            # Calculate averages
            avg_response_time = db.session.query(db.func.avg(QueryLog.response_time)).scalar() or 0.0
            
            # Get category distribution
            category_dist = {}
            categories = db.session.query(QueryLog.category, db.func.count(QueryLog.category)).group_by(QueryLog.category).all()
            for category, count in categories:
                category_dist[category] = count
            
            # Active agents
            active_agents = db.session.query(AgentRegistration).filter_by(is_active=True).count()
            
            return RouterStats(
                total_queries=total_queries,
                successful_queries=successful_queries,
                failed_queries=failed_queries,
                average_response_time=avg_response_time,
                active_agents=active_agents,
                category_distribution=category_dist
            )
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return RouterStats()
    
    def _update_stats(self, result: QueryResult, processing_time: float):
        """Update internal statistics"""
        self._stats.total_queries += 1
        
        if result.success:
            self._stats.successful_queries += 1
        else:
            self._stats.failed_queries += 1
        
        # Update average response time
        if self._stats.total_queries == 1:
            self._stats.average_response_time = processing_time
        else:
            alpha = 0.1  # Exponential moving average factor
            self._stats.average_response_time = (
                alpha * processing_time + 
                (1 - alpha) * self._stats.average_response_time
            )

# Singleton instance
_ml_router_instance = None

def get_ml_router() -> MLRouterService:
    """
    Get the singleton ML router service instance
    
    Returns:
        MLRouterService instance
    """
    global _ml_router_instance
    
    if _ml_router_instance is None:
        _ml_router_instance = MLRouterService()
        
        # Initialize if not already done
        if not _ml_router_instance.is_initialized:
            try:
                # Run initialization in a thread to avoid blocking
                import threading
                
                def init_router():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(_ml_router_instance.initialize())
                        loop.close()
                        logger.info("✅ ML Router background initialization completed")
                    except Exception as e:
                        logger.error(f"❌ ML Router background initialization failed: {e}")
                
                init_thread = threading.Thread(target=init_router, daemon=True)
                init_thread.start()
                
            except Exception as e:
                logger.error(f"Failed to start ML router initialization: {e}")
    
    return _ml_router_instance

# Export for backward compatibility
MLRouterService = MLRouterService