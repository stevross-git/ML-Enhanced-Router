"""
GraphQL Agent Mesh API Schema
Provides unified GraphQL endpoint for agent mesh operations
"""

import graphene
from graphene import ObjectType, String, List, Float, Int, Boolean, Field, Mutation, Schema
from typing import Optional, Dict, Any, List as ListType
import json
import asyncio
from datetime import datetime

# Import with fallback handling
try:
    from ml_router import MLEnhancedQueryRouter
except ImportError:
    MLEnhancedQueryRouter = None

try:
    from ai_models import AIModelManager
except ImportError:
    AIModelManager = None

try:
    from collaborative_router import CollaborativeRouter
except ImportError:
    CollaborativeRouter = None

try:
    from advanced_ml_classifier import AdvancedMLClassifier, QueryAnalysis
except ImportError:
    AdvancedMLClassifier = None
    QueryAnalysis = None

try:
    from intelligent_routing_engine import IntelligentRoutingEngine
except ImportError:
    IntelligentRoutingEngine = None

try:
    from predictive_analytics_engine import PredictiveAnalyticsEngine
except ImportError:
    PredictiveAnalyticsEngine = None

try:
    from real_time_analytics import RealTimeAnalytics
except ImportError:
    RealTimeAnalytics = None

try:
    from auto_chain_generator import AutoChainGenerator
except ImportError:
    AutoChainGenerator = None


class AgentCapability(ObjectType):
    """Agent capability GraphQL type"""
    name = String()
    description = String()
    confidence = Float()


class AgentInfo(ObjectType):
    """Agent information GraphQL type"""
    id = String()
    name = String()
    description = String()
    provider = String()
    model_name = String()
    capabilities = List(AgentCapability)
    is_active = Boolean()
    load_factor = Float()
    success_rate = Float()
    avg_response_time = Float()
    cost_per_1k_tokens = Float()
    context_window = Int()
    supports_streaming = Boolean()
    supports_multimodal = Boolean()


class QueryCategory(ObjectType):
    """Query category GraphQL type"""
    name = String()
    confidence = Float()
    description = String()


class QueryClassification(ObjectType):
    """Query classification result GraphQL type"""
    query = String()
    primary_category = String()
    categories = List(QueryCategory)
    confidence = Float()
    complexity = Float()
    intent = String()
    technical_level = String()
    estimated_tokens = Int()
    processing_time = Float()
    suggested_agents = List(String)
    required_capabilities = List(String)


class RoutingDecision(ObjectType):
    """Routing decision GraphQL type"""
    agent_id = String()
    agent_name = String()
    confidence = Float()
    cost_estimate = Float()
    estimated_response_time = Float()
    routing_strategy = String()
    fallback_agents = List(String)
    reasoning = String()


class AgentExecution(ObjectType):
    """Agent execution result GraphQL type"""
    agent_id = String()
    response = String()
    execution_time = Float()
    tokens_used = Int()
    cost = Float()
    success = Boolean()
    error_message = String()
    metadata = String()  # JSON string


class CollaborativeResult(ObjectType):
    """Collaborative AI result GraphQL type"""
    session_id = String()
    query = String()
    primary_response = String()
    agent_responses = List(String)
    synthesis = String()
    confidence = Float()
    total_tokens = Int()
    total_cost = Float()
    execution_time = Float()
    agents_used = List(String)


class SystemMetrics(ObjectType):
    """System metrics GraphQL type"""
    total_queries = Int()
    total_agents = Int()
    active_agents = Int()
    avg_response_time = Float()
    success_rate = Float()
    cache_hit_rate = Float()
    total_tokens_used = Int()
    total_cost = Float()
    uptime = Float()

class ChainStep(ObjectType):
    """Chain step GraphQL type"""
    step_id = String()
    step_type = String()
    agent_id = String()
    agent_name = String()
    description = String()
    input_from = String()
    expected_output = String()
    estimated_cost = Float()
    estimated_time = Float()

class AgentChain(ObjectType):
    """Agent chain GraphQL type"""
    chain_id = String()
    query = String()
    steps = List(ChainStep)
    estimated_cost = Float()
    estimated_time = Float()
    complexity_score = Float()
    created_at = String()

class ChainAnalysis(ObjectType):
    """Chain analysis GraphQL type"""
    query = String()
    complexity = Float()
    intent = String()
    domain = String()
    requires_rag = Boolean()
    requires_debate = Boolean()
    requires_research = Boolean()
    requires_creativity = Boolean()
    output_format = String()
    estimated_steps = Int()

class ChainTemplate(ObjectType):
    """Chain template GraphQL type"""
    name = String()
    description = String()
    steps = List(String)

class ChainStats(ObjectType):
    """Chain generator statistics GraphQL type"""
    available_templates = Int()
    step_types = Int()
    model_count = Int()
    components_available = String()

class ChainExecution(ObjectType):
    """Chain execution result GraphQL type"""
    chain_id = String()
    query = String()
    success_rate = Float()
    total_cost = Float()
    total_time = Float()
    results = List(String)


class PredictionResult(ObjectType):
    """Prediction result GraphQL type"""
    prediction_type = String()
    value = Float()
    confidence = Float()
    timeframe = String()
    metadata = String()


class Query(ObjectType):
    """Main GraphQL Query type"""
    
    # Agent mesh queries
    agents = List(AgentInfo, active_only=Boolean(default_value=False))
    agent = Field(AgentInfo, id=String(required=True))
    
    # Query classification
    classify = Field(QueryClassification, query=String(required=True))
    
    # Routing decisions
    route = Field(RoutingDecision, 
                  query=String(required=True),
                  strategy=String(default_value="ml_optimized"))
    
    # Multi-agent routing
    route_collaborative = Field(CollaborativeResult,
                               query=String(required=True),
                               agents=List(String),
                               max_agents=Int(default_value=3))
    
    # System metrics
    metrics = Field(SystemMetrics)
    
    # Predictions
    predict = Field(PredictionResult,
                   prediction_type=String(required=True),
                   timeframe=String(default_value="1h"))
    
    # Auto Chain Generator
    chain_templates = List(ChainTemplate)
    chain_stats = Field(ChainStats)
    analyze_chain = Field(ChainAnalysis, query=String(required=True))
    generate_chain = Field(AgentChain, query=String(required=True))
    execute_chain = Field(ChainExecution, query=String(required=True))

    def resolve_agents(self, info, active_only=False):
        """Resolve agents query"""
        try:
            if not AIModelManager:
                return []
            
            model_manager = AIModelManager()
            models = model_manager.get_all_models()
            
            agents = []
            for model in models:
                if active_only and not model.get('is_active', False):
                    continue
                    
                agent = AgentInfo(
                    id=model.get('id', ''),
                    name=model.get('name', ''),
                    description=model.get('description', ''),
                    provider=model.get('provider', ''),
                    model_name=model.get('model_name', ''),
                    capabilities=[
                        AgentCapability(
                            name=cap,
                            description=f"Supports {cap}",
                            confidence=0.9
                        ) for cap in model.get('capabilities', [])
                    ],
                    is_active=model.get('is_active', False),
                    load_factor=model.get('load_factor', 0.0),
                    success_rate=model.get('success_rate', 0.0),
                    avg_response_time=model.get('avg_response_time', 0.0),
                    cost_per_1k_tokens=model.get('cost_per_1k_tokens', 0.0),
                    context_window=model.get('context_window', 0),
                    supports_streaming=model.get('supports_streaming', False),
                    supports_multimodal=model.get('supports_multimodal', False)
                )
                agents.append(agent)
            
            return agents
            
        except Exception as e:
            print(f"Error resolving agents: {e}")
            return []

    def resolve_agent(self, info, id):
        """Resolve single agent query"""
        try:
            agents = self.resolve_agents(info, active_only=False)
            return next((agent for agent in agents if agent.id == id), None)
        except Exception as e:
            print(f"Error resolving agent {id}: {e}")
            return None

    def resolve_classify(self, info, query):
        """Resolve query classification"""
        try:
            classifier = AdvancedMLClassifier()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            analysis = loop.run_until_complete(classifier.analyze(query))
            
            # Convert to GraphQL types
            categories = [
                QueryCategory(
                    name=cat.value,
                    confidence=0.8,  # Default confidence
                    description=f"Category: {cat.value}"
                ) for cat in [analysis.primary_category]
            ]
            
            suggested_agents = []
            # Get suggested agents based on classification
            model_manager = AIModelManager()
            models = model_manager.get_all_models()
            for model in models[:3]:  # Top 3 suggestions
                if model.get('is_active', False):
                    suggested_agents.append(model.get('id', ''))
            
            return QueryClassification(
                query=query,
                primary_category=analysis.primary_category.value,
                categories=categories,
                confidence=analysis.confidence,
                complexity=analysis.complexity,
                intent=analysis.intent.value,
                technical_level=analysis.technical_level,
                estimated_tokens=analysis.estimated_tokens,
                processing_time=analysis.processing_time,
                suggested_agents=suggested_agents,
                required_capabilities=analysis.required_capabilities
            )
            
        except Exception as e:
            print(f"Error classifying query: {e}")
            return None

    def resolve_route(self, info, query, strategy="ml_optimized"):
        """Resolve routing decision"""
        try:
            router = MLEnhancedQueryRouter()
            
            # Get routing decision
            result = router.route_query(query, strategy=strategy)
            
            if result and 'agent_id' in result:
                return RoutingDecision(
                    agent_id=result['agent_id'],
                    agent_name=result.get('agent_name', ''),
                    confidence=result.get('confidence', 0.0),
                    cost_estimate=result.get('cost_estimate', 0.0),
                    estimated_response_time=result.get('estimated_response_time', 0.0),
                    routing_strategy=strategy,
                    fallback_agents=result.get('fallback_agents', []),
                    reasoning=result.get('reasoning', '')
                )
            
            return None
            
        except Exception as e:
            print(f"Error routing query: {e}")
            return None

    def resolve_route_collaborative(self, info, query, agents=None, max_agents=3):
        """Resolve collaborative routing"""
        try:
            collaborative_router = CollaborativeRouter()
            
            # Process collaborative query
            result = collaborative_router.process_collaborative_query(
                query, 
                selected_agents=agents,
                max_agents=max_agents
            )
            
            if result:
                return CollaborativeResult(
                    session_id=result.get('session_id', ''),
                    query=query,
                    primary_response=result.get('primary_response', ''),
                    agent_responses=result.get('agent_responses', []),
                    synthesis=result.get('synthesis', ''),
                    confidence=result.get('confidence', 0.0),
                    total_tokens=result.get('total_tokens', 0),
                    total_cost=result.get('total_cost', 0.0),
                    execution_time=result.get('execution_time', 0.0),
                    agents_used=result.get('agents_used', [])
                )
            
            return None
            
        except Exception as e:
            print(f"Error in collaborative routing: {e}")
            return None

    def resolve_metrics(self, info):
        """Resolve system metrics"""
        try:
            analytics = RealTimeAnalytics()
            metrics = analytics.get_system_metrics()
            
            return SystemMetrics(
                total_queries=metrics.get('total_queries', 0),
                total_agents=metrics.get('total_agents', 0),
                active_agents=metrics.get('active_agents', 0),
                avg_response_time=metrics.get('avg_response_time', 0.0),
                success_rate=metrics.get('success_rate', 0.0),
                cache_hit_rate=metrics.get('cache_hit_rate', 0.0),
                total_tokens_used=metrics.get('total_tokens_used', 0),
                total_cost=metrics.get('total_cost', 0.0),
                uptime=metrics.get('uptime', 0.0)
            )
            
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return SystemMetrics(
                total_queries=0,
                total_agents=0,
                active_agents=0,
                avg_response_time=0.0,
                success_rate=0.0,
                cache_hit_rate=0.0,
                total_tokens_used=0,
                total_cost=0.0,
                uptime=0.0
            )
    
    def resolve_chain_templates(self, info):
        """Resolve chain templates query"""
        try:
            if not AutoChainGenerator:
                return []
            
            generator = AutoChainGenerator()
            templates = []
            
            for name, template in generator.chain_templates.items():
                templates.append(ChainTemplate(
                    name=name,
                    description=f"Template for {name.replace('_', ' ').title()}",
                    steps=[step.value for step in template]
                ))
            
            return templates
            
        except Exception as e:
            print(f"Error getting chain templates: {e}")
            return []
    
    def resolve_chain_stats(self, info):
        """Resolve chain generator statistics"""
        try:
            if not AutoChainGenerator:
                return ChainStats(
                    available_templates=0,
                    step_types=0,
                    model_count=0,
                    components_available="{}"
                )
            
            generator = AutoChainGenerator()
            stats = generator.get_chain_stats()
            
            return ChainStats(
                available_templates=stats.get('available_templates', 0),
                step_types=stats.get('step_types', 0),
                model_count=stats.get('model_count', 0),
                components_available=json.dumps(stats.get('components_available', {}))
            )
            
        except Exception as e:
            print(f"Error getting chain stats: {e}")
            return ChainStats(
                available_templates=0,
                step_types=0,
                model_count=0,
                components_available="{}"
            )
    
    def resolve_analyze_chain(self, info, query):
        """Resolve chain analysis query"""
        try:
            if not AutoChainGenerator:
                return None
            
            generator = AutoChainGenerator()
            analysis = generator.analyze_query_for_chain(query)
            
            return ChainAnalysis(
                query=analysis.get('query', ''),
                complexity=analysis.get('complexity', 0.0),
                intent=analysis.get('intent', ''),
                domain=analysis.get('domain', ''),
                requires_rag=analysis.get('requires_rag', False),
                requires_debate=analysis.get('requires_debate', False),
                requires_research=analysis.get('requires_research', False),
                requires_creativity=analysis.get('requires_creativity', False),
                output_format=analysis.get('output_format', ''),
                estimated_steps=analysis.get('estimated_steps', 0)
            )
            
        except Exception as e:
            print(f"Error analyzing chain: {e}")
            return None
    
    def resolve_generate_chain(self, info, query):
        """Resolve chain generation query"""
        try:
            if not AutoChainGenerator:
                return None
            
            generator = AutoChainGenerator()
            chain = generator.generate_chain(query)
            
            steps = []
            for step in chain.steps:
                steps.append(ChainStep(
                    step_id=step.step_id,
                    step_type=step.step_type.value,
                    agent_id=step.agent_id,
                    agent_name=step.agent_name,
                    description=step.description,
                    input_from=step.input_from,
                    expected_output=step.expected_output,
                    estimated_cost=step.parameters.get('estimated_cost', 0.0),
                    estimated_time=step.parameters.get('estimated_time', 0.0)
                ))
            
            return AgentChain(
                chain_id=chain.chain_id,
                query=chain.query,
                steps=steps,
                estimated_cost=chain.estimated_cost,
                estimated_time=chain.estimated_time,
                complexity_score=chain.complexity_score,
                created_at=chain.created_at.isoformat()
            )
            
        except Exception as e:
            print(f"Error generating chain: {e}")
            return None
    
    def resolve_execute_chain(self, info, query):
        """Resolve chain execution query"""
        try:
            if not AutoChainGenerator:
                return None
            
            generator = AutoChainGenerator()
            chain = generator.generate_chain(query)
            
            # Execute chain asynchronously
            async def execute():
                return await generator.execute_chain(chain)
            
            # Run in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(execute())
            finally:
                loop.close()
            
            # Calculate metrics
            success_rate = sum(1 for r in results if r.success) / len(results) if results else 0
            total_cost = sum(r.cost for r in results)
            total_time = sum(r.execution_time for r in results)
            
            return ChainExecution(
                chain_id=chain.chain_id,
                query=query,
                success_rate=success_rate,
                total_cost=total_cost,
                total_time=total_time,
                results=[r.output for r in results if r.success]
            )
            
        except Exception as e:
            print(f"Error executing chain: {e}")
            return None

    def resolve_predict(self, info, prediction_type, timeframe="1h"):
        """Resolve prediction query"""
        try:
            predictor = PredictiveAnalyticsEngine()
            
            # Get prediction
            prediction = predictor.get_prediction(prediction_type, timeframe)
            
            if prediction:
                return PredictionResult(
                    prediction_type=prediction_type,
                    value=prediction.get('value', 0.0),
                    confidence=prediction.get('confidence', 0.0),
                    timeframe=timeframe,
                    metadata=json.dumps(prediction.get('metadata', {}))
                )
            
            return None
            
        except Exception as e:
            print(f"Error getting prediction: {e}")
            return None


class ExecuteAgent(Mutation):
    """Execute agent mutation"""
    
    class Arguments:
        agent_id = String(required=True)
        query = String(required=True)
        system_message = String()
        temperature = Float(default_value=0.7)
        max_tokens = Int(default_value=1000)
    
    Output = AgentExecution
    
    def mutate(self, info, agent_id, query, system_message=None, temperature=0.7, max_tokens=1000):
        """Execute agent with query"""
        try:
            start_time = datetime.now()
            
            # Get model manager
            model_manager = AIModelManager()
            
            # Execute query
            result = model_manager.generate_response(
                agent_id,
                query,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result and 'response' in result:
                return AgentExecution(
                    agent_id=agent_id,
                    response=result['response'],
                    execution_time=execution_time,
                    tokens_used=result.get('tokens_used', 0),
                    cost=result.get('cost', 0.0),
                    success=True,
                    error_message=None,
                    metadata=json.dumps(result.get('metadata', {}))
                )
            else:
                return AgentExecution(
                    agent_id=agent_id,
                    response="",
                    execution_time=execution_time,
                    tokens_used=0,
                    cost=0.0,
                    success=False,
                    error_message="No response generated",
                    metadata="{}"
                )
                
        except Exception as e:
            return AgentExecution(
                agent_id=agent_id,
                response="",
                execution_time=0.0,
                tokens_used=0,
                cost=0.0,
                success=False,
                error_message=str(e),
                metadata="{}"
            )


class Mutation(ObjectType):
    """Main GraphQL Mutation type"""
    execute_agent = ExecuteAgent.Field()


# Create the GraphQL schema
schema = Schema(query=Query, mutation=Mutation)