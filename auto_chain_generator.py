"""
Auto Chain Generator / Composer
Dynamically generates multi-step agent chains per query
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from ai_models import AIModelManager, AIModel
from rag_chat import SimpleRAGChat
from collaborative_router import CollaborativeRouter
from advanced_ml_classifier import AdvancedMLClassifier

logger = logging.getLogger(__name__)

class ChainStepType(Enum):
    """Types of chain steps"""
    RAG_RETRIEVAL = "rag_retrieval"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    CRITIQUE = "critique"
    DEBATE = "debate"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    TRANSLATION = "translation"
    EXTRACTION = "extraction"
    COMPARISON = "comparison"
    RESEARCH = "research"
    PLANNING = "planning"
    EXECUTION = "execution"

@dataclass
class ChainStep:
    """Individual step in an agent chain"""
    step_id: str
    step_type: ChainStepType
    agent_id: str
    agent_name: str
    description: str
    input_from: Optional[str] = None  # Previous step ID
    parameters: Dict[str, Any] = None
    expected_output: str = ""
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class AgentChain:
    """Complete agent chain configuration"""
    chain_id: str
    query: str
    steps: List[ChainStep]
    estimated_cost: float = 0.0
    estimated_time: float = 0.0
    complexity_score: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ChainExecutionResult:
    """Result of executing a chain step"""
    step_id: str
    success: bool
    output: str
    execution_time: float
    tokens_used: int
    cost: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class AutoChainGenerator:
    """Automatically generates multi-step agent chains based on query analysis"""
    
    def __init__(self):
        self.model_manager = AIModelManager()
        self.rag_chat = None
        self.collaborative_router = None
        self.classifier = None
        
        # Initialize components with error handling
        try:
            self.rag_chat = SimpleRAGChat()
        except Exception as e:
            logger.warning(f"RAG Chat not available: {e}")
            
        try:
            self.collaborative_router = CollaborativeRouter()
        except Exception as e:
            logger.warning(f"Collaborative Router not available: {e}")
            
        try:
            self.classifier = AdvancedMLClassifier()
        except Exception as e:
            logger.warning(f"Advanced ML Classifier not available: {e}")
        
        # Chain templates for common patterns
        self.chain_templates = {
            "summarize_and_debate": [
                ChainStepType.RAG_RETRIEVAL,
                ChainStepType.CLASSIFICATION,
                ChainStepType.SUMMARIZATION,
                ChainStepType.DEBATE,
                ChainStepType.CRITIQUE
            ],
            "research_and_analyze": [
                ChainStepType.RAG_RETRIEVAL,
                ChainStepType.RESEARCH,
                ChainStepType.ANALYSIS,
                ChainStepType.SYNTHESIS
            ],
            "creative_generation": [
                ChainStepType.CLASSIFICATION,
                ChainStepType.PLANNING,
                ChainStepType.GENERATION,
                ChainStepType.CRITIQUE,
                ChainStepType.VALIDATION
            ],
            "document_processing": [
                ChainStepType.RAG_RETRIEVAL,
                ChainStepType.EXTRACTION,
                ChainStepType.SUMMARIZATION,
                ChainStepType.ANALYSIS
            ],
            "compare_and_contrast": [
                ChainStepType.RAG_RETRIEVAL,
                ChainStepType.EXTRACTION,
                ChainStepType.COMPARISON,
                ChainStepType.SYNTHESIS
            ],
            "translation_and_critique": [
                ChainStepType.TRANSLATION,
                ChainStepType.VALIDATION,
                ChainStepType.CRITIQUE
            ]
        }
        
        # Agent specializations
        self.agent_specializations = {
            ChainStepType.RAG_RETRIEVAL: ["rag_specialist"],
            ChainStepType.CLASSIFICATION: ["classifier", "analyst"],
            ChainStepType.SUMMARIZATION: ["summarizer", "analyst"],
            ChainStepType.ANALYSIS: ["analyst", "technical", "researcher"],
            ChainStepType.GENERATION: ["creative", "technical"],
            ChainStepType.CRITIQUE: ["critic", "analyst"],
            ChainStepType.DEBATE: ["debater", "critic"],
            ChainStepType.SYNTHESIS: ["synthesizer", "analyst"],
            ChainStepType.VALIDATION: ["validator", "technical"],
            ChainStepType.TRANSLATION: ["translator", "multilingual"],
            ChainStepType.EXTRACTION: ["extractor", "analyst"],
            ChainStepType.COMPARISON: ["comparator", "analyst"],
            ChainStepType.RESEARCH: ["researcher", "analyst"],
            ChainStepType.PLANNING: ["planner", "strategic"],
            ChainStepType.EXECUTION: ["executor", "technical"]
        }
        
        logger.info("Auto Chain Generator initialized")
    
    def analyze_query_for_chain(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine optimal chain composition"""
        try:
            analysis = {
                "query": query,
                "complexity": self._calculate_complexity(query),
                "intent": self._detect_intent(query),
                "domain": self._detect_domain(query),
                "requires_rag": self._requires_rag(query),
                "requires_debate": self._requires_debate(query),
                "requires_research": self._requires_research(query),
                "requires_creativity": self._requires_creativity(query),
                "output_format": self._detect_output_format(query),
                "estimated_steps": self._estimate_steps(query)
            }
            
            logger.info(f"Query analysis completed for: {query[:50]}...")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {
                "query": query,
                "complexity": 0.5,
                "intent": "general",
                "domain": "general",
                "requires_rag": False,
                "requires_debate": False,
                "requires_research": False,
                "requires_creativity": False,
                "output_format": "text",
                "estimated_steps": 3
            }
    
    def generate_chain(self, query: str) -> AgentChain:
        """Generate optimal agent chain for the given query"""
        try:
            # Analyze the query
            analysis = self.analyze_query_for_chain(query)
            
            # Determine chain template
            template = self._select_chain_template(analysis)
            
            # Generate chain steps
            steps = self._generate_chain_steps(query, template, analysis)
            
            # Calculate estimates
            estimated_cost = sum(step.parameters.get('estimated_cost', 0.01) for step in steps)
            estimated_time = sum(step.parameters.get('estimated_time', 2.0) for step in steps)
            
            # Create chain
            chain = AgentChain(
                chain_id=f"chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                query=query,
                steps=steps,
                estimated_cost=estimated_cost,
                estimated_time=estimated_time,
                complexity_score=analysis['complexity']
            )
            
            logger.info(f"Generated chain with {len(steps)} steps for query: {query[:50]}...")
            return chain
            
        except Exception as e:
            logger.error(f"Error generating chain: {e}")
            # Return basic fallback chain
            return self._create_fallback_chain(query)
    
    def _select_chain_template(self, analysis: Dict[str, Any]) -> List[ChainStepType]:
        """Select appropriate chain template based on analysis"""
        query_lower = analysis['query'].lower()
        
        # Pattern matching for specific templates
        if any(word in query_lower for word in ['summarize', 'debate', 'argue', 'counter']):
            return self.chain_templates['summarize_and_debate']
        elif any(word in query_lower for word in ['research', 'analyze', 'investigate']):
            return self.chain_templates['research_and_analyze']
        elif any(word in query_lower for word in ['create', 'generate', 'write', 'compose']):
            return self.chain_templates['creative_generation']
        elif any(word in query_lower for word in ['compare', 'contrast', 'difference']):
            return self.chain_templates['compare_and_contrast']
        elif any(word in query_lower for word in ['translate', 'convert']):
            return self.chain_templates['translation_and_critique']
        elif analysis['requires_rag']:
            return self.chain_templates['document_processing']
        else:
            # Default adaptive template
            template = [ChainStepType.CLASSIFICATION]
            
            if analysis['requires_rag']:
                template.insert(0, ChainStepType.RAG_RETRIEVAL)
            
            if analysis['requires_research']:
                template.append(ChainStepType.RESEARCH)
            
            if analysis['requires_creativity']:
                template.append(ChainStepType.GENERATION)
            else:
                template.append(ChainStepType.ANALYSIS)
            
            if analysis['requires_debate']:
                template.append(ChainStepType.DEBATE)
            
            template.append(ChainStepType.SYNTHESIS)
            
            return template
    
    def _generate_chain_steps(self, query: str, template: List[ChainStepType], analysis: Dict[str, Any]) -> List[ChainStep]:
        """Generate specific chain steps from template"""
        steps = []
        
        for i, step_type in enumerate(template):
            # Select appropriate agent
            agent_id, agent_name = self._select_agent_for_step(step_type, analysis)
            
            # Determine input source
            input_from = steps[-1].step_id if steps else None
            
            # Create step
            step = ChainStep(
                step_id=f"step_{i+1}_{step_type.value}",
                step_type=step_type,
                agent_id=agent_id,
                agent_name=agent_name,
                description=self._generate_step_description(step_type, query, analysis),
                input_from=input_from,
                parameters=self._generate_step_parameters(step_type, query, analysis),
                expected_output=self._generate_expected_output(step_type, query)
            )
            
            steps.append(step)
        
        return steps
    
    def _select_agent_for_step(self, step_type: ChainStepType, analysis: Dict[str, Any]) -> Tuple[str, str]:
        """Select the best agent for a specific step type"""
        try:
            # Get available models
            models = self.model_manager.get_all_models()
            
            # Get specializations for this step type
            preferred_specializations = self.agent_specializations.get(step_type, ["general"])
            
            # Find best matching agent
            best_agent = None
            best_score = 0
            
            for model in models:
                if hasattr(model, 'id'):
                    # Calculate compatibility score
                    score = self._calculate_agent_compatibility(model, step_type, analysis)
                    
                    if score > best_score:
                        best_score = score
                        best_agent = model
            
            if best_agent:
                return best_agent.id, best_agent.name
            else:
                # Fallback to first available model
                if models and hasattr(models[0], 'id'):
                    return models[0].id, models[0].name
                else:
                    return "gpt-4o", "GPT-4o (Multi-modal)"
                    
        except Exception as e:
            logger.error(f"Error selecting agent for step {step_type}: {e}")
            return "gpt-4o", "GPT-4o (Multi-modal)"
    
    def _calculate_agent_compatibility(self, model: AIModel, step_type: ChainStepType, analysis: Dict[str, Any]) -> float:
        """Calculate how well an agent fits a specific step type"""
        score = 0.5  # Base score
        
        # Check model capabilities
        if hasattr(model, 'capabilities'):
            capabilities = [cap.value for cap in model.capabilities]
            
            # RAG steps prefer models with good reasoning
            if step_type == ChainStepType.RAG_RETRIEVAL and 'reasoning' in capabilities:
                score += 0.3
            
            # Creative steps prefer models with generation capabilities
            if step_type in [ChainStepType.GENERATION, ChainStepType.DEBATE] and 'text_generation' in capabilities:
                score += 0.2
            
            # Analysis steps prefer reasoning capabilities
            if step_type in [ChainStepType.ANALYSIS, ChainStepType.CLASSIFICATION] and 'reasoning' in capabilities:
                score += 0.2
            
            # Multimodal steps prefer multimodal models
            if step_type in [ChainStepType.EXTRACTION, ChainStepType.ANALYSIS] and 'multimodal' in capabilities:
                score += 0.1
        
        # Consider cost efficiency
        if hasattr(model, 'cost_per_1k_tokens'):
            if model.cost_per_1k_tokens < 0.01:  # Low cost models
                score += 0.1
        
        # Consider context window for complex tasks
        if hasattr(model, 'context_window'):
            if analysis['complexity'] > 0.7 and model.context_window > 8000:
                score += 0.1
        
        return min(score, 1.0)
    
    def _generate_step_description(self, step_type: ChainStepType, query: str, analysis: Dict[str, Any]) -> str:
        """Generate description for a chain step"""
        descriptions = {
            ChainStepType.RAG_RETRIEVAL: f"Retrieve relevant documents and context for: {query}",
            ChainStepType.CLASSIFICATION: f"Classify and categorize the query: {query}",
            ChainStepType.SUMMARIZATION: f"Summarize the key points and findings",
            ChainStepType.ANALYSIS: f"Analyze the content and provide insights",
            ChainStepType.GENERATION: f"Generate new content based on the analysis",
            ChainStepType.CRITIQUE: f"Provide critical evaluation and feedback",
            ChainStepType.DEBATE: f"Present counter-arguments and alternative perspectives",
            ChainStepType.SYNTHESIS: f"Synthesize all findings into a coherent response",
            ChainStepType.VALIDATION: f"Validate the results for accuracy and completeness",
            ChainStepType.TRANSLATION: f"Translate content as requested",
            ChainStepType.EXTRACTION: f"Extract key information and data points",
            ChainStepType.COMPARISON: f"Compare and contrast different elements",
            ChainStepType.RESEARCH: f"Conduct additional research on the topic",
            ChainStepType.PLANNING: f"Plan the approach and strategy",
            ChainStepType.EXECUTION: f"Execute the planned actions"
        }
        
        return descriptions.get(step_type, f"Process the query: {query}")
    
    def _generate_step_parameters(self, step_type: ChainStepType, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for a chain step"""
        base_params = {
            'temperature': 0.7,
            'max_tokens': 1000,
            'estimated_cost': 0.01,
            'estimated_time': 2.0
        }
        
        # Step-specific parameters
        if step_type == ChainStepType.RAG_RETRIEVAL:
            base_params.update({
                'max_documents': 5,
                'similarity_threshold': 0.7,
                'estimated_cost': 0.005
            })
        elif step_type == ChainStepType.CLASSIFICATION:
            base_params.update({
                'temperature': 0.3,
                'max_tokens': 500,
                'estimated_cost': 0.005
            })
        elif step_type == ChainStepType.GENERATION:
            base_params.update({
                'temperature': 0.8,
                'max_tokens': 2000,
                'estimated_cost': 0.02
            })
        elif step_type == ChainStepType.CRITIQUE:
            base_params.update({
                'temperature': 0.6,
                'max_tokens': 1500,
                'estimated_cost': 0.015
            })
        elif step_type == ChainStepType.DEBATE:
            base_params.update({
                'temperature': 0.7,
                'max_tokens': 1500,
                'estimated_cost': 0.015
            })
        
        return base_params
    
    def _generate_expected_output(self, step_type: ChainStepType, query: str) -> str:
        """Generate expected output description for a step"""
        outputs = {
            ChainStepType.RAG_RETRIEVAL: "Retrieved documents and relevant context",
            ChainStepType.CLASSIFICATION: "Query category and classification metadata",
            ChainStepType.SUMMARIZATION: "Concise summary of key points",
            ChainStepType.ANALYSIS: "Detailed analysis and insights",
            ChainStepType.GENERATION: "Generated content or response",
            ChainStepType.CRITIQUE: "Critical evaluation and feedback",
            ChainStepType.DEBATE: "Counter-arguments and alternative perspectives",
            ChainStepType.SYNTHESIS: "Synthesized final response",
            ChainStepType.VALIDATION: "Validation results and accuracy assessment",
            ChainStepType.TRANSLATION: "Translated content",
            ChainStepType.EXTRACTION: "Extracted key information",
            ChainStepType.COMPARISON: "Comparison analysis",
            ChainStepType.RESEARCH: "Research findings and data",
            ChainStepType.PLANNING: "Strategic plan and approach",
            ChainStepType.EXECUTION: "Execution results"
        }
        
        return outputs.get(step_type, "Processed output")
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score"""
        complexity = 0.3  # Base complexity
        
        # Length factor
        if len(query) > 100:
            complexity += 0.2
        
        # Keyword indicators
        complex_keywords = ['analyze', 'compare', 'debate', 'research', 'synthesize', 'critique']
        for keyword in complex_keywords:
            if keyword in query.lower():
                complexity += 0.1
        
        # Question complexity
        if '?' in query:
            complexity += 0.1
        
        return min(complexity, 1.0)
    
    def _detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['summarize', 'summary']):
            return 'summarization'
        elif any(word in query_lower for word in ['analyze', 'analysis']):
            return 'analysis'
        elif any(word in query_lower for word in ['create', 'generate', 'write']):
            return 'generation'
        elif any(word in query_lower for word in ['compare', 'contrast']):
            return 'comparison'
        elif any(word in query_lower for word in ['translate']):
            return 'translation'
        elif any(word in query_lower for word in ['research', 'find', 'search']):
            return 'research'
        elif any(word in query_lower for word in ['debate', 'argue']):
            return 'debate'
        else:
            return 'general'
    
    def _detect_domain(self, query: str) -> str:
        """Detect the domain of the query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['code', 'programming', 'software']):
            return 'technical'
        elif any(word in query_lower for word in ['business', 'market', 'strategy']):
            return 'business'
        elif any(word in query_lower for word in ['science', 'research', 'study']):
            return 'scientific'
        elif any(word in query_lower for word in ['creative', 'story', 'poem']):
            return 'creative'
        elif any(word in query_lower for word in ['legal', 'law', 'regulation']):
            return 'legal'
        elif any(word in query_lower for word in ['medical', 'health', 'medicine']):
            return 'medical'
        else:
            return 'general'
    
    def _requires_rag(self, query: str) -> bool:
        """Determine if query requires RAG retrieval"""
        rag_indicators = ['document', 'file', 'paper', 'article', 'this', 'these', 'reference']
        return any(indicator in query.lower() for indicator in rag_indicators)
    
    def _requires_debate(self, query: str) -> bool:
        """Determine if query requires debate/counter-arguments"""
        debate_indicators = ['debate', 'argue', 'counter', 'opposite', 'alternative', 'pros and cons']
        return any(indicator in query.lower() for indicator in debate_indicators)
    
    def _requires_research(self, query: str) -> bool:
        """Determine if query requires additional research"""
        research_indicators = ['research', 'investigate', 'explore', 'find out', 'discover']
        return any(indicator in query.lower() for indicator in research_indicators)
    
    def _requires_creativity(self, query: str) -> bool:
        """Determine if query requires creative generation"""
        creative_indicators = ['create', 'generate', 'write', 'compose', 'design', 'imagine']
        return any(indicator in query.lower() for indicator in creative_indicators)
    
    def _detect_output_format(self, query: str) -> str:
        """Detect expected output format"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['json', 'structured']):
            return 'json'
        elif any(word in query_lower for word in ['markdown', 'md']):
            return 'markdown'
        elif any(word in query_lower for word in ['html']):
            return 'html'
        elif any(word in query_lower for word in ['list', 'bullet']):
            return 'list'
        elif any(word in query_lower for word in ['table']):
            return 'table'
        else:
            return 'text'
    
    def _estimate_steps(self, query: str) -> int:
        """Estimate number of steps needed"""
        base_steps = 3
        
        if self._requires_rag(query):
            base_steps += 1
        if self._requires_debate(query):
            base_steps += 2
        if self._requires_research(query):
            base_steps += 1
        if len(query) > 200:
            base_steps += 1
        
        return min(base_steps, 8)  # Cap at 8 steps
    
    def _create_fallback_chain(self, query: str) -> AgentChain:
        """Create a simple fallback chain"""
        steps = [
            ChainStep(
                step_id="step_1_analysis",
                step_type=ChainStepType.ANALYSIS,
                agent_id="gpt-4o",
                agent_name="GPT-4o (Multi-modal)",
                description=f"Analyze the query: {query}",
                parameters={'temperature': 0.7, 'max_tokens': 1000}
            ),
            ChainStep(
                step_id="step_2_synthesis",
                step_type=ChainStepType.SYNTHESIS,
                agent_id="gpt-4o",
                agent_name="GPT-4o (Multi-modal)",
                description="Synthesize the analysis into a response",
                input_from="step_1_analysis",
                parameters={'temperature': 0.7, 'max_tokens': 1000}
            )
        ]
        
        return AgentChain(
            chain_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query=query,
            steps=steps,
            estimated_cost=0.02,
            estimated_time=4.0,
            complexity_score=0.5
        )
    
    async def execute_chain(self, chain: AgentChain) -> List[ChainExecutionResult]:
        """Execute a complete agent chain"""
        results = []
        context = {}
        
        logger.info(f"Starting execution of chain {chain.chain_id} with {len(chain.steps)} steps")
        
        for i, step in enumerate(chain.steps):
            try:
                logger.info(f"Executing step {i+1}/{len(chain.steps)}: {step.step_id}")
                
                # Prepare input
                input_text = chain.query
                if step.input_from and step.input_from in context:
                    input_text = context[step.input_from]
                
                # Execute step
                result = await self._execute_step(step, input_text)
                
                # Store result
                results.append(result)
                context[step.step_id] = result.output
                
                logger.info(f"Step {step.step_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Error executing step {step.step_id}: {e}")
                
                # Create error result
                error_result = ChainExecutionResult(
                    step_id=step.step_id,
                    success=False,
                    output="",
                    execution_time=0.0,
                    tokens_used=0,
                    cost=0.0,
                    error_message=str(e)
                )
                results.append(error_result)
                
                # Stop execution on critical errors
                if step.step_type in [ChainStepType.RAG_RETRIEVAL, ChainStepType.CLASSIFICATION]:
                    break
        
        logger.info(f"Chain {chain.chain_id} execution completed")
        return results
    
    async def _execute_step(self, step: ChainStep, input_text: str) -> ChainExecutionResult:
        """Execute a single chain step"""
        start_time = datetime.now()
        
        try:
            # Get the agent
            agent = self.model_manager.get_model_by_id(step.agent_id)
            if not agent:
                raise ValueError(f"Agent {step.agent_id} not found")
            
            # Prepare the prompt based on step type
            prompt = self._prepare_step_prompt(step, input_text)
            
            # Execute based on step type
            if step.step_type == ChainStepType.RAG_RETRIEVAL:
                output = await self._execute_rag_step(input_text, step.parameters)
            else:
                output = await self._execute_ai_step(agent, prompt, step.parameters)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Estimate tokens and cost
            tokens_used = len(output.split()) * 1.3  # Rough estimate
            cost = tokens_used * step.parameters.get('estimated_cost', 0.01) / 1000
            
            return ChainExecutionResult(
                step_id=step.step_id,
                success=True,
                output=output,
                execution_time=execution_time,
                tokens_used=int(tokens_used),
                cost=cost,
                metadata={'step_type': step.step_type.value}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ChainExecutionResult(
                step_id=step.step_id,
                success=False,
                output="",
                execution_time=execution_time,
                tokens_used=0,
                cost=0.0,
                error_message=str(e)
            )
    
    def _prepare_step_prompt(self, step: ChainStep, input_text: str) -> str:
        """Prepare prompt for a specific step type"""
        base_prompt = f"Task: {step.description}\n\nInput: {input_text}\n\n"
        
        if step.step_type == ChainStepType.CLASSIFICATION:
            return base_prompt + "Classify this query and provide category, intent, and complexity analysis."
        elif step.step_type == ChainStepType.SUMMARIZATION:
            return base_prompt + "Provide a concise summary of the key points."
        elif step.step_type == ChainStepType.ANALYSIS:
            return base_prompt + "Analyze this content and provide detailed insights."
        elif step.step_type == ChainStepType.GENERATION:
            return base_prompt + "Generate new content based on the analysis."
        elif step.step_type == ChainStepType.CRITIQUE:
            return base_prompt + "Provide critical evaluation and identify potential issues."
        elif step.step_type == ChainStepType.DEBATE:
            return base_prompt + "Present counter-arguments and alternative perspectives."
        elif step.step_type == ChainStepType.SYNTHESIS:
            return base_prompt + "Synthesize all the previous findings into a coherent response."
        else:
            return base_prompt + "Process this content according to the task description."
    
    async def _execute_rag_step(self, query: str, parameters: Dict[str, Any]) -> str:
        """Execute RAG retrieval step"""
        try:
            if self.rag_chat:
                # Use RAG system to retrieve relevant documents
                response = await self.rag_chat.get_response(query, parameters.get('max_documents', 5))
                return response
            else:
                return f"RAG retrieval for: {query} (RAG system not available)"
        except Exception as e:
            logger.error(f"RAG execution error: {e}")
            return f"RAG retrieval failed: {str(e)}"
    
    async def _execute_ai_step(self, agent: AIModel, prompt: str, parameters: Dict[str, Any]) -> str:
        """Execute AI model step"""
        try:
            # This is a simplified execution - in practice, you'd use the actual AI model
            # For demonstration, we'll return a structured response
            return f"AI Response from {agent.name}: {prompt[:200]}... (processed with {parameters.get('temperature', 0.7)} temperature)"
        except Exception as e:
            logger.error(f"AI execution error: {e}")
            return f"AI execution failed: {str(e)}"
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get statistics about chain generation"""
        return {
            "available_templates": len(self.chain_templates),
            "step_types": len(ChainStepType),
            "components_available": {
                "rag_chat": self.rag_chat is not None,
                "collaborative_router": self.collaborative_router is not None,
                "classifier": self.classifier is not None
            },
            "model_count": len(self.model_manager.get_all_models())
        }