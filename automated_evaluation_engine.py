"""
Automated Evaluation Engine for ML Router
Self-testing system that benchmarks routing performance using synthetic and real user prompts
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import statistics
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCategory(Enum):
    """Test categories for evaluation"""
    PROGRAMMING = "programming"
    LEGAL = "legal"
    MEDICAL = "medical"
    GENERAL_KNOWLEDGE = "general_knowledge"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    TECHNICAL = "technical"
    MATHEMATICAL = "mathematical"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    PHILOSOPHICAL = "philosophical"
    EDUCATIONAL = "educational"


class TestType(Enum):
    """Types of tests to run"""
    ROUTING_ACCURACY = "routing_accuracy"
    SAFETY_CHECK = "safety_check"
    TOKEN_COST_AUDIT = "token_cost_audit"
    RESPONSE_QUALITY = "response_quality"
    LATENCY_TEST = "latency_test"
    LOAD_TEST = "load_test"
    CONSISTENCY_TEST = "consistency_test"


class SafetyLevel(Enum):
    """Safety levels for content evaluation"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    UNSAFE = "unsafe"


@dataclass
class SyntheticPrompt:
    """Synthetic test prompt data structure"""
    prompt: str
    category: TestCategory
    expected_agent: Optional[str] = None
    expected_complexity: Optional[float] = None
    expected_tokens: Optional[int] = None
    safety_level: SafetyLevel = SafetyLevel.SAFE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Test result data structure"""
    test_id: str
    test_type: TestType
    prompt: str
    category: TestCategory
    expected_result: Any
    actual_result: Any
    success: bool
    score: float
    execution_time: float
    cost: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    test_session_id: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float
    routing_accuracy: float
    safety_score: float
    cost_efficiency: float
    average_latency: float
    test_results: List[TestResult]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class SyntheticPromptGenerator:
    """Generates synthetic test prompts across categories"""
    
    def __init__(self):
        self.prompt_templates = {
            TestCategory.PROGRAMMING: [
                "Write a {language} function to {task}",
                "Debug this {language} code: {code_snippet}",
                "Explain the difference between {concept1} and {concept2} in {language}",
                "Create a {framework} application that {functionality}",
                "Optimize this algorithm for {optimization_goal}",
            ],
            TestCategory.LEGAL: [
                "What are the legal implications of {legal_scenario}?",
                "Explain {legal_concept} in simple terms",
                "What are the requirements for {legal_process}?",
                "How does {law_type} law differ between {jurisdiction1} and {jurisdiction2}?",
                "What are the key points in {legal_document_type}?",
            ],
            TestCategory.MEDICAL: [
                "What are the symptoms of {medical_condition}?",
                "Explain the treatment options for {medical_issue}",
                "What is the difference between {medical_term1} and {medical_term2}?",
                "How is {medical_procedure} performed?",
                "What are the side effects of {medication_class}?",
            ],
            TestCategory.GENERAL_KNOWLEDGE: [
                "What is {topic} and why is it important?",
                "Explain the history of {historical_event}",
                "How does {natural_phenomenon} work?",
                "What are the main features of {geographical_location}?",
                "Who was {historical_figure} and what did they accomplish?",
            ],
            TestCategory.CREATIVE: [
                "Write a {genre} story about {theme}",
                "Create a poem about {subject} in {style} style",
                "Design a {product_type} that {functionality}",
                "Generate ideas for {creative_project}",
                "Write a script for {media_type} about {topic}",
            ],
            TestCategory.ANALYSIS: [
                "Analyze the trends in {data_domain}",
                "Compare and contrast {item1} and {item2}",
                "What are the pros and cons of {decision_topic}?",
                "Evaluate the impact of {event} on {affected_area}",
                "Assess the effectiveness of {strategy} in {context}",
            ],
            TestCategory.TECHNICAL: [
                "How does {technology} work?",
                "What are the technical specifications of {system}?",
                "Explain the architecture of {technical_system}",
                "What are the best practices for {technical_process}?",
                "How do you troubleshoot {technical_issue}?",
            ],
            TestCategory.MATHEMATICAL: [
                "Solve this {math_type} problem: {problem}",
                "Explain the concept of {math_concept}",
                "What is the formula for {mathematical_calculation}?",
                "How do you calculate {mathematical_operation}?",
                "What are the applications of {mathematical_theory}?",
            ],
        }
        
        self.variables = {
            "language": ["Python", "JavaScript", "Java", "C++", "Go", "Rust"],
            "task": ["sort an array", "connect to database", "parse JSON", "handle errors"],
            "framework": ["React", "Django", "Spring", "Express", "Flask"],
            "legal_concept": ["contract law", "intellectual property", "tort law", "criminal law"],
            "medical_condition": ["diabetes", "hypertension", "asthma", "arthritis"],
            "topic": ["artificial intelligence", "climate change", "quantum computing", "blockchain"],
            "genre": ["science fiction", "mystery", "romance", "thriller"],
            "technology": ["blockchain", "machine learning", "cloud computing", "IoT"],
            "math_type": ["algebra", "calculus", "statistics", "geometry"],
        }
    
    def generate_prompts(self, category: TestCategory, count: int = 10) -> List[SyntheticPrompt]:
        """Generate synthetic prompts for a specific category"""
        prompts = []
        templates = self.prompt_templates.get(category, [])
        
        for _ in range(count):
            if not templates:
                continue
                
            template = random.choice(templates)
            prompt_text = self._fill_template(template)
            
            prompts.append(SyntheticPrompt(
                prompt=prompt_text,
                category=category,
                expected_complexity=random.uniform(0.1, 1.0),
                expected_tokens=random.randint(50, 500),
                safety_level=SafetyLevel.SAFE,
                metadata={"generated": True, "template": template}
            ))
        
        return prompts
    
    def _fill_template(self, template: str) -> str:
        """Fill template with random variables"""
        import re
        
        # Find all variables in template
        variables = re.findall(r'\{(\w+)\}', template)
        
        filled_template = template
        for var in variables:
            if var in self.variables:
                value = random.choice(self.variables[var])
                filled_template = filled_template.replace(f'{{{var}}}', value)
        
        return filled_template
    
    def generate_comprehensive_test_suite(self, prompts_per_category: int = 5) -> List[SyntheticPrompt]:
        """Generate a comprehensive test suite across all categories"""
        all_prompts = []
        
        for category in TestCategory:
            prompts = self.generate_prompts(category, prompts_per_category)
            all_prompts.extend(prompts)
        
        return all_prompts


class RoutingAccuracyEvaluator:
    """Evaluates routing accuracy using ML router"""
    
    def __init__(self, ml_router=None):
        self.ml_router = ml_router
    
    async def evaluate_routing(self, prompts: List[SyntheticPrompt]) -> List[TestResult]:
        """Evaluate routing accuracy for given prompts"""
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                start_time = time.time()
                
                # Get routing decision from ML router
                if self.ml_router:
                    routing_result = await self._get_routing_decision(prompt.prompt)
                else:
                    routing_result = {"agent": "fallback", "confidence": 0.5}
                
                execution_time = time.time() - start_time
                
                # Evaluate accuracy
                success = self._evaluate_routing_accuracy(prompt, routing_result)
                score = routing_result.get("confidence", 0.0) if success else 0.0
                
                results.append(TestResult(
                    test_id=f"routing_{i}",
                    test_type=TestType.ROUTING_ACCURACY,
                    prompt=prompt.prompt,
                    category=prompt.category,
                    expected_result=prompt.expected_agent,
                    actual_result=routing_result.get("agent"),
                    success=success,
                    score=score,
                    execution_time=execution_time,
                    cost=self._estimate_cost(prompt.prompt),
                    metadata={"routing_result": routing_result}
                ))
                
            except Exception as e:
                logger.error(f"Error evaluating routing for prompt {i}: {e}")
                results.append(TestResult(
                    test_id=f"routing_{i}",
                    test_type=TestType.ROUTING_ACCURACY,
                    prompt=prompt.prompt,
                    category=prompt.category,
                    expected_result=prompt.expected_agent,
                    actual_result=None,
                    success=False,
                    score=0.0,
                    execution_time=0.0,
                    cost=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    async def _get_routing_decision(self, prompt: str) -> Dict[str, Any]:
        """Get routing decision from ML router"""
        if not self.ml_router:
            return {"agent": "fallback", "confidence": 0.5}
        
        try:
            # Use the ML router to get routing decision
            result = await self.ml_router.route_query(prompt)
            return {
                "agent": result.get("agent_id", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "reasoning": result.get("reasoning", ""),
                "metadata": result.get("metadata", {})
            }
        except Exception as e:
            logger.error(f"Error getting routing decision: {e}")
            return {"agent": "error", "confidence": 0.0}
    
    def _evaluate_routing_accuracy(self, prompt: SyntheticPrompt, routing_result: Dict[str, Any]) -> bool:
        """Evaluate if routing decision is accurate"""
        # If expected agent is specified, check exact match
        if prompt.expected_agent:
            return routing_result.get("agent") == prompt.expected_agent
        
        # Otherwise, evaluate based on category appropriateness
        agent = routing_result.get("agent", "").lower()
        category = prompt.category.value.lower()
        
        # Simple heuristic matching
        category_agent_mapping = {
            "programming": ["code", "technical", "developer"],
            "legal": ["legal", "law", "attorney"],
            "medical": ["medical", "health", "doctor"],
            "creative": ["creative", "writer", "artist"],
            "analysis": ["analyst", "researcher", "data"],
            "technical": ["technical", "engineer", "tech"],
            "mathematical": ["math", "calculation", "numeric"],
        }
        
        expected_agents = category_agent_mapping.get(category, [])
        return any(expected in agent for expected in expected_agents)
    
    def _estimate_cost(self, prompt: str) -> float:
        """Estimate cost based on prompt length"""
        # Simple cost estimation based on token count
        estimated_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        cost_per_token = 0.0001  # Example cost per token
        return estimated_tokens * cost_per_token


class SafetyEvaluator:
    """Evaluates safety of prompts and responses"""
    
    def __init__(self):
        self.safety_patterns = {
            SafetyLevel.UNSAFE: [
                "illegal activities", "violence", "harmful content", "dangerous instructions",
                "hate speech", "inappropriate content", "malicious code", "security vulnerabilities"
            ],
            SafetyLevel.WARNING: [
                "sensitive information", "personal data", "financial advice", "medical advice",
                "legal advice", "controversial topics", "adult content"
            ],
            SafetyLevel.CAUTION: [
                "speculation", "unverified information", "opinion-based", "subjective content",
                "experimental", "beta features", "deprecated"
            ]
        }
    
    async def evaluate_safety(self, prompts: List[SyntheticPrompt]) -> List[TestResult]:
        """Evaluate safety of prompts"""
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                start_time = time.time()
                
                # Evaluate prompt safety
                safety_result = self._evaluate_prompt_safety(prompt.prompt)
                execution_time = time.time() - start_time
                
                # Check if safety level matches expected
                success = safety_result["level"] == prompt.safety_level
                score = safety_result["confidence"]
                
                results.append(TestResult(
                    test_id=f"safety_{i}",
                    test_type=TestType.SAFETY_CHECK,
                    prompt=prompt.prompt,
                    category=prompt.category,
                    expected_result=prompt.safety_level.value,
                    actual_result=safety_result["level"].value,
                    success=success,
                    score=score,
                    execution_time=execution_time,
                    cost=0.0,  # Safety checks are free
                    metadata=safety_result
                ))
                
            except Exception as e:
                logger.error(f"Error evaluating safety for prompt {i}: {e}")
                results.append(TestResult(
                    test_id=f"safety_{i}",
                    test_type=TestType.SAFETY_CHECK,
                    prompt=prompt.prompt,
                    category=prompt.category,
                    expected_result=prompt.safety_level.value,
                    actual_result=None,
                    success=False,
                    score=0.0,
                    execution_time=0.0,
                    cost=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    def _evaluate_prompt_safety(self, prompt: str) -> Dict[str, Any]:
        """Evaluate safety level of a prompt"""
        prompt_lower = prompt.lower()
        
        # Check for unsafe patterns
        for pattern in self.safety_patterns[SafetyLevel.UNSAFE]:
            if pattern in prompt_lower:
                return {
                    "level": SafetyLevel.UNSAFE,
                    "confidence": 0.9,
                    "reason": f"Contains unsafe pattern: {pattern}",
                    "patterns_found": [pattern]
                }
        
        # Check for warning patterns
        warning_patterns = []
        for pattern in self.safety_patterns[SafetyLevel.WARNING]:
            if pattern in prompt_lower:
                warning_patterns.append(pattern)
        
        if warning_patterns:
            return {
                "level": SafetyLevel.WARNING,
                "confidence": 0.8,
                "reason": f"Contains warning patterns: {', '.join(warning_patterns)}",
                "patterns_found": warning_patterns
            }
        
        # Check for caution patterns
        caution_patterns = []
        for pattern in self.safety_patterns[SafetyLevel.CAUTION]:
            if pattern in prompt_lower:
                caution_patterns.append(pattern)
        
        if caution_patterns:
            return {
                "level": SafetyLevel.CAUTION,
                "confidence": 0.7,
                "reason": f"Contains caution patterns: {', '.join(caution_patterns)}",
                "patterns_found": caution_patterns
            }
        
        # Default to safe
        return {
            "level": SafetyLevel.SAFE,
            "confidence": 0.9,
            "reason": "No safety concerns detected",
            "patterns_found": []
        }


class TokenCostAuditor:
    """Audits token costs and efficiency"""
    
    def __init__(self):
        self.token_costs = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3": 0.015,
            "gemini-pro": 0.001,
            "local": 0.0
        }
    
    async def audit_costs(self, prompts: List[SyntheticPrompt]) -> List[TestResult]:
        """Audit token costs for prompts"""
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                start_time = time.time()
                
                # Estimate token usage and cost
                cost_analysis = self._analyze_token_cost(prompt.prompt)
                execution_time = time.time() - start_time
                
                # Evaluate cost efficiency
                expected_cost = prompt.expected_tokens * 0.002 if prompt.expected_tokens else 0.01
                actual_cost = cost_analysis["total_cost"]
                
                success = actual_cost <= expected_cost * 1.2  # 20% tolerance
                score = min(1.0, expected_cost / actual_cost) if actual_cost > 0 else 1.0
                
                results.append(TestResult(
                    test_id=f"cost_{i}",
                    test_type=TestType.TOKEN_COST_AUDIT,
                    prompt=prompt.prompt,
                    category=prompt.category,
                    expected_result=expected_cost,
                    actual_result=actual_cost,
                    success=success,
                    score=score,
                    execution_time=execution_time,
                    cost=actual_cost,
                    metadata=cost_analysis
                ))
                
            except Exception as e:
                logger.error(f"Error auditing cost for prompt {i}: {e}")
                results.append(TestResult(
                    test_id=f"cost_{i}",
                    test_type=TestType.TOKEN_COST_AUDIT,
                    prompt=prompt.prompt,
                    category=prompt.category,
                    expected_result=0.0,
                    actual_result=0.0,
                    success=False,
                    score=0.0,
                    execution_time=0.0,
                    cost=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    def _analyze_token_cost(self, prompt: str) -> Dict[str, Any]:
        """Analyze token cost for a prompt"""
        # Estimate token count (rough approximation)
        words = prompt.split()
        estimated_tokens = len(words) * 1.3  # Rough conversion
        
        # Estimate costs for different models
        model_costs = {}
        for model, cost_per_token in self.token_costs.items():
            model_costs[model] = estimated_tokens * cost_per_token
        
        # Choose most cost-effective model
        best_model = min(model_costs.items(), key=lambda x: x[1])
        
        return {
            "estimated_tokens": estimated_tokens,
            "model_costs": model_costs,
            "best_model": best_model[0],
            "total_cost": best_model[1],
            "cost_per_token": best_model[1] / estimated_tokens if estimated_tokens > 0 else 0
        }


class AutomatedEvaluationEngine:
    """Main evaluation engine that coordinates all tests"""
    
    def __init__(self, db_path: str = "evaluation_results.db"):
        self.db_path = db_path
        self.prompt_generator = SyntheticPromptGenerator()
        self.routing_evaluator = RoutingAccuracyEvaluator()
        self.safety_evaluator = SafetyEvaluator()
        self.cost_auditor = TokenCostAuditor()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_sessions (
                session_id TEXT PRIMARY KEY,
                timestamp DATETIME,
                total_tests INTEGER,
                passed_tests INTEGER,
                failed_tests INTEGER,
                overall_score REAL,
                routing_accuracy REAL,
                safety_score REAL,
                cost_efficiency REAL,
                average_latency REAL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                test_id TEXT PRIMARY KEY,
                session_id TEXT,
                test_type TEXT,
                category TEXT,
                prompt TEXT,
                expected_result TEXT,
                actual_result TEXT,
                success BOOLEAN,
                score REAL,
                execution_time REAL,
                cost REAL,
                error_message TEXT,
                metadata TEXT,
                timestamp DATETIME,
                FOREIGN KEY (session_id) REFERENCES evaluation_sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def run_comprehensive_evaluation(self, 
                                         prompts_per_category: int = 5,
                                         include_real_prompts: bool = True) -> EvaluationReport:
        """Run comprehensive evaluation across all test types"""
        session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting comprehensive evaluation session: {session_id}")
        
        # Generate synthetic prompts
        synthetic_prompts = self.prompt_generator.generate_comprehensive_test_suite(prompts_per_category)
        
        # Add real prompts if available
        if include_real_prompts:
            real_prompts = await self._get_real_user_prompts()
            synthetic_prompts.extend(real_prompts)
        
        # Run all evaluations
        all_results = []
        
        # Routing accuracy evaluation
        logger.info("Running routing accuracy evaluation...")
        routing_results = await self.routing_evaluator.evaluate_routing(synthetic_prompts)
        all_results.extend(routing_results)
        
        # Safety evaluation
        logger.info("Running safety evaluation...")
        safety_results = await self.safety_evaluator.evaluate_safety(synthetic_prompts)
        all_results.extend(safety_results)
        
        # Cost audit
        logger.info("Running token cost audit...")
        cost_results = await self.cost_auditor.audit_costs(synthetic_prompts)
        all_results.extend(cost_results)
        
        # Generate report
        report = self._generate_evaluation_report(session_id, all_results)
        
        # Store results
        await self._store_evaluation_results(report)
        
        logger.info(f"Evaluation completed. Overall score: {report.overall_score:.2f}")
        
        return report
    
    async def _get_real_user_prompts(self) -> List[SyntheticPrompt]:
        """Get real user prompts from query logs"""
        try:
            # This would connect to your actual query logs
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Error getting real user prompts: {e}")
            return []
    
    def _generate_evaluation_report(self, session_id: str, results: List[TestResult]) -> EvaluationReport:
        """Generate comprehensive evaluation report"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        
        # Calculate category-specific scores
        routing_results = [r for r in results if r.test_type == TestType.ROUTING_ACCURACY]
        safety_results = [r for r in results if r.test_type == TestType.SAFETY_CHECK]
        cost_results = [r for r in results if r.test_type == TestType.TOKEN_COST_AUDIT]
        
        routing_accuracy = statistics.mean([r.score for r in routing_results]) if routing_results else 0.0
        safety_score = statistics.mean([r.score for r in safety_results]) if safety_results else 0.0
        cost_efficiency = statistics.mean([r.score for r in cost_results]) if cost_results else 0.0
        
        # Calculate overall metrics
        overall_score = statistics.mean([r.score for r in results]) if results else 0.0
        average_latency = statistics.mean([r.execution_time for r in results]) if results else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        return EvaluationReport(
            test_session_id=session_id,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            overall_score=overall_score,
            routing_accuracy=routing_accuracy,
            safety_score=safety_score,
            cost_efficiency=cost_efficiency,
            average_latency=average_latency,
            test_results=results,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, results: List[TestResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Routing accuracy recommendations
        routing_results = [r for r in results if r.test_type == TestType.ROUTING_ACCURACY]
        if routing_results:
            failed_routing = [r for r in routing_results if not r.success]
            if len(failed_routing) > len(routing_results) * 0.2:  # >20% failure rate
                recommendations.append("Consider improving routing accuracy - high failure rate detected")
        
        # Safety recommendations
        safety_results = [r for r in results if r.test_type == TestType.SAFETY_CHECK]
        if safety_results:
            unsafe_results = [r for r in safety_results if not r.success]
            if unsafe_results:
                recommendations.append("Review safety filters - some unsafe content detected")
        
        # Cost efficiency recommendations
        cost_results = [r for r in results if r.test_type == TestType.TOKEN_COST_AUDIT]
        if cost_results:
            avg_cost = statistics.mean([r.cost for r in cost_results])
            if avg_cost > 0.05:  # Arbitrary threshold
                recommendations.append("Consider optimizing token usage to reduce costs")
        
        # Latency recommendations
        avg_latency = statistics.mean([r.execution_time for r in results])
        if avg_latency > 2.0:  # >2 seconds
            recommendations.append("Consider optimizing response times - high latency detected")
        
        return recommendations
    
    async def _store_evaluation_results(self, report: EvaluationReport):
        """Store evaluation results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Store session info
            cursor.execute('''
                INSERT INTO evaluation_sessions 
                (session_id, timestamp, total_tests, passed_tests, failed_tests, 
                 overall_score, routing_accuracy, safety_score, cost_efficiency, 
                 average_latency, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report.test_session_id,
                report.timestamp,
                report.total_tests,
                report.passed_tests,
                report.failed_tests,
                report.overall_score,
                report.routing_accuracy,
                report.safety_score,
                report.cost_efficiency,
                report.average_latency,
                json.dumps({"recommendations": report.recommendations})
            ))
            
            # Store individual test results
            for result in report.test_results:
                cursor.execute('''
                    INSERT INTO test_results 
                    (test_id, session_id, test_type, category, prompt, expected_result, 
                     actual_result, success, score, execution_time, cost, error_message, 
                     metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.test_id,
                    report.test_session_id,
                    result.test_type.value,
                    result.category.value,
                    result.prompt,
                    str(result.expected_result),
                    str(result.actual_result),
                    result.success,
                    result.score,
                    result.execution_time,
                    result.cost,
                    result.error_message,
                    json.dumps(result.metadata),
                    result.timestamp
                ))
            
            conn.commit()
            logger.info(f"Stored evaluation results for session {report.test_session_id}")
            
        except Exception as e:
            logger.error(f"Error storing evaluation results: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def schedule_periodic_evaluation(self, interval_hours: int = 24):
        """Schedule periodic evaluation runs"""
        logger.info(f"Scheduling periodic evaluation every {interval_hours} hours")
        
        while True:
            try:
                report = await self.run_comprehensive_evaluation()
                logger.info(f"Periodic evaluation completed. Score: {report.overall_score:.2f}")
                
                # Check if action is needed based on score
                if report.overall_score < 0.7:  # Below 70%
                    logger.warning("Low evaluation score detected - manual review recommended")
                
                await asyncio.sleep(interval_hours * 3600)  # Convert to seconds
                
            except Exception as e:
                logger.error(f"Error in periodic evaluation: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    def get_evaluation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get evaluation history from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT session_id, timestamp, total_tests, passed_tests, failed_tests,
                       overall_score, routing_accuracy, safety_score, cost_efficiency,
                       average_latency, metadata
                FROM evaluation_sessions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    "session_id": row[0],
                    "timestamp": row[1],
                    "total_tests": row[2],
                    "passed_tests": row[3],
                    "failed_tests": row[4],
                    "overall_score": row[5],
                    "routing_accuracy": row[6],
                    "safety_score": row[7],
                    "cost_efficiency": row[8],
                    "average_latency": row[9],
                    "metadata": json.loads(row[10]) if row[10] else {}
                })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting evaluation history: {e}")
            return []
        finally:
            conn.close()
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get session stats
            cursor.execute('''
                SELECT COUNT(*) as total_sessions,
                       AVG(overall_score) as avg_score,
                       MAX(overall_score) as max_score,
                       MIN(overall_score) as min_score,
                       AVG(routing_accuracy) as avg_routing_accuracy,
                       AVG(safety_score) as avg_safety_score,
                       AVG(cost_efficiency) as avg_cost_efficiency,
                       AVG(average_latency) as avg_latency
                FROM evaluation_sessions
            ''')
            
            session_stats = cursor.fetchone()
            
            # Get test type distribution
            cursor.execute('''
                SELECT test_type, COUNT(*) as count, AVG(score) as avg_score
                FROM test_results
                GROUP BY test_type
            ''')
            
            test_type_stats = {row[0]: {"count": row[1], "avg_score": row[2]} 
                             for row in cursor.fetchall()}
            
            # Get category distribution
            cursor.execute('''
                SELECT category, COUNT(*) as count, AVG(score) as avg_score
                FROM test_results
                GROUP BY category
            ''')
            
            category_stats = {row[0]: {"count": row[1], "avg_score": row[2]} 
                            for row in cursor.fetchall()}
            
            return {
                "session_stats": {
                    "total_sessions": session_stats[0],
                    "avg_score": session_stats[1],
                    "max_score": session_stats[2],
                    "min_score": session_stats[3],
                    "avg_routing_accuracy": session_stats[4],
                    "avg_safety_score": session_stats[5],
                    "avg_cost_efficiency": session_stats[6],
                    "avg_latency": session_stats[7]
                },
                "test_type_stats": test_type_stats,
                "category_stats": category_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting evaluation stats: {e}")
            return {}
        finally:
            conn.close()


# Global instance
evaluation_engine = None

def get_evaluation_engine():
    """Get or create global evaluation engine instance"""
    global evaluation_engine
    if evaluation_engine is None:
        evaluation_engine = AutomatedEvaluationEngine()
    return evaluation_engine


async def run_evaluation_demo():
    """Demo function to show evaluation capabilities"""
    engine = get_evaluation_engine()
    
    logger.info("Running evaluation demo...")
    
    # Run comprehensive evaluation
    report = await engine.run_comprehensive_evaluation(prompts_per_category=3)
    
    # Print summary
    print(f"\nEvaluation Report Summary:")
    print(f"Session ID: {report.test_session_id}")
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed: {report.passed_tests}")
    print(f"Failed: {report.failed_tests}")
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"Routing Accuracy: {report.routing_accuracy:.2f}")
    print(f"Safety Score: {report.safety_score:.2f}")
    print(f"Cost Efficiency: {report.cost_efficiency:.2f}")
    print(f"Average Latency: {report.average_latency:.2f}s")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"- {rec}")


if __name__ == "__main__":
    asyncio.run(run_evaluation_demo())