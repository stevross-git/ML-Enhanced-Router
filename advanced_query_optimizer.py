#!/usr/bin/env python3
"""
Advanced Query Optimizer with AI-powered query enhancement
Provides intelligent query optimization and enhancement capabilities
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import numpy as np

from advanced_ml_classifier import QueryAnalysis, QueryCategory, QueryIntent

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of query optimizations"""
    SEMANTIC_ENHANCEMENT = "semantic_enhancement"
    CONTEXT_EXPANSION = "context_expansion"
    QUERY_DECOMPOSITION = "query_decomposition"
    INTENT_CLARIFICATION = "intent_clarification"
    PARAMETER_EXTRACTION = "parameter_extraction"
    AMBIGUITY_RESOLUTION = "ambiguity_resolution"
    COMPLEXITY_REDUCTION = "complexity_reduction"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


@dataclass
class QueryOptimization:
    """Query optimization result"""
    original_query: str
    optimized_query: str
    optimization_type: OptimizationType
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    improvement_score: float = 0.0


@dataclass
class QueryEnhancement:
    """Query enhancement with multiple optimizations"""
    original_query: str
    enhanced_query: str
    optimizations: List[QueryOptimization]
    overall_confidence: float
    suggested_context: Dict[str, Any] = field(default_factory=dict)
    related_queries: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    complexity_reduction: float = 0.0


class SemanticEnhancer:
    """Semantic query enhancement"""
    
    def __init__(self):
        self.synonyms = {
            "analyze": ["examine", "investigate", "study", "evaluate", "assess"],
            "create": ["generate", "build", "make", "develop", "construct"],
            "explain": ["describe", "clarify", "elaborate", "detail", "illustrate"],
            "find": ["locate", "discover", "identify", "search", "retrieve"],
            "compare": ["contrast", "evaluate", "assess", "differentiate", "measure"],
            "solve": ["resolve", "fix", "address", "tackle", "handle"],
            "optimize": ["improve", "enhance", "refine", "streamline", "perfect"],
            "implement": ["deploy", "execute", "realize", "establish", "install"]
        }
        
        self.domain_expansions = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "nlp": "natural language processing",
            "ui": "user interface",
            "ux": "user experience",
            "api": "application programming interface",
            "db": "database",
            "sql": "structured query language",
            "http": "hypertext transfer protocol",
            "css": "cascading style sheets",
            "html": "hypertext markup language",
            "js": "javascript",
            "py": "python",
            "aws": "amazon web services",
            "gcp": "google cloud platform"
        }
        
    def enhance_query(self, query: str, analysis: QueryAnalysis) -> QueryOptimization:
        """Enhance query with semantic improvements"""
        start_time = time.time()
        
        enhanced_query = query
        changes = []
        
        # Expand abbreviations
        for abbr, full_form in self.domain_expansions.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                enhanced_query = re.sub(pattern, full_form, enhanced_query, flags=re.IGNORECASE)
                changes.append(f"Expanded '{abbr}' to '{full_form}'")
                
        # Add context-specific enhancements
        if analysis.primary_category == QueryCategory.TECHNICAL:
            enhanced_query = self._enhance_technical_query(enhanced_query, changes)
        elif analysis.primary_category == QueryCategory.CODING:
            enhanced_query = self._enhance_coding_query(enhanced_query, changes)
        elif analysis.primary_category == QueryCategory.CREATIVE:
            enhanced_query = self._enhance_creative_query(enhanced_query, changes)
            
        # Calculate improvement score
        improvement_score = len(changes) * 0.2 + (len(enhanced_query) - len(query)) / len(query) * 0.1
        
        return QueryOptimization(
            original_query=query,
            optimized_query=enhanced_query,
            optimization_type=OptimizationType.SEMANTIC_ENHANCEMENT,
            confidence=min(0.8, 0.5 + improvement_score),
            reasoning="; ".join(changes) if changes else "No semantic enhancements needed",
            processing_time=time.time() - start_time,
            improvement_score=improvement_score
        )
        
    def _enhance_technical_query(self, query: str, changes: List[str]) -> str:
        """Enhance technical queries"""
        technical_additions = {
            "performance": "performance optimization and monitoring",
            "scalability": "scalability and horizontal scaling",
            "security": "security best practices and implementation",
            "deployment": "deployment strategies and automation",
            "monitoring": "monitoring, logging, and observability"
        }
        
        query_lower = query.lower()
        for keyword, enhancement in technical_additions.items():
            if keyword in query_lower and enhancement not in query_lower:
                query = query.replace(keyword, enhancement)
                changes.append(f"Enhanced '{keyword}' with technical context")
                
        return query
        
    def _enhance_coding_query(self, query: str, changes: List[str]) -> str:
        """Enhance coding queries"""
        if "best practices" not in query.lower():
            query += " following best practices"
            changes.append("Added best practices requirement")
            
        if "error handling" not in query.lower() and any(word in query.lower() for word in ["function", "method", "code"]):
            query += " with proper error handling"
            changes.append("Added error handling requirement")
            
        return query
        
    def _enhance_creative_query(self, query: str, changes: List[str]) -> str:
        """Enhance creative queries"""
        if "original" not in query.lower() and "unique" not in query.lower():
            query += " with original and creative approach"
            changes.append("Added creativity requirement")
            
        return query


class ContextExpander:
    """Context expansion for queries"""
    
    def __init__(self):
        self.context_templates = {
            QueryCategory.TECHNICAL: {
                "environment": "production environment",
                "constraints": "performance and scalability constraints",
                "requirements": "enterprise requirements"
            },
            QueryCategory.CODING: {
                "language": "modern programming language",
                "framework": "popular frameworks",
                "testing": "unit testing and integration testing"
            },
            QueryCategory.BUSINESS: {
                "industry": "current industry trends",
                "metrics": "key performance indicators",
                "stakeholders": "stakeholder requirements"
            }
        }
        
    def expand_context(self, query: str, analysis: QueryAnalysis) -> QueryOptimization:
        """Expand query context"""
        start_time = time.time()
        
        expanded_query = query
        expansions = []
        
        # Add category-specific context
        if analysis.primary_category in self.context_templates:
            templates = self.context_templates[analysis.primary_category]
            
            for context_type, context_value in templates.items():
                if context_type not in query.lower():
                    expanded_query += f" considering {context_value}"
                    expansions.append(f"Added {context_type} context")
                    
        # Add complexity-based context
        if analysis.complexity > 0.7:
            expanded_query += " with step-by-step breakdown"
            expansions.append("Added complexity handling")
            
        # Add domain-specific context
        for domain in analysis.domain_expertise:
            if domain not in query.lower():
                expanded_query += f" with {domain} expertise"
                expansions.append(f"Added {domain} domain context")
                
        improvement_score = len(expansions) * 0.15
        
        return QueryOptimization(
            original_query=query,
            optimized_query=expanded_query,
            optimization_type=OptimizationType.CONTEXT_EXPANSION,
            confidence=min(0.9, 0.6 + improvement_score),
            reasoning="; ".join(expansions) if expansions else "No context expansion needed",
            processing_time=time.time() - start_time,
            improvement_score=improvement_score
        )


class QueryDecomposer:
    """Query decomposition for complex queries"""
    
    def __init__(self):
        self.decomposition_patterns = [
            r'(.+?)\s+and\s+(.+)',
            r'(.+?)\s+then\s+(.+)',
            r'(.+?)\s+followed\s+by\s+(.+)',
            r'(.+?)\s+as\s+well\s+as\s+(.+)',
            r'(.+?)\s+plus\s+(.+)',
            r'(.+?)\s+also\s+(.+)'
        ]
        
    def decompose_query(self, query: str, analysis: QueryAnalysis) -> QueryOptimization:
        """Decompose complex queries into subqueries"""
        start_time = time.time()
        
        if not analysis.multi_step or analysis.complexity < 0.5:
            return QueryOptimization(
                original_query=query,
                optimized_query=query,
                optimization_type=OptimizationType.QUERY_DECOMPOSITION,
                confidence=0.3,
                reasoning="Query doesn't need decomposition",
                processing_time=time.time() - start_time,
                improvement_score=0.0
            )
            
        subqueries = []
        decomposed = False
        
        for pattern in self.decomposition_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                subqueries = [match.group(1).strip(), match.group(2).strip()]
                decomposed = True
                break
                
        if not decomposed:
            # Try sentence splitting
            sentences = re.split(r'[.!?]+', query)
            if len(sentences) > 1:
                subqueries = [s.strip() for s in sentences if s.strip()]
                decomposed = True
                
        if decomposed and len(subqueries) > 1:
            # Create structured decomposition
            optimized_query = "Please address the following in order:\n"
            for i, subquery in enumerate(subqueries, 1):
                optimized_query += f"{i}. {subquery}\n"
                
            improvement_score = 0.3 + (len(subqueries) - 1) * 0.1
            
            return QueryOptimization(
                original_query=query,
                optimized_query=optimized_query.strip(),
                optimization_type=OptimizationType.QUERY_DECOMPOSITION,
                confidence=0.8,
                reasoning=f"Decomposed into {len(subqueries)} subqueries",
                processing_time=time.time() - start_time,
                improvement_score=improvement_score,
                metadata={"subqueries": subqueries}
            )
        else:
            return QueryOptimization(
                original_query=query,
                optimized_query=query,
                optimization_type=OptimizationType.QUERY_DECOMPOSITION,
                confidence=0.3,
                reasoning="Query couldn't be effectively decomposed",
                processing_time=time.time() - start_time,
                improvement_score=0.0
            )


class IntentClarifier:
    """Intent clarification for ambiguous queries"""
    
    def __init__(self):
        self.clarification_patterns = {
            QueryIntent.INFORMATION_SEEKING: [
                "Please provide comprehensive information about",
                "I need to understand",
                "Can you explain in detail"
            ],
            QueryIntent.PROBLEM_SOLVING: [
                "Help me solve this problem:",
                "I need a solution for",
                "How can I resolve"
            ],
            QueryIntent.CREATIVE_GENERATION: [
                "Please create",
                "Generate a creative",
                "Design and develop"
            ],
            QueryIntent.COMPARISON: [
                "Compare and contrast",
                "Analyze the differences between",
                "Evaluate the pros and cons of"
            ],
            QueryIntent.RECOMMENDATION: [
                "Please recommend",
                "What would you suggest for",
                "Advise me on the best"
            ]
        }
        
    def clarify_intent(self, query: str, analysis: QueryAnalysis) -> QueryOptimization:
        """Clarify query intent"""
        start_time = time.time()
        
        if analysis.confidence > 0.8:
            return QueryOptimization(
                original_query=query,
                optimized_query=query,
                optimization_type=OptimizationType.INTENT_CLARIFICATION,
                confidence=0.4,
                reasoning="Query intent is already clear",
                processing_time=time.time() - start_time,
                improvement_score=0.0
            )
            
        # Add intent-specific prefixes
        intent_patterns = self.clarification_patterns.get(analysis.intent, [])
        if intent_patterns:
            prefix = intent_patterns[0]  # Use first pattern
            
            # Check if query already has clear intent
            if not any(pattern.split()[0].lower() in query.lower() for pattern in intent_patterns):
                clarified_query = f"{prefix} {query.lower()}"
                
                improvement_score = 0.25
                
                return QueryOptimization(
                    original_query=query,
                    optimized_query=clarified_query,
                    optimization_type=OptimizationType.INTENT_CLARIFICATION,
                    confidence=0.7,
                    reasoning=f"Added {analysis.intent.value} intent clarification",
                    processing_time=time.time() - start_time,
                    improvement_score=improvement_score
                )
                
        return QueryOptimization(
            original_query=query,
            optimized_query=query,
            optimization_type=OptimizationType.INTENT_CLARIFICATION,
            confidence=0.3,
            reasoning="No intent clarification needed",
            processing_time=time.time() - start_time,
            improvement_score=0.0
        )


class ParameterExtractor:
    """Parameter extraction from queries"""
    
    def __init__(self):
        self.parameter_patterns = {
            "language": r'\b(python|javascript|java|c\+\+|c#|ruby|go|rust|swift|kotlin)\b',
            "framework": r'\b(react|angular|vue|django|flask|spring|express|laravel)\b',
            "database": r'\b(mysql|postgresql|mongodb|redis|sqlite|oracle)\b',
            "platform": r'\b(aws|azure|gcp|kubernetes|docker|heroku)\b',
            "version": r'\b(v?\d+\.?\d*\.?\d*)\b',
            "size": r'\b(\d+\s*(kb|mb|gb|tb|users|requests|items))\b',
            "time": r'\b(\d+\s*(seconds?|minutes?|hours?|days?|weeks?|months?))\b'
        }
        
    def extract_parameters(self, query: str, analysis: QueryAnalysis) -> QueryOptimization:
        """Extract and structure parameters from query"""
        start_time = time.time()
        
        extracted_params = {}
        
        for param_type, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                extracted_params[param_type] = matches
                
        if not extracted_params:
            return QueryOptimization(
                original_query=query,
                optimized_query=query,
                optimization_type=OptimizationType.PARAMETER_EXTRACTION,
                confidence=0.3,
                reasoning="No extractable parameters found",
                processing_time=time.time() - start_time,
                improvement_score=0.0
            )
            
        # Create structured query with parameters
        optimized_query = query + "\n\nExtracted parameters:\n"
        for param_type, values in extracted_params.items():
            optimized_query += f"- {param_type}: {', '.join(values)}\n"
            
        improvement_score = len(extracted_params) * 0.1
        
        return QueryOptimization(
            original_query=query,
            optimized_query=optimized_query,
            optimization_type=OptimizationType.PARAMETER_EXTRACTION,
            confidence=0.8,
            reasoning=f"Extracted {len(extracted_params)} parameter types",
            processing_time=time.time() - start_time,
            improvement_score=improvement_score,
            metadata={"extracted_parameters": extracted_params}
        )


class AmbiguityResolver:
    """Ambiguity resolution for unclear queries"""
    
    def __init__(self):
        self.ambiguous_terms = {
            "it": "the subject/object",
            "this": "the mentioned item",
            "that": "the referenced item",
            "these": "the mentioned items",
            "those": "the referenced items",
            "best": "most suitable/optimal",
            "good": "effective/appropriate",
            "fast": "high-performance/efficient",
            "simple": "straightforward/easy to implement",
            "complex": "sophisticated/advanced"
        }
        
    def resolve_ambiguity(self, query: str, analysis: QueryAnalysis) -> QueryOptimization:
        """Resolve ambiguous terms in query"""
        start_time = time.time()
        
        resolved_query = query
        resolutions = []
        
        query_words = query.lower().split()
        
        for ambiguous_term, clarification in self.ambiguous_terms.items():
            if ambiguous_term in query_words:
                # Add clarification in parentheses
                pattern = r'\b' + re.escape(ambiguous_term) + r'\b'
                resolved_query = re.sub(
                    pattern, 
                    f"{ambiguous_term} ({clarification})", 
                    resolved_query, 
                    flags=re.IGNORECASE
                )
                resolutions.append(f"Clarified '{ambiguous_term}' as '{clarification}'")
                
        if not resolutions:
            return QueryOptimization(
                original_query=query,
                optimized_query=query,
                optimization_type=OptimizationType.AMBIGUITY_RESOLUTION,
                confidence=0.4,
                reasoning="No ambiguous terms found",
                processing_time=time.time() - start_time,
                improvement_score=0.0
            )
            
        improvement_score = len(resolutions) * 0.15
        
        return QueryOptimization(
            original_query=query,
            optimized_query=resolved_query,
            optimization_type=OptimizationType.AMBIGUITY_RESOLUTION,
            confidence=0.7,
            reasoning="; ".join(resolutions),
            processing_time=time.time() - start_time,
            improvement_score=improvement_score
        )


class AdvancedQueryOptimizer:
    """Main advanced query optimizer"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.semantic_enhancer = SemanticEnhancer()
        self.context_expander = ContextExpander()
        self.query_decomposer = QueryDecomposer()
        self.intent_clarifier = IntentClarifier()
        self.parameter_extractor = ParameterExtractor()
        self.ambiguity_resolver = AmbiguityResolver()
        
        self.optimization_history: deque = deque(maxlen=1000)
        self.performance_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "average_improvement": 0.0,
            "optimization_types": defaultdict(int)
        }
        
    async def optimize_query(self, query: str, analysis: QueryAnalysis, 
                           optimization_types: Optional[List[OptimizationType]] = None) -> QueryEnhancement:
        """Perform comprehensive query optimization"""
        start_time = time.time()
        
        # Default optimization types
        if optimization_types is None:
            optimization_types = [
                OptimizationType.SEMANTIC_ENHANCEMENT,
                OptimizationType.CONTEXT_EXPANSION,
                OptimizationType.QUERY_DECOMPOSITION,
                OptimizationType.INTENT_CLARIFICATION,
                OptimizationType.PARAMETER_EXTRACTION,
                OptimizationType.AMBIGUITY_RESOLUTION
            ]
            
        optimizations = []
        current_query = query
        
        # Apply optimizations in order
        for opt_type in optimization_types:
            if opt_type == OptimizationType.SEMANTIC_ENHANCEMENT:
                optimization = self.semantic_enhancer.enhance_query(current_query, analysis)
            elif opt_type == OptimizationType.CONTEXT_EXPANSION:
                optimization = self.context_expander.expand_context(current_query, analysis)
            elif opt_type == OptimizationType.QUERY_DECOMPOSITION:
                optimization = self.query_decomposer.decompose_query(current_query, analysis)
            elif opt_type == OptimizationType.INTENT_CLARIFICATION:
                optimization = self.intent_clarifier.clarify_intent(current_query, analysis)
            elif opt_type == OptimizationType.PARAMETER_EXTRACTION:
                optimization = self.parameter_extractor.extract_parameters(current_query, analysis)
            elif opt_type == OptimizationType.AMBIGUITY_RESOLUTION:
                optimization = self.ambiguity_resolver.resolve_ambiguity(current_query, analysis)
            else:
                continue
                
            optimizations.append(optimization)
            
            # Update current query if optimization is significant
            if optimization.improvement_score > 0.1:
                current_query = optimization.optimized_query
                
        # Calculate overall metrics
        overall_confidence = sum(opt.confidence for opt in optimizations) / len(optimizations) if optimizations else 0.5
        total_improvement = sum(opt.improvement_score for opt in optimizations)
        
        # Generate suggested context
        suggested_context = self._generate_suggested_context(query, analysis, optimizations)
        
        # Generate related queries
        related_queries = self._generate_related_queries(query, analysis)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(query, current_query, analysis)
        
        # Calculate complexity reduction
        complexity_reduction = max(0, len(query) - len(current_query)) / len(query) if len(query) > 0 else 0
        
        enhancement = QueryEnhancement(
            original_query=query,
            enhanced_query=current_query,
            optimizations=optimizations,
            overall_confidence=overall_confidence,
            suggested_context=suggested_context,
            related_queries=related_queries,
            quality_score=quality_score,
            complexity_reduction=complexity_reduction
        )
        
        # Record optimization
        self._record_optimization(enhancement, time.time() - start_time)
        
        return enhancement
        
    def _generate_suggested_context(self, query: str, analysis: QueryAnalysis, 
                                  optimizations: List[QueryOptimization]) -> Dict[str, Any]:
        """Generate suggested context for the query"""
        context = {
            "user_level": analysis.technical_level,
            "domain": analysis.domain_expertise,
            "intent": analysis.intent.value,
            "complexity": analysis.complexity,
            "suggested_constraints": [],
            "suggested_requirements": []
        }
        
        # Add context based on category
        if analysis.primary_category == QueryCategory.TECHNICAL:
            context["suggested_constraints"].extend([
                "production environment",
                "scalability requirements",
                "performance benchmarks"
            ])
        elif analysis.primary_category == QueryCategory.CODING:
            context["suggested_constraints"].extend([
                "code quality standards",
                "testing requirements",
                "documentation needs"
            ])
        elif analysis.primary_category == QueryCategory.CREATIVE:
            context["suggested_requirements"].extend([
                "originality",
                "target audience",
                "creative constraints"
            ])
            
        return context
        
    def _generate_related_queries(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Generate related queries"""
        related = []
        
        # Generate variations based on intent
        if analysis.intent == QueryIntent.INFORMATION_SEEKING:
            related.append(f"What are the best practices for {query.lower()}?")
            related.append(f"What are common mistakes when {query.lower()}?")
        elif analysis.intent == QueryIntent.PROBLEM_SOLVING:
            related.append(f"What are alternative approaches to {query.lower()}?")
            related.append(f"How to troubleshoot issues with {query.lower()}?")
        elif analysis.intent == QueryIntent.CREATIVE_GENERATION:
            related.append(f"What are innovative ways to {query.lower()}?")
            related.append(f"What are examples of successful {query.lower()}?")
            
        # Generate category-specific variations
        if analysis.primary_category == QueryCategory.TECHNICAL:
            related.append(f"What are the performance implications of {query.lower()}?")
            related.append(f"How to scale {query.lower()} for enterprise use?")
        elif analysis.primary_category == QueryCategory.CODING:
            related.append(f"What are the testing strategies for {query.lower()}?")
            related.append(f"How to optimize {query.lower()} for better performance?")
            
        return related[:5]  # Return top 5 related queries
        
    def _calculate_quality_score(self, original: str, optimized: str, analysis: QueryAnalysis) -> float:
        """Calculate quality score for optimization"""
        score = 0.5  # Base score
        
        # Length improvement
        if len(optimized) > len(original):
            score += 0.1
            
        # Clarity improvement
        if "please" in optimized.lower() and "please" not in original.lower():
            score += 0.05
            
        # Structure improvement
        if "\n" in optimized and "\n" not in original:
            score += 0.1
            
        # Specificity improvement
        original_words = len(original.split())
        optimized_words = len(optimized.split())
        if optimized_words > original_words * 1.2:
            score += 0.15
            
        # Complexity handling
        if analysis.complexity > 0.7 and "step" in optimized.lower():
            score += 0.1
            
        return min(score, 1.0)
        
    def _record_optimization(self, enhancement: QueryEnhancement, processing_time: float):
        """Record optimization for analytics"""
        record = {
            "timestamp": datetime.now(),
            "original_length": len(enhancement.original_query),
            "optimized_length": len(enhancement.enhanced_query),
            "optimization_count": len(enhancement.optimizations),
            "overall_confidence": enhancement.overall_confidence,
            "quality_score": enhancement.quality_score,
            "processing_time": processing_time
        }
        
        self.optimization_history.append(record)
        
        # Update stats
        self.performance_stats["total_optimizations"] += 1
        if enhancement.quality_score > 0.6:
            self.performance_stats["successful_optimizations"] += 1
            
        # Update average improvement
        total_improvement = sum(record["quality_score"] for record in self.optimization_history)
        self.performance_stats["average_improvement"] = total_improvement / len(self.optimization_history)
        
        # Update optimization type stats
        for opt in enhancement.optimizations:
            self.performance_stats["optimization_types"][opt.optimization_type.value] += 1
            
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.optimization_history:
            return {"message": "No optimization history"}
            
        recent_optimizations = list(self.optimization_history)[-100:]  # Last 100
        
        return {
            "total_optimizations": self.performance_stats["total_optimizations"],
            "success_rate": (self.performance_stats["successful_optimizations"] / 
                           self.performance_stats["total_optimizations"]) * 100,
            "average_improvement": self.performance_stats["average_improvement"],
            "optimization_types": dict(self.performance_stats["optimization_types"]),
            "recent_performance": {
                "avg_processing_time": sum(r["processing_time"] for r in recent_optimizations) / len(recent_optimizations),
                "avg_quality_score": sum(r["quality_score"] for r in recent_optimizations) / len(recent_optimizations),
                "avg_length_increase": sum(r["optimized_length"] - r["original_length"] for r in recent_optimizations) / len(recent_optimizations)
            }
        }
        
    async def batch_optimize(self, queries: List[str], analyses: List[QueryAnalysis]) -> List[QueryEnhancement]:
        """Batch optimize multiple queries"""
        if len(queries) != len(analyses):
            raise ValueError("Number of queries and analyses must match")
            
        tasks = []
        for query, analysis in zip(queries, analyses):
            task = asyncio.create_task(self.optimize_query(query, analysis))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        return results
        
    def get_optimization_recommendations(self, query: str, analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """Get optimization recommendations for a query"""
        recommendations = []
        
        # Check for common issues
        if analysis.complexity > 0.8:
            recommendations.append({
                "type": "complexity",
                "title": "Consider Query Decomposition",
                "description": "This query is complex and might benefit from being broken down into smaller parts",
                "priority": "high"
            })
            
        if analysis.confidence < 0.6:
            recommendations.append({
                "type": "clarity",
                "title": "Add Intent Clarification",
                "description": "The query intent is unclear, consider adding clarifying language",
                "priority": "medium"
            })
            
        if len(query) < 20:
            recommendations.append({
                "type": "detail",
                "title": "Add More Context",
                "description": "Query is very short, consider adding more context and details",
                "priority": "medium"
            })
            
        if not any(word in query.lower() for word in ["please", "help", "how", "what", "why"]):
            recommendations.append({
                "type": "politeness",
                "title": "Add Polite Language",
                "description": "Consider using more polite and clear language",
                "priority": "low"
            })
            
        return recommendations