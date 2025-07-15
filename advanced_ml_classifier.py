#!/usr/bin/env python3
"""
Advanced ML-Enhanced Query Classifier with DistilBERT Integration
Provides intelligent query classification with multiple ML models
"""

import asyncio
import json
import logging
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from enum import Enum
import re

# ML imports
try:
    from transformers import (
        DistilBertTokenizer, 
        DistilBertForSequenceClassification,
        pipeline
    )
    from sentence_transformers import SentenceTransformer, util
    import torch
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)


class QueryCategory(Enum):
    """Enhanced query categories"""
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    MATHEMATICAL = "mathematical"
    CODING = "coding"
    RESEARCH = "research"
    PHILOSOPHICAL = "philosophical"
    PRACTICAL = "practical"
    EDUCATIONAL = "educational"
    CONVERSATIONAL = "conversational"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    LEGAL = "legal"
    MEDICAL = "medical"
    ENTERTAINMENT = "entertainment"


class QueryIntent(Enum):
    """Query intent classification"""
    INFORMATION_SEEKING = "information_seeking"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_GENERATION = "creative_generation"
    ANALYSIS_REQUEST = "analysis_request"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    EXPLANATION = "explanation"
    STEP_BY_STEP = "step_by_step"
    VALIDATION = "validation"
    BRAINSTORMING = "brainstorming"


@dataclass
class QueryAnalysis:
    """Enhanced query analysis result"""
    text: str
    categories: List[QueryCategory]
    primary_category: QueryCategory
    confidence: float
    complexity: float
    required_capabilities: List[str]
    context_needed: bool
    multi_step: bool
    priority: int
    estimated_tokens: int
    intent: QueryIntent
    sentiment: str
    language: str
    technical_level: str
    domain_expertise: List[str]
    processing_time: float
    metadata: Dict[str, Any]


class AdvancedMLClassifier:
    """Advanced ML-Enhanced Query Classifier"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ml_model = None
        self.tokenizer = None
        self.similarity_model = None
        self.category_map = {}
        self.initialized = False
        
        # Enhanced category keywords with weights
        self.category_keywords = {
            QueryCategory.ANALYSIS: {
                "analyze": 1.0, "examine": 0.9, "investigate": 0.9, "pattern": 0.8,
                "trend": 0.8, "data": 0.7, "insight": 0.9, "metrics": 0.8,
                "statistics": 0.9, "evaluation": 0.8, "assessment": 0.8
            },
            QueryCategory.CREATIVE: {
                "create": 1.0, "write": 0.9, "imagine": 0.9, "design": 0.9,
                "story": 0.8, "poem": 0.8, "generate": 0.8, "invent": 0.8,
                "brainstorm": 0.9, "conceptualize": 0.8, "compose": 0.8
            },
            QueryCategory.TECHNICAL: {
                "technical": 1.0, "system": 0.8, "architecture": 0.9, "infrastructure": 0.9,
                "deploy": 0.8, "configure": 0.8, "optimize": 0.8, "performance": 0.8,
                "scalability": 0.9, "security": 0.8, "monitoring": 0.8
            },
            QueryCategory.MATHEMATICAL: {
                "calculate": 1.0, "solve": 0.9, "equation": 0.9, "formula": 0.9,
                "math": 0.8, "compute": 0.8, "derivative": 0.8, "integral": 0.8,
                "probability": 0.8, "statistics": 0.8, "algebra": 0.8
            },
            QueryCategory.CODING: {
                "code": 1.0, "program": 0.9, "function": 0.8, "debug": 0.9,
                "implement": 0.8, "algorithm": 0.9, "script": 0.8, "api": 0.8,
                "database": 0.8, "framework": 0.8, "library": 0.8
            },
            QueryCategory.RESEARCH: {
                "research": 1.0, "study": 0.9, "find": 0.7, "discover": 0.8,
                "investigate": 0.8, "source": 0.8, "literature": 0.9, "survey": 0.8,
                "methodology": 0.9, "hypothesis": 0.9, "evidence": 0.8
            },
            QueryCategory.BUSINESS: {
                "business": 1.0, "strategy": 0.9, "market": 0.8, "revenue": 0.8,
                "profit": 0.8, "investment": 0.8, "roi": 0.9, "budget": 0.8,
                "forecast": 0.8, "competitor": 0.8, "customer": 0.7
            },
            QueryCategory.SCIENTIFIC: {
                "experiment": 1.0, "hypothesis": 0.9, "theory": 0.9, "research": 0.8,
                "method": 0.8, "analysis": 0.8, "observation": 0.8, "conclusion": 0.8,
                "peer review": 0.9, "publication": 0.8, "laboratory": 0.8
            }
        }
        
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.INFORMATION_SEEKING: [
                r"what is", r"tell me about", r"explain", r"describe", r"define"
            ],
            QueryIntent.PROBLEM_SOLVING: [
                r"how to", r"solve", r"fix", r"resolve", r"troubleshoot"
            ],
            QueryIntent.CREATIVE_GENERATION: [
                r"create", r"generate", r"write", r"design", r"compose"
            ],
            QueryIntent.COMPARISON: [
                r"compare", r"versus", r"vs", r"difference", r"better"
            ],
            QueryIntent.RECOMMENDATION: [
                r"recommend", r"suggest", r"best", r"should i", r"which"
            ]
        }
        
        # Technical level indicators
        self.technical_indicators = {
            "beginner": ["simple", "basic", "easy", "introduction", "getting started"],
            "intermediate": ["advanced", "complex", "detailed", "comprehensive"],
            "expert": ["enterprise", "production", "optimization", "architecture", "scalable"]
        }
        
    async def initialize(self):
        """Initialize ML models and components"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, using enhanced keyword-based classification")
            self.initialized = True
            return
            
        try:
            # Initialize sentence transformer for semantic analysis
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Try to load custom model if available
            model_path = self.config.get('ml_model_path', './models/query_classifier')
            
            if os.path.exists(model_path):
                logger.info(f"Loading custom ML model from {model_path}")
                self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
                self.ml_model = DistilBertForSequenceClassification.from_pretrained(model_path)
                self.ml_model.eval()
                
                # Load category mapping
                mapping_path = os.path.join(model_path, "category_mapping.json")
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'r') as f:
                        self.category_map = json.load(f)
            else:
                logger.info("Custom model not found, using pre-trained DistilBERT")
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                self.ml_model = DistilBertForSequenceClassification.from_pretrained(
                    'distilbert-base-uncased',
                    num_labels=len(QueryCategory)
                )
                
            self.initialized = True
            logger.info("Advanced ML classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            self.initialized = True  # Continue with keyword-based classification
            
    async def classify_with_ml(self, query: str) -> Tuple[QueryCategory, float]:
        """Classify query using ML model"""
        if not self.initialized or not self.ml_model:
            return await self.classify_with_enhanced_keywords(query)
            
        try:
            # Tokenize and classify
            inputs = self.tokenizer(
                query, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.ml_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
            # Get prediction
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Map to QueryCategory
            if self.category_map:
                category_name = self.category_map.get(str(predicted_class), "CONVERSATIONAL")
                category = QueryCategory[category_name]
            else:
                category = list(QueryCategory)[predicted_class % len(QueryCategory)]
                
            return category, confidence
            
        except Exception as e:
            logger.error(f"ML classification error: {e}")
            return await self.classify_with_enhanced_keywords(query)
            
    async def classify_with_enhanced_keywords(self, query: str) -> Tuple[QueryCategory, float]:
        """Enhanced keyword-based classification with weights"""
        query_lower = query.lower()
        category_scores = defaultdict(float)
        
        for category, keywords in self.category_keywords.items():
            for keyword, weight in keywords.items():
                if keyword in query_lower:
                    # Position-based weighting
                    position = query_lower.find(keyword)
                    position_weight = 1.0 - (position / len(query_lower)) * 0.3
                    
                    # Frequency weighting
                    frequency = query_lower.count(keyword)
                    frequency_weight = min(frequency * 0.5, 1.0)
                    
                    # Context weighting
                    context_weight = self._get_context_weight(query_lower, keyword)
                    
                    final_score = weight * position_weight * frequency_weight * context_weight
                    category_scores[category] += final_score
                    
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            total_score = sum(category_scores.values())
            confidence = min(best_category[1] / total_score, 1.0) if total_score > 0 else 0.5
            return best_category[0], confidence
        else:
            return QueryCategory.CONVERSATIONAL, 0.5
            
    def _get_context_weight(self, query: str, keyword: str) -> float:
        """Calculate context-based weight for keyword"""
        keyword_index = query.find(keyword)
        if keyword_index == -1:
            return 1.0
            
        # Check surrounding context
        start = max(0, keyword_index - 20)
        end = min(len(query), keyword_index + len(keyword) + 20)
        context = query[start:end]
        
        # Boost weight if keyword appears in important contexts
        if any(marker in context for marker in ["how to", "please", "need to", "want to"]):
            return 1.3
        if any(marker in context for marker in ["not", "don't", "avoid"]):
            return 0.7
            
        return 1.0
        
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent using pattern matching"""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
                    
        # Default intent based on structure
        if "?" in query:
            return QueryIntent.INFORMATION_SEEKING
        elif any(word in query_lower for word in ["create", "make", "build", "generate"]):
            return QueryIntent.CREATIVE_GENERATION
        elif any(word in query_lower for word in ["compare", "better", "versus"]):
            return QueryIntent.COMPARISON
        else:
            return QueryIntent.INFORMATION_SEEKING
            
    def _calculate_complexity(self, query: str) -> float:
        """Calculate enhanced query complexity"""
        factors = {
            "length": min(len(query) / 1000, 1.0) * 0.15,
            "technical_terms": min(len(re.findall(r'\b[A-Z]{2,}\b', query)) / 8, 1.0) * 0.2,
            "nested_clauses": min(query.count(",") / 6, 1.0) * 0.15,
            "questions": min(query.count("?") / 4, 1.0) * 0.15,
            "code_blocks": min(query.count("```") / 3, 1.0) * 0.15,
            "numbers": min(len(re.findall(r'\d+', query)) / 15, 1.0) * 0.1,
            "special_chars": min(len(re.findall(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\?]', query)) / 20, 1.0) * 0.1
        }
        
        # Multi-step indicators
        if re.search(r'(and then|after that|followed by|subsequently|next|finally)', query, re.IGNORECASE):
            factors["multi_step"] = 0.2
            
        # Complex language patterns
        if re.search(r'(however|although|nevertheless|furthermore|moreover)', query, re.IGNORECASE):
            factors["complex_language"] = 0.15
            
        return min(sum(factors.values()), 1.0)
        
    def _extract_capabilities(self, query: str) -> List[str]:
        """Extract required capabilities from query"""
        capabilities = []
        
        capability_patterns = {
            "web_search": [r"search", r"find online", r"latest", r"current", r"recent"],
            "data_analysis": [r"analyze", r"statistics", r"data", r"chart", r"graph"],
            "code_execution": [r"run code", r"execute", r"compile", r"test"],
            "file_processing": [r"file", r"document", r"upload", r"download", r"parse"],
            "image_processing": [r"image", r"photo", r"picture", r"visual", r"graphic"],
            "math_computation": [r"calculate", r"compute", r"solve", r"equation", r"formula"],
            "language_translation": [r"translate", r"language", r"foreign", r"multilingual"],
            "api_integration": [r"api", r"integrate", r"connect", r"webhook", r"endpoint"],
            "database_query": [r"database", r"sql", r"query", r"table", r"record"],
            "real_time_data": [r"live", r"real-time", r"streaming", r"current", r"now"]
        }
        
        query_lower = query.lower()
        for capability, patterns in capability_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                capabilities.append(capability)
                
        return capabilities
        
    def _determine_technical_level(self, query: str) -> str:
        """Determine technical level of query"""
        query_lower = query.lower()
        
        scores = {"beginner": 0, "intermediate": 0, "expert": 0}
        
        for level, indicators in self.technical_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    scores[level] += 1
                    
        # Additional scoring based on complexity
        if len(query) > 200:
            scores["expert"] += 1
        elif len(query) > 100:
            scores["intermediate"] += 1
        else:
            scores["beginner"] += 1
            
        if re.search(r'[A-Z]{2,}', query):  # Acronyms
            scores["expert"] += 1
            
        return max(scores.items(), key=lambda x: x[1])[0]
        
    def _analyze_sentiment(self, query: str) -> str:
        """Basic sentiment analysis"""
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "please", "thank"]
        negative_words = ["bad", "terrible", "awful", "hate", "wrong", "error", "problem"]
        urgent_words = ["urgent", "asap", "immediately", "quick", "fast", "emergency"]
        
        query_lower = query.lower()
        
        positive_count = sum(1 for word in positive_words if word in query_lower)
        negative_count = sum(1 for word in negative_words if word in query_lower)
        urgent_count = sum(1 for word in urgent_words if word in query_lower)
        
        if urgent_count > 0:
            return "urgent"
        elif positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
            
    async def analyze(self, query: str, context: Optional[Dict] = None) -> QueryAnalysis:
        """Complete enhanced query analysis"""
        start_time = datetime.now()
        
        # Primary classification
        primary_category, confidence = await self.classify_with_ml(query)
        
        # Secondary classifications for multi-label support
        categories = [primary_category]
        if confidence < 0.8:  # Add secondary categories if primary confidence is low
            secondary_scores = defaultdict(float)
            for category, keywords in self.category_keywords.items():
                if category != primary_category:
                    for keyword, weight in keywords.items():
                        if keyword in query.lower():
                            secondary_scores[category] += weight
            
            if secondary_scores:
                secondary_category = max(secondary_scores.items(), key=lambda x: x[1])
                if secondary_category[1] > 0.3:  # Threshold for secondary category
                    categories.append(secondary_category[0])
        
        # Advanced analysis
        complexity = self._calculate_complexity(query)
        required_capabilities = self._extract_capabilities(query)
        multi_step = self._is_multi_step(query)
        priority = self._determine_priority(query, context)
        intent = self._detect_intent(query)
        sentiment = self._analyze_sentiment(query)
        technical_level = self._determine_technical_level(query)
        language = self._detect_language(query)
        domain_expertise = self._extract_domain_expertise(query)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryAnalysis(
            text=query,
            categories=categories,
            primary_category=primary_category,
            confidence=confidence,
            complexity=complexity,
            required_capabilities=required_capabilities,
            context_needed=bool(context) or "context" in query.lower(),
            multi_step=multi_step,
            priority=priority,
            estimated_tokens=len(query.split()) * 6,  # Enhanced token estimation
            intent=intent,
            sentiment=sentiment,
            language=language,
            technical_level=technical_level,
            domain_expertise=domain_expertise,
            processing_time=processing_time,
            metadata={
                "query_length": len(query),
                "word_count": len(query.split()),
                "has_code": "```" in query,
                "has_urls": bool(re.search(r'http[s]?://\S+', query)),
                "has_emails": bool(re.search(r'\S+@\S+', query)),
                "punctuation_density": len(re.findall(r'[.!?]', query)) / len(query) if query else 0,
                "question_count": query.count("?"),
                "exclamation_count": query.count("!"),
                "classification_method": "ml" if self.ml_model else "keywords"
            }
        )
        
    def _is_multi_step(self, query: str) -> bool:
        """Check if query requires multiple steps"""
        multi_step_indicators = [
            "and then", "after that", "followed by", "subsequently", "next", "finally",
            "first", "second", "third", "step 1", "step 2", "then", "afterwards"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in multi_step_indicators)
        
    def _determine_priority(self, query: str, context: Optional[Dict] = None) -> int:
        """Determine query priority (1-10)"""
        priority = 5  # Default priority
        
        query_lower = query.lower()
        
        # Urgency indicators
        urgent_words = ["urgent", "asap", "immediately", "emergency", "critical", "breaking"]
        if any(word in query_lower for word in urgent_words):
            priority += 3
            
        # Importance indicators
        important_words = ["important", "crucial", "vital", "essential", "key"]
        if any(word in query_lower for word in important_words):
            priority += 2
            
        # Complexity adjustment
        if len(query) > 500:
            priority += 1
            
        # Context-based adjustment
        if context and context.get("user_tier") == "premium":
            priority += 1
            
        return min(priority, 10)
        
    def _detect_language(self, query: str) -> str:
        """Basic language detection"""
        # Simple heuristic - can be enhanced with proper language detection
        if re.search(r'[а-яё]', query.lower()):
            return "ru"
        elif re.search(r'[一-龯]', query):
            return "zh"
        elif re.search(r'[ひらがなカタカナ]', query):
            return "ja"
        elif re.search(r'[가-힣]', query):
            return "ko"
        elif re.search(r'[àâäáãåçéèêëïíîìñóòôöõúùûüýÿ]', query.lower()):
            return "fr"
        elif re.search(r'[äöüß]', query.lower()):
            return "de"
        elif re.search(r'[ñáéíóúü]', query.lower()):
            return "es"
        else:
            return "en"
            
    def _extract_domain_expertise(self, query: str) -> List[str]:
        """Extract domain expertise areas"""
        domains = {
            "machine_learning": ["ml", "machine learning", "neural", "deep learning", "ai"],
            "web_development": ["html", "css", "javascript", "react", "nodejs", "web"],
            "data_science": ["data", "pandas", "numpy", "matplotlib", "analysis"],
            "cybersecurity": ["security", "encryption", "vulnerability", "penetration"],
            "cloud_computing": ["aws", "azure", "gcp", "cloud", "kubernetes", "docker"],
            "mobile_development": ["android", "ios", "mobile", "app", "swift", "kotlin"],
            "database": ["sql", "database", "mysql", "postgresql", "mongodb"],
            "devops": ["ci/cd", "jenkins", "git", "deployment", "automation"],
            "blockchain": ["blockchain", "cryptocurrency", "smart contract", "web3"],
            "finance": ["trading", "investment", "portfolio", "financial", "market"]
        }
        
        query_lower = query.lower()
        detected_domains = []
        
        for domain, keywords in domains.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_domains.append(domain)
                
        return detected_domains
        
    async def get_similar_queries(self, query: str, query_history: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar queries from history using semantic similarity"""
        if not self.similarity_model or not query_history:
            return []
            
        try:
            # Encode query and history
            query_embedding = self.similarity_model.encode(query)
            history_embeddings = self.similarity_model.encode(query_history)
            
            # Calculate similarities
            similarities = util.cos_sim(query_embedding, history_embeddings)[0]
            
            # Get top similar queries
            similar_queries = []
            for i, similarity in enumerate(similarities):
                if similarity > 0.7:  # Threshold for similarity
                    similar_queries.append((query_history[i], float(similarity)))
                    
            # Sort by similarity and return top k
            similar_queries.sort(key=lambda x: x[1], reverse=True)
            return similar_queries[:top_k]
            
        except Exception as e:
            logger.error(f"Error calculating query similarity: {e}")
            return []
            
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""
        return {
            "initialized": self.initialized,
            "ml_available": ML_AVAILABLE,
            "models_loaded": {
                "distilbert": self.ml_model is not None,
                "sentence_transformer": self.similarity_model is not None
            },
            "categories_count": len(QueryCategory),
            "intents_count": len(QueryIntent),
            "capability_patterns": len(self._extract_capabilities("")),
            "technical_levels": list(self.technical_indicators.keys())
        }