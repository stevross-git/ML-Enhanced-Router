"""
ML Router Service
Core machine learning query routing and classification service
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass

from flask import current_app
from ..models import QueryLog, AgentRegistration
from ..extensions import db
from ..utils.exceptions import ServiceError, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result of query processing"""
    query: str
    category: str
    confidence: float
    agent_id: Optional[str]
    agent_name: Optional[str]
    response: Any
    response_time: float
    tokens_used: int
    cost: float
    cache_hit: bool
    status: str
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class ClassificationResult:
    """Result of query classification"""
    category: str
    confidence: float
    subcategory: Optional[str] = None
    suggested_agents: List[str] = None
    metadata: Dict = None

class MLRouterService:
    """
    Main ML Router Service
    Handles query classification, routing, and response processing
    """
    
    def __init__(self):
        self.initialized = False
        self.ml_classifier = None
        self.total_queries = 0
        self.successful_routes = 0
        self.failed_routes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.avg_response_time = 0.0
        self.category_stats = {}
        
    async def initialize(self):
        """Initialize the ML router and classifier"""
        try:
            logger.info("🤖 Initializing ML Router Service...")
            
            # Initialize ML classifier
            await self._initialize_classifier()
            
            # Load agent registry
            await self._load_agents()
            
            # Initialize category statistics
            self._initialize_stats()
            
            self.initialized = True
            logger.info("✅ ML Router Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ ML Router initialization failed: {e}")
            raise ServiceError(f"Failed to initialize ML Router: {e}")
    
    async def _initialize_classifier(self):
        """Initialize the ML classification model"""
        try:
            # Import ML components (avoid circular imports)
            from transformers import pipeline
            
            # Load pre-trained classification model
            self.ml_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                return_all_scores=True
            )
            
            logger.info("🧠 ML classifier initialized")
            
        except ImportError:
            logger.warning("⚠️ Transformers not available, using fallback classifier")
            self.ml_classifier = self._create_