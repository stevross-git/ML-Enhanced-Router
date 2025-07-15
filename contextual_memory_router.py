#!/usr/bin/env python3
"""
Contextual Memory Routing System
Implements vector-based routing with memory of past successful routes
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import sqlite3
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for ML dependencies
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    logger.warning("Vector libraries not available, using fallback implementations")

class RoutingStrategy(Enum):
    """Routing strategies for contextual memory"""
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class RoutingMemory:
    """Memory of successful routing decisions"""
    query: str
    query_embedding: List[float]
    agent_id: str
    success_score: float
    response_time: float
    user_satisfaction: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextualRoute:
    """Contextual routing decision"""
    query: str
    recommended_agent: str
    confidence: float
    similarity_score: float
    matched_memories: List[RoutingMemory]
    reasoning: str
    fallback_agents: List[str] = field(default_factory=list)
    estimated_success: float = 0.0

class VectorMemoryStore:
    """Vector-based memory storage for routing decisions"""
    
    def __init__(self, collection_name: str = "routing_memory"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedder = None
        
        if VECTOR_AVAILABLE:
            self._initialize_vector_store()
        else:
            logger.warning("Vector store not available, using fallback storage")
            self._initialize_fallback_store()
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_routing_memory"
            ))
            
            # Try to get existing collection or create new one
            try:
                self.collection = self.client.get_collection(self.collection_name)
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Routing memory for contextual decisions"}
                )
            
            # Initialize embedder
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info(f"Vector memory store initialized with {self.collection.count()} memories")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            VECTOR_AVAILABLE = False
            self._initialize_fallback_store()
    
    def _initialize_fallback_store(self):
        """Initialize fallback SQLite store"""
        self.db_path = "routing_memory.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS routing_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                success_score REAL NOT NULL,
                response_time REAL NOT NULL,
                user_satisfaction REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                context TEXT,
                metadata TEXT,
                UNIQUE(query_hash, agent_id)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_query_hash ON routing_memory(query_hash)
        ''')
        
        conn.commit()
        conn.close()
    
    async def store_memory(self, memory: RoutingMemory):
        """Store a routing memory"""
        try:
            if VECTOR_AVAILABLE and self.collection:
                # Store in vector database
                memory_id = f"{hash(memory.query)}_{memory.agent_id}_{int(memory.timestamp.timestamp())}"
                
                self.collection.upsert(
                    ids=[memory_id],
                    embeddings=[memory.query_embedding],
                    metadatas=[{
                        "agent_id": memory.agent_id,
                        "success_score": memory.success_score,
                        "response_time": memory.response_time,
                        "user_satisfaction": memory.user_satisfaction,
                        "timestamp": memory.timestamp.isoformat(),
                        "context": json.dumps(memory.context),
                        "metadata": json.dumps(memory.metadata)
                    }],
                    documents=[memory.query]
                )
            else:
                # Store in SQLite fallback
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                query_hash = str(hash(memory.query))
                cursor.execute('''
                    INSERT OR REPLACE INTO routing_memory (
                        query, query_hash, agent_id, success_score,
                        response_time, user_satisfaction, timestamp,
                        context, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory.query,
                    query_hash,
                    memory.agent_id,
                    memory.success_score,
                    memory.response_time,
                    memory.user_satisfaction,
                    memory.timestamp,
                    json.dumps(memory.context),
                    json.dumps(memory.metadata)
                ))
                
                conn.commit()
                conn.close()
                
            logger.debug(f"Stored routing memory for query: {memory.query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error storing routing memory: {e}")
    
    async def find_similar_memories(self, query: str, top_k: int = 5) -> List[RoutingMemory]:
        """Find similar routing memories"""
        try:
            if VECTOR_AVAILABLE and self.collection and self.embedder:
                # Vector-based search
                query_embedding = self.embedder.encode(query).tolist()
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
                memories = []
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    memory = RoutingMemory(
                        query=doc,
                        query_embedding=query_embedding,
                        agent_id=metadata['agent_id'],
                        success_score=metadata['success_score'],
                        response_time=metadata['response_time'],
                        user_satisfaction=metadata['user_satisfaction'],
                        timestamp=datetime.fromisoformat(metadata['timestamp']),
                        context=json.loads(metadata['context']),
                        metadata=json.loads(metadata['metadata'])
                    )
                    memories.append(memory)
                
                return memories
            else:
                # Fallback keyword search
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Simple keyword matching
                query_words = query.lower().split()
                placeholders = ' OR '.join(['query LIKE ?'] * len(query_words))
                search_terms = [f'%{word}%' for word in query_words]
                
                cursor.execute(f'''
                    SELECT query, agent_id, success_score, response_time,
                           user_satisfaction, timestamp, context, metadata
                    FROM routing_memory
                    WHERE {placeholders}
                    ORDER BY success_score DESC, user_satisfaction DESC
                    LIMIT ?
                ''', search_terms + [top_k])
                
                memories = []
                for row in cursor.fetchall():
                    memory = RoutingMemory(
                        query=row[0],
                        query_embedding=[],
                        agent_id=row[1],
                        success_score=row[2],
                        response_time=row[3],
                        user_satisfaction=row[4],
                        timestamp=datetime.fromisoformat(row[5]),
                        context=json.loads(row[6]),
                        metadata=json.loads(row[7])
                    )
                    memories.append(memory)
                
                conn.close()
                return memories
                
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        try:
            if VECTOR_AVAILABLE and self.collection:
                count = self.collection.count()
                return {
                    "total_memories": count,
                    "store_type": "vector",
                    "vector_available": True
                }
            else:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM routing_memory')
                count = cursor.fetchone()[0]
                
                cursor.execute('SELECT agent_id, COUNT(*) FROM routing_memory GROUP BY agent_id')
                agent_counts = dict(cursor.fetchall())
                
                conn.close()
                
                return {
                    "total_memories": count,
                    "store_type": "fallback",
                    "vector_available": False,
                    "agent_distribution": agent_counts
                }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"total_memories": 0, "store_type": "error"}

class ContextualMemoryRouter:
    """Main contextual memory routing system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.memory_store = VectorMemoryStore()
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.memory_weight = self.config.get('memory_weight', 0.6)
        self.recency_weight = self.config.get('recency_weight', 0.2)
        self.success_weight = self.config.get('success_weight', 0.2)
        
        # Statistics
        self.total_routes = 0
        self.memory_hits = 0
        self.successful_routes = 0
    
    async def route_with_memory(self, query: str, context: Dict[str, Any] = None,
                               strategy: RoutingStrategy = RoutingStrategy.HYBRID) -> ContextualRoute:
        """Route query using contextual memory"""
        self.total_routes += 1
        
        try:
            # Find similar memories
            similar_memories = await self.memory_store.find_similar_memories(query, top_k=10)
            
            if not similar_memories:
                return ContextualRoute(
                    query=query,
                    recommended_agent="default",
                    confidence=0.1,
                    similarity_score=0.0,
                    matched_memories=[],
                    reasoning="No similar memories found, using default routing"
                )
            
            # Score and rank agents based on memory
            agent_scores = {}
            for memory in similar_memories:
                agent_id = memory.agent_id
                
                # Calculate similarity score (placeholder for actual implementation)
                similarity = self._calculate_similarity(query, memory.query)
                
                # Calculate recency score
                hours_ago = (datetime.now() - memory.timestamp).total_seconds() / 3600
                recency_score = max(0, 1 - (hours_ago / 168))  # Decay over a week
                
                # Combined score
                score = (
                    similarity * self.memory_weight +
                    recency_score * self.recency_weight +
                    memory.success_score * self.success_weight
                )
                
                if agent_id not in agent_scores:
                    agent_scores[agent_id] = {
                        'score': 0,
                        'memories': [],
                        'avg_similarity': 0,
                        'total_success': 0
                    }
                
                agent_scores[agent_id]['score'] += score
                agent_scores[agent_id]['memories'].append(memory)
                agent_scores[agent_id]['avg_similarity'] += similarity
                agent_scores[agent_id]['total_success'] += memory.success_score
            
            # Normalize scores
            for agent_id in agent_scores:
                memory_count = len(agent_scores[agent_id]['memories'])
                agent_scores[agent_id]['avg_similarity'] /= memory_count
                agent_scores[agent_id]['total_success'] /= memory_count
            
            # Select best agent
            best_agent = max(agent_scores.keys(), key=lambda x: agent_scores[x]['score'])
            best_score = agent_scores[best_agent]['score']
            
            # Create fallback list
            fallback_agents = sorted(
                [aid for aid in agent_scores.keys() if aid != best_agent],
                key=lambda x: agent_scores[x]['score'],
                reverse=True
            )[:3]
            
            self.memory_hits += 1
            
            return ContextualRoute(
                query=query,
                recommended_agent=best_agent,
                confidence=min(best_score, 1.0),
                similarity_score=agent_scores[best_agent]['avg_similarity'],
                matched_memories=agent_scores[best_agent]['memories'],
                reasoning=f"Found {len(similar_memories)} similar memories, "
                          f"best match: {best_agent} with score {best_score:.3f}",
                fallback_agents=fallback_agents,
                estimated_success=agent_scores[best_agent]['total_success']
            )
            
        except Exception as e:
            logger.error(f"Error in contextual routing: {e}")
            return ContextualRoute(
                query=query,
                recommended_agent="default",
                confidence=0.1,
                similarity_score=0.0,
                matched_memories=[],
                reasoning=f"Error in routing: {e}"
            )
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries (placeholder)"""
        # Simple word overlap similarity
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def record_routing_outcome(self, query: str, agent_id: str,
                                   success_score: float, response_time: float,
                                   user_satisfaction: float = 0.8,
                                   context: Dict[str, Any] = None):
        """Record outcome of routing decision"""
        try:
            # Create embedding if vector store available
            query_embedding = []
            if VECTOR_AVAILABLE and self.memory_store.embedder:
                query_embedding = self.memory_store.embedder.encode(query).tolist()
            
            memory = RoutingMemory(
                query=query,
                query_embedding=query_embedding,
                agent_id=agent_id,
                success_score=success_score,
                response_time=response_time,
                user_satisfaction=user_satisfaction,
                timestamp=datetime.now(),
                context=context or {},
                metadata={}
            )
            
            await self.memory_store.store_memory(memory)
            
            if success_score > 0.7:
                self.successful_routes += 1
                
        except Exception as e:
            logger.error(f"Error recording routing outcome: {e}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get contextual routing statistics"""
        memory_stats = self.memory_store.get_memory_stats()
        
        return {
            "contextual_memory_routing": {
                "total_routes": self.total_routes,
                "memory_hits": self.memory_hits,
                "successful_routes": self.successful_routes,
                "hit_rate": self.memory_hits / max(self.total_routes, 1),
                "success_rate": self.successful_routes / max(self.total_routes, 1),
                "memory_store": memory_stats,
                "vector_available": VECTOR_AVAILABLE,
                "config": {
                    "similarity_threshold": self.similarity_threshold,
                    "memory_weight": self.memory_weight,
                    "recency_weight": self.recency_weight,
                    "success_weight": self.success_weight
                }
            }
        }

# Global instance
contextual_memory_router = None

def get_contextual_memory_router():
    """Get global contextual memory router instance"""
    global contextual_memory_router
    if contextual_memory_router is None:
        contextual_memory_router = ContextualMemoryRouter()
    return contextual_memory_router