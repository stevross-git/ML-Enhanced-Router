#!/usr/bin/env python3
"""
Active Learning Feedback Loop System
Implements continuous model improvement through user feedback
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
import threading
from queue import Queue

# Check for Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using local queue only")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for ML dependencies
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import TrainingArguments, Trainer
    from torch.utils.data import Dataset
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML libraries not available, using fallback implementations")

class FeedbackType(Enum):
    """Types of feedback for active learning"""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    MISSING_CONTEXT = "missing_context"
    WRONG_AGENT = "wrong_agent"
    POOR_QUALITY = "poor_quality"
    EXCELLENT = "excellent"

class FeedbackPriority(Enum):
    """Priority levels for feedback processing"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class UserFeedback:
    """User feedback data structure"""
    query: str
    agent_id: str
    feedback_type: FeedbackType
    score: float  # 0.0 to 1.0
    user_id: Optional[str] = None
    expected_category: Optional[str] = None
    expected_agent: Optional[str] = None
    comments: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: FeedbackPriority = FeedbackPriority.MEDIUM
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingExample:
    """Training example for model retraining"""
    query: str
    true_category: str
    predicted_category: str
    feedback_score: float
    agent_id: str
    timestamp: datetime
    features: Dict[str, Any] = field(default_factory=dict)

class FeedbackProcessor:
    """Process and validate user feedback"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._init_database()
        self.feedback_queue = Queue()
        self.processing_thread = None
        self._start_processing()
    
    def _init_database(self):
        """Initialize feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                score REAL NOT NULL,
                user_id TEXT,
                expected_category TEXT,
                expected_agent TEXT,
                comments TEXT,
                timestamp DATETIME NOT NULL,
                priority TEXT NOT NULL,
                confidence REAL NOT NULL,
                metadata TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                true_category TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                feedback_score REAL NOT NULL,
                agent_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                features TEXT,
                used_in_training BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _start_processing(self):
        """Start background processing thread"""
        self.processing_thread = threading.Thread(target=self._process_feedback_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_feedback_loop(self):
        """Background processing loop for feedback"""
        while True:
            try:
                if not self.feedback_queue.empty():
                    feedback = self.feedback_queue.get()
                    self._process_single_feedback(feedback)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error processing feedback: {e}")
    
    def _process_single_feedback(self, feedback: UserFeedback):
        """Process a single feedback item"""
        try:
            # Store feedback
            self._store_feedback(feedback)
            
            # Convert to training example if applicable
            if feedback.expected_category:
                training_example = TrainingExample(
                    query=feedback.query,
                    true_category=feedback.expected_category,
                    predicted_category=feedback.metadata.get('predicted_category', 'unknown'),
                    feedback_score=feedback.score,
                    agent_id=feedback.agent_id,
                    timestamp=feedback.timestamp,
                    features=feedback.metadata
                )
                self._store_training_example(training_example)
            
            # Flag for potential retraining if critical
            if feedback.priority == FeedbackPriority.CRITICAL:
                self._flag_for_retraining(feedback)
                
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
    
    def _store_feedback(self, feedback: UserFeedback):
        """Store feedback in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (
                query, agent_id, feedback_type, score, user_id,
                expected_category, expected_agent, comments,
                timestamp, priority, confidence, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.query,
            feedback.agent_id,
            feedback.feedback_type.value,
            feedback.score,
            feedback.user_id,
            feedback.expected_category,
            feedback.expected_agent,
            feedback.comments,
            feedback.timestamp,
            feedback.priority.value,
            feedback.confidence,
            json.dumps(feedback.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _store_training_example(self, example: TrainingExample):
        """Store training example in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_examples (
                query, true_category, predicted_category,
                feedback_score, agent_id, timestamp, features
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            example.query,
            example.true_category,
            example.predicted_category,
            example.feedback_score,
            example.agent_id,
            example.timestamp,
            json.dumps(example.features)
        ))
        
        conn.commit()
        conn.close()
    
    def _flag_for_retraining(self, feedback: UserFeedback):
        """Flag feedback for potential model retraining"""
        logger.info(f"Critical feedback received, flagging for retraining: {feedback.query}")
        # Could trigger immediate retraining or add to priority queue
    
    def submit_feedback(self, feedback: UserFeedback):
        """Submit feedback for processing"""
        self.feedback_queue.put(feedback)
        logger.info(f"Feedback submitted for query: {feedback.query[:50]}...")
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM feedback')
        total_feedback = cursor.fetchone()[0]
        
        cursor.execute('SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type')
        feedback_by_type = dict(cursor.fetchall())
        
        cursor.execute('SELECT AVG(score) FROM feedback')
        avg_score = cursor.fetchone()[0] or 0.0
        
        cursor.execute('SELECT COUNT(*) FROM training_examples')
        total_training_examples = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_feedback": total_feedback,
            "feedback_by_type": feedback_by_type,
            "average_score": avg_score,
            "total_training_examples": total_training_examples
        }

class ActiveLearningSystem:
    """Main active learning system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feedback_processor = FeedbackProcessor()
        self.retraining_threshold = self.config.get('retraining_threshold', 100)
        self.model_update_interval = self.config.get('model_update_interval', 3600)  # 1 hour
        self.last_update = datetime.now()
        
        # Redis for distributed feedback
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
            except:
                logger.warning("Redis not available, using local queue only")
                self.redis_client = None
    
    async def collect_feedback(self, query: str, agent_id: str, 
                              feedback_type: FeedbackType, score: float,
                              user_id: Optional[str] = None,
                              expected_category: Optional[str] = None,
                              comments: Optional[str] = None,
                              metadata: Dict[str, Any] = None) -> bool:
        """Collect user feedback"""
        try:
            feedback = UserFeedback(
                query=query,
                agent_id=agent_id,
                feedback_type=feedback_type,
                score=score,
                user_id=user_id,
                expected_category=expected_category,
                comments=comments,
                metadata=metadata or {}
            )
            
            # Submit for processing
            self.feedback_processor.submit_feedback(feedback)
            
            # Also send to Redis if available
            if self.redis_client:
                try:
                    feedback_data = {
                        "query": query,
                        "agent_id": agent_id,
                        "feedback_type": feedback_type.value,
                        "score": score,
                        "timestamp": datetime.now().isoformat(),
                        "user_id": user_id,
                        "expected_category": expected_category,
                        "comments": comments,
                        "metadata": metadata or {}
                    }
                    await asyncio.to_thread(
                        self.redis_client.rpush, 
                        "router_feedback_queue", 
                        json.dumps(feedback_data)
                    )
                except Exception as e:
                    logger.error(f"Failed to send feedback to Redis: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return False
    
    async def get_training_candidates(self, limit: int = 100) -> List[TrainingExample]:
        """Get training examples for model retraining"""
        conn = sqlite3.connect(self.feedback_processor.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT query, true_category, predicted_category,
                   feedback_score, agent_id, timestamp, features
            FROM training_examples
            WHERE used_in_training = FALSE
            ORDER BY feedback_score ASC, timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        examples = []
        for row in cursor.fetchall():
            examples.append(TrainingExample(
                query=row[0],
                true_category=row[1],
                predicted_category=row[2],
                feedback_score=row[3],
                agent_id=row[4],
                timestamp=datetime.fromisoformat(row[5]),
                features=json.loads(row[6]) if row[6] else {}
            ))
        
        conn.close()
        return examples
    
    async def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        stats = self.feedback_processor.get_feedback_stats()
        
        # Check if we have enough feedback
        if stats['total_training_examples'] < self.retraining_threshold:
            return False
        
        # Check if enough time has passed
        if datetime.now() - self.last_update < timedelta(seconds=self.model_update_interval):
            return False
        
        # Check if average score is below threshold
        if stats['average_score'] < 0.7:
            return True
        
        return False
    
    async def trigger_retraining(self):
        """Trigger model retraining with collected feedback"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, cannot retrain model")
            return False
        
        try:
            # Get training examples
            examples = await self.get_training_candidates()
            
            if len(examples) < 10:
                logger.info("Not enough training examples for retraining")
                return False
            
            logger.info(f"Starting model retraining with {len(examples)} examples")
            
            # This would integrate with your existing ML classifier
            # For now, just log the action
            logger.info("Model retraining completed (placeholder)")
            self.last_update = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get active learning system statistics"""
        stats = self.feedback_processor.get_feedback_stats()
        
        return {
            "active_learning": {
                "total_feedback": stats['total_feedback'],
                "feedback_by_type": stats['feedback_by_type'],
                "average_score": stats['average_score'],
                "training_examples": stats['total_training_examples'],
                "retraining_threshold": self.retraining_threshold,
                "last_update": self.last_update.isoformat(),
                "redis_available": self.redis_client is not None,
                "ml_available": ML_AVAILABLE
            }
        }

# Global instance
active_learning_system = None

def get_active_learning_system():
    """Get global active learning system instance"""
    global active_learning_system
    if active_learning_system is None:
        active_learning_system = ActiveLearningSystem()
    return active_learning_system