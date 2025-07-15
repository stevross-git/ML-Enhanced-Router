"""
Temporal Memory Weighting System
Manages memory decay, revival, and temporal relevance
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
import threading
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryPriority(Enum):
    """Memory priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PINNED = "pinned"
    CRITICAL = "critical"

class DecayType(Enum):
    """Types of memory decay"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    STEP = "step"

@dataclass
class MemoryDecayConfig:
    """Configuration for memory decay"""
    decay_type: DecayType = DecayType.EXPONENTIAL
    half_life_days: float = 30.0  # Days for confidence to halve
    minimum_confidence: float = 0.1  # Minimum confidence before archival
    inactivity_threshold_days: float = 21.0  # Days of inactivity before decay acceleration
    revival_boost: float = 0.2  # Confidence boost when memory is accessed
    max_revival_confidence: float = 0.9  # Maximum confidence after revival

@dataclass
class TemporalMemoryWeight:
    """Represents temporal weighting for a memory"""
    memory_id: str
    original_confidence: float
    current_confidence: float
    priority: MemoryPriority
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    decay_rate: float = 0.0
    is_pinned: bool = False
    inactivity_days: float = 0.0
    revival_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class TemporalMemoryWeightingSystem:
    """Manages temporal aspects of memory weighting"""
    
    def __init__(self, db_path: str = "temporal_memory.db", 
                 config: MemoryDecayConfig = None):
        self.db_path = db_path
        self.config = config or MemoryDecayConfig()
        self.memory_weights: Dict[str, TemporalMemoryWeight] = {}
        self.lock = threading.Lock()
        
        self._init_database()
        self._load_from_database()
        
        # Start background decay processing
        self._start_decay_processor()
    
    def _init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS temporal_memory_weights (
                memory_id TEXT PRIMARY KEY,
                original_confidence REAL NOT NULL,
                current_confidence REAL NOT NULL,
                priority TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                last_accessed TIMESTAMP NOT NULL,
                access_count INTEGER DEFAULT 0,
                decay_rate REAL DEFAULT 0.0,
                is_pinned BOOLEAN DEFAULT FALSE,
                inactivity_days REAL DEFAULT 0.0,
                revival_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_decay_logs (
                log_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                old_confidence REAL,
                new_confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_from_database(self):
        """Load existing memory weights from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM temporal_memory_weights')
        for row in cursor.fetchall():
            weight = TemporalMemoryWeight(
                memory_id=row[0],
                original_confidence=row[1],
                current_confidence=row[2],
                priority=MemoryPriority(row[3]),
                created_at=datetime.fromisoformat(row[4]),
                last_accessed=datetime.fromisoformat(row[5]),
                access_count=row[6] or 0,
                decay_rate=row[7] or 0.0,
                is_pinned=bool(row[8]),
                inactivity_days=row[9] or 0.0,
                revival_count=row[10] or 0,
                metadata=json.loads(row[11]) if row[11] else {}
            )
            self.memory_weights[row[0]] = weight
        
        conn.close()
    
    def _start_decay_processor(self):
        """Start background thread for memory decay processing"""
        def decay_processor():
            import time
            while True:
                try:
                    self._process_memory_decay()
                    time.sleep(3600)  # Process every hour
                except Exception as e:
                    logger.error(f"Error in decay processor: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        thread = threading.Thread(target=decay_processor, daemon=True)
        thread.start()
    
    def register_memory(self, memory_id: str, initial_confidence: float,
                       priority: MemoryPriority = MemoryPriority.MEDIUM,
                       created_at: datetime = None) -> bool:
        """Register a new memory for temporal weighting"""
        if created_at is None:
            created_at = datetime.now()
        
        with self.lock:
            if memory_id in self.memory_weights:
                return False  # Already registered
            
            weight = TemporalMemoryWeight(
                memory_id=memory_id,
                original_confidence=initial_confidence,
                current_confidence=initial_confidence,
                priority=priority,
                created_at=created_at,
                last_accessed=created_at,
                is_pinned=(priority == MemoryPriority.PINNED)
            )
            
            self.memory_weights[memory_id] = weight
            self._save_weight_to_db(weight)
            
            return True
    
    def access_memory(self, memory_id: str, 
                     revival_context: str = None) -> Tuple[float, bool]:
        """Access a memory and apply temporal weighting"""
        with self.lock:
            if memory_id not in self.memory_weights:
                return 0.0, False
            
            weight = self.memory_weights[memory_id]
            old_confidence = weight.current_confidence
            
            # Update access information
            weight.last_accessed = datetime.now()
            weight.access_count += 1
            
            # Calculate inactivity days
            weight.inactivity_days = 0.0  # Reset inactivity
            
            # Apply revival boost if confidence is low
            if weight.current_confidence < 0.5 and not weight.is_pinned:
                revival_boost = self.config.revival_boost
                weight.current_confidence = min(
                    weight.current_confidence + revival_boost,
                    self.config.max_revival_confidence
                )
                weight.revival_count += 1
                
                # Log revival event
                self._log_decay_event(
                    memory_id=memory_id,
                    event_type="revival",
                    old_confidence=old_confidence,
                    new_confidence=weight.current_confidence,
                    details=f"Revival boost applied: {revival_boost}"
                )
            
            self._save_weight_to_db(weight)
            
            return weight.current_confidence, True
    
    def pin_memory(self, memory_id: str, pinned: bool = True) -> bool:
        """Pin or unpin a memory to prevent/allow decay"""
        with self.lock:
            if memory_id not in self.memory_weights:
                return False
            
            weight = self.memory_weights[memory_id]
            weight.is_pinned = pinned
            
            if pinned:
                weight.priority = MemoryPriority.PINNED
                # Restore confidence to original level
                weight.current_confidence = weight.original_confidence
            else:
                weight.priority = MemoryPriority.MEDIUM
            
            self._save_weight_to_db(weight)
            
            # Log pin event
            self._log_decay_event(
                memory_id=memory_id,
                event_type="pin" if pinned else "unpin",
                old_confidence=weight.current_confidence,
                new_confidence=weight.current_confidence,
                details=f"Memory {'pinned' if pinned else 'unpinned'}"
            )
            
            return True
    
    def _process_memory_decay(self):
        """Process memory decay for all memories"""
        current_time = datetime.now()
        
        with self.lock:
            for memory_id, weight in self.memory_weights.items():
                if weight.is_pinned:
                    continue  # Skip pinned memories
                
                # Calculate time since last access
                time_since_access = current_time - weight.last_accessed
                weight.inactivity_days = time_since_access.days
                
                # Check if memory should be processed for decay
                if weight.inactivity_days >= self.config.inactivity_threshold_days:
                    old_confidence = weight.current_confidence
                    new_confidence = self._calculate_decay(weight, current_time)
                    
                    if new_confidence != old_confidence:
                        weight.current_confidence = new_confidence
                        self._save_weight_to_db(weight)
                        
                        # Log decay event
                        self._log_decay_event(
                            memory_id=memory_id,
                            event_type="decay",
                            old_confidence=old_confidence,
                            new_confidence=new_confidence,
                            details=f"Inactivity: {weight.inactivity_days} days"
                        )
    
    def _calculate_decay(self, weight: TemporalMemoryWeight, 
                        current_time: datetime) -> float:
        """Calculate memory decay based on configuration"""
        if weight.is_pinned:
            return weight.current_confidence
        
        # Time factors
        time_since_creation = current_time - weight.created_at
        time_since_access = current_time - weight.last_accessed
        
        days_since_creation = time_since_creation.days
        days_since_access = time_since_access.days
        
        # Base decay calculation
        if self.config.decay_type == DecayType.EXPONENTIAL:
            decay_factor = math.exp(-days_since_access / self.config.half_life_days)
        elif self.config.decay_type == DecayType.LINEAR:
            decay_factor = max(0, 1 - (days_since_access / (self.config.half_life_days * 2)))
        elif self.config.decay_type == DecayType.LOGARITHMIC:
            decay_factor = 1 / (1 + math.log(1 + days_since_access / self.config.half_life_days))
        elif self.config.decay_type == DecayType.STEP:
            if days_since_access < self.config.half_life_days:
                decay_factor = 1.0
            elif days_since_access < self.config.half_life_days * 2:
                decay_factor = 0.5
            else:
                decay_factor = 0.25
        else:
            decay_factor = 1.0
        
        # Apply priority modifiers
        if weight.priority == MemoryPriority.HIGH:
            decay_factor = decay_factor * 1.5  # Slower decay
        elif weight.priority == MemoryPriority.CRITICAL:
            decay_factor = decay_factor * 2.0  # Much slower decay
        elif weight.priority == MemoryPriority.LOW:
            decay_factor = decay_factor * 0.5  # Faster decay
        
        # Apply access count bonus (frequently accessed memories decay slower)
        access_bonus = min(0.3, weight.access_count * 0.02)
        decay_factor = min(1.0, decay_factor + access_bonus)
        
        # Calculate new confidence
        new_confidence = weight.original_confidence * decay_factor
        
        # Apply minimum confidence threshold
        return max(new_confidence, self.config.minimum_confidence)
    
    def get_memory_weight(self, memory_id: str) -> Optional[TemporalMemoryWeight]:
        """Get temporal weight for a memory"""
        return self.memory_weights.get(memory_id)
    
    def get_weighted_memories(self, memory_ids: List[str],
                            min_confidence: float = 0.1) -> List[Tuple[str, float]]:
        """Get weighted memories with confidence scores"""
        results = []
        
        for memory_id in memory_ids:
            weight = self.memory_weights.get(memory_id)
            if weight and weight.current_confidence >= min_confidence:
                results.append((memory_id, weight.current_confidence))
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def suggest_memory_cleanup(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """Suggest memories for cleanup based on low confidence"""
        suggestions = []
        
        for memory_id, weight in self.memory_weights.items():
            if (weight.current_confidence < 0.3 and 
                not weight.is_pinned and
                weight.inactivity_days > self.config.inactivity_threshold_days):
                
                suggestions.append({
                    "memory_id": memory_id,
                    "current_confidence": weight.current_confidence,
                    "original_confidence": weight.original_confidence,
                    "inactivity_days": weight.inactivity_days,
                    "last_accessed": weight.last_accessed.isoformat(),
                    "suggestion": "Consider archiving or removing this memory",
                    "reason": f"Low confidence ({weight.current_confidence:.2f}) and {weight.inactivity_days} days inactive"
                })
        
        # Sort by confidence (lowest first)
        suggestions.sort(key=lambda x: x["current_confidence"])
        return suggestions
    
    def get_memory_insights(self, memory_id: str) -> Dict[str, Any]:
        """Get insights about a memory's temporal behavior"""
        weight = self.memory_weights.get(memory_id)
        if not weight:
            return {"error": "Memory not found"}
        
        # Calculate decay rate
        age_days = (datetime.now() - weight.created_at).days
        if age_days > 0:
            decay_rate = (weight.original_confidence - weight.current_confidence) / age_days
        else:
            decay_rate = 0.0
        
        return {
            "memory_id": memory_id,
            "temporal_status": {
                "original_confidence": weight.original_confidence,
                "current_confidence": weight.current_confidence,
                "confidence_change": weight.current_confidence - weight.original_confidence,
                "decay_rate_per_day": decay_rate,
                "is_pinned": weight.is_pinned,
                "priority": weight.priority.value
            },
            "access_patterns": {
                "access_count": weight.access_count,
                "last_accessed": weight.last_accessed.isoformat(),
                "inactivity_days": weight.inactivity_days,
                "revival_count": weight.revival_count
            },
            "recommendations": self._get_memory_recommendations(weight),
            "predicted_decay": self._predict_future_decay(weight)
        }
    
    def _get_memory_recommendations(self, weight: TemporalMemoryWeight) -> List[str]:
        """Get recommendations for a memory"""
        recommendations = []
        
        if weight.current_confidence < 0.3 and not weight.is_pinned:
            recommendations.append("Consider pinning this memory if it's important")
        
        if weight.inactivity_days > self.config.inactivity_threshold_days:
            recommendations.append(f"Memory hasn't been accessed for {weight.inactivity_days} days")
        
        if weight.revival_count > 3:
            recommendations.append("This memory has been revived multiple times - consider pinning it")
        
        if weight.access_count > 10:
            recommendations.append("Frequently accessed memory - consider upgrading priority")
        
        return recommendations
    
    def _predict_future_decay(self, weight: TemporalMemoryWeight) -> Dict[str, float]:
        """Predict future confidence levels"""
        if weight.is_pinned:
            return {
                "in_7_days": weight.current_confidence,
                "in_30_days": weight.current_confidence,
                "in_90_days": weight.current_confidence
            }
        
        current_time = datetime.now()
        predictions = {}
        
        for days in [7, 30, 90]:
            future_time = current_time + timedelta(days=days)
            
            # Simulate weight at future time
            temp_weight = TemporalMemoryWeight(
                memory_id=weight.memory_id,
                original_confidence=weight.original_confidence,
                current_confidence=weight.current_confidence,
                priority=weight.priority,
                created_at=weight.created_at,
                last_accessed=weight.last_accessed,
                access_count=weight.access_count,
                is_pinned=weight.is_pinned
            )
            
            predicted_confidence = self._calculate_decay(temp_weight, future_time)
            predictions[f"in_{days}_days"] = predicted_confidence
        
        return predictions
    
    def _save_weight_to_db(self, weight: TemporalMemoryWeight):
        """Save memory weight to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO temporal_memory_weights 
            (memory_id, original_confidence, current_confidence, priority,
             created_at, last_accessed, access_count, decay_rate, is_pinned,
             inactivity_days, revival_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            weight.memory_id,
            weight.original_confidence,
            weight.current_confidence,
            weight.priority.value,
            weight.created_at.isoformat(),
            weight.last_accessed.isoformat(),
            weight.access_count,
            weight.decay_rate,
            weight.is_pinned,
            weight.inactivity_days,
            weight.revival_count,
            json.dumps(weight.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _log_decay_event(self, memory_id: str, event_type: str,
                        old_confidence: float, new_confidence: float,
                        details: str):
        """Log a decay event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        log_id = f"{event_type}_{memory_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cursor.execute('''
            INSERT INTO memory_decay_logs 
            (log_id, memory_id, event_type, old_confidence, new_confidence, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (log_id, memory_id, event_type, old_confidence, new_confidence, details))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_memories = len(self.memory_weights)
        if total_memories == 0:
            return {"total_memories": 0}
        
        # Calculate statistics
        pinned_count = sum(1 for w in self.memory_weights.values() if w.is_pinned)
        low_confidence_count = sum(1 for w in self.memory_weights.values() if w.current_confidence < 0.3)
        high_confidence_count = sum(1 for w in self.memory_weights.values() if w.current_confidence > 0.7)
        inactive_count = sum(1 for w in self.memory_weights.values() if w.inactivity_days > self.config.inactivity_threshold_days)
        
        avg_confidence = sum(w.current_confidence for w in self.memory_weights.values()) / total_memories
        avg_access_count = sum(w.access_count for w in self.memory_weights.values()) / total_memories
        
        return {
            "total_memories": total_memories,
            "pinned_memories": pinned_count,
            "low_confidence_memories": low_confidence_count,
            "high_confidence_memories": high_confidence_count,
            "inactive_memories": inactive_count,
            "average_confidence": avg_confidence,
            "average_access_count": avg_access_count,
            "decay_config": {
                "decay_type": self.config.decay_type.value,
                "half_life_days": self.config.half_life_days,
                "inactivity_threshold_days": self.config.inactivity_threshold_days
            }
        }

# Global instance
temporal_memory_system = None

def get_temporal_memory_system():
    """Get or create global temporal memory system instance"""
    global temporal_memory_system
    if temporal_memory_system is None:
        temporal_memory_system = TemporalMemoryWeightingSystem()
    return temporal_memory_system