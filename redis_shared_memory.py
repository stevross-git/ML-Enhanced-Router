"""
Redis Shared Memory Layer for AI-to-AI Communication
Implements hot tier caching with context management and intelligent data storage
"""

import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis
from redis.connection import ConnectionPool
import pickle
import gzip

logger = logging.getLogger(__name__)

@dataclass
class ContextMetadata:
    """Metadata for stored context"""
    context_id: str
    created_at: int
    last_accessed: int
    access_count: int
    size_bytes: int
    expiry_time: int
    associated_agents: List[str]
    content_hash: str
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int
    cache_hits: int
    cache_misses: int
    evictions: int
    total_size_bytes: int
    avg_response_time_ms: float
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

class RedisSharedMemory:
    """Redis-based shared memory system for AI communication"""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        socket_timeout: int = 30,
        socket_connect_timeout: int = 30,
        max_memory_mb: int = 1024,
        enable_compression: bool = True
    ):
        """Initialize Redis shared memory system"""
        
        self.host = host
        self.port = port
        self.db = db
        self.enable_compression = enable_compression
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Create connection pool
        self.pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=False  # We handle binary data
        )
        
        # Initialize Redis client
        self.redis_client = redis.Redis(connection_pool=self.pool)
        
        # Key prefixes for different data types
        self.CONTEXT_PREFIX = "ctx:"
        self.METADATA_PREFIX = "meta:"
        self.STATS_PREFIX = "stats:"
        self.AGENT_PREFIX = "agent:"
        self.SESSION_PREFIX = "session:"
        
        # Initialize statistics
        self.stats = CacheStats(0, 0, 0, 0, 0, 0.0)
        
        # Test connection
        self._test_connection()
        
        logger.info(f"Redis shared memory initialized: {host}:{port}/{db}")
    
    def _test_connection(self):
        """Test Redis connection"""
        try:
            self.redis_client.ping()
            logger.info("Redis connection successful")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def store_context(
        self,
        context_id: str,
        content: Any,
        associated_agents: List[str],
        expiry_seconds: int = 3600,
        tags: List[str] = None
    ) -> bool:
        """Store context data in Redis"""
        try:
            start_time = time.time()
            
            # Serialize content
            serialized_content = self._serialize_data(content)
            
            # Calculate content hash
            content_hash = hashlib.sha256(serialized_content).hexdigest()
            
            # Create metadata
            current_time = int(time.time())
            metadata = ContextMetadata(
                context_id=context_id,
                created_at=current_time,
                last_accessed=current_time,
                access_count=0,
                size_bytes=len(serialized_content),
                expiry_time=current_time + expiry_seconds,
                associated_agents=associated_agents,
                content_hash=content_hash,
                tags=tags or []
            )
            
            # Store content and metadata
            context_key = f"{self.CONTEXT_PREFIX}{context_id}"
            metadata_key = f"{self.METADATA_PREFIX}{context_id}"
            
            pipe = self.redis_client.pipeline()
            pipe.setex(context_key, expiry_seconds, serialized_content)
            pipe.setex(metadata_key, expiry_seconds, self._serialize_data(asdict(metadata)))
            
            # Index by agents
            for agent_id in associated_agents:
                agent_key = f"{self.AGENT_PREFIX}{agent_id}"
                pipe.sadd(agent_key, context_id)
                pipe.expire(agent_key, expiry_seconds)
            
            # Index by tags
            for tag in (tags or []):
                tag_key = f"tag:{tag}"
                pipe.sadd(tag_key, context_id)
                pipe.expire(tag_key, expiry_seconds)
            
            pipe.execute()
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(cache_hit=False, processing_time=processing_time)
            
            logger.info(f"Stored context {context_id} ({len(serialized_content)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error storing context {context_id}: {e}")
            return False
    
    def retrieve_context(self, context_id: str, agent_id: Optional[str] = None) -> Optional[Any]:
        """Retrieve context data from Redis"""
        try:
            start_time = time.time()
            
            context_key = f"{self.CONTEXT_PREFIX}{context_id}"
            metadata_key = f"{self.METADATA_PREFIX}{context_id}"
            
            # Get content and metadata
            pipe = self.redis_client.pipeline()
            pipe.get(context_key)
            pipe.get(metadata_key)
            results = pipe.execute()
            
            serialized_content, serialized_metadata = results
            
            if not serialized_content:
                self._update_stats(cache_hit=False, processing_time=(time.time() - start_time) * 1000)
                return None
            
            # Deserialize content
            content = self._deserialize_data(serialized_content)
            
            # Update access metadata
            if serialized_metadata:
                metadata = self._deserialize_data(serialized_metadata)
                metadata['last_accessed'] = int(time.time())
                metadata['access_count'] += 1
                
                # Check agent access permissions
                if agent_id and agent_id not in metadata['associated_agents']:
                    logger.warning(f"Agent {agent_id} attempted unauthorized access to context {context_id}")
                    return None
                
                # Update metadata in Redis
                self.redis_client.setex(
                    metadata_key,
                    metadata['expiry_time'] - int(time.time()),
                    self._serialize_data(metadata)
                )
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(cache_hit=True, processing_time=processing_time)
            
            logger.info(f"Retrieved context {context_id}")
            return content
            
        except Exception as e:
            logger.error(f"Error retrieving context {context_id}: {e}")
            return None
    
    def delete_context(self, context_id: str) -> bool:
        """Delete context from Redis"""
        try:
            context_key = f"{self.CONTEXT_PREFIX}{context_id}"
            metadata_key = f"{self.METADATA_PREFIX}{context_id}"
            
            # Get metadata to clean up indexes
            serialized_metadata = self.redis_client.get(metadata_key)
            if serialized_metadata:
                metadata = self._deserialize_data(serialized_metadata)
                
                # Remove from agent indexes
                for agent_id in metadata.get('associated_agents', []):
                    agent_key = f"{self.AGENT_PREFIX}{agent_id}"
                    self.redis_client.srem(agent_key, context_id)
                
                # Remove from tag indexes
                for tag in metadata.get('tags', []):
                    tag_key = f"tag:{tag}"
                    self.redis_client.srem(tag_key, context_id)
            
            # Delete content and metadata
            pipe = self.redis_client.pipeline()
            pipe.delete(context_key)
            pipe.delete(metadata_key)
            deleted_count = sum(pipe.execute())
            
            logger.info(f"Deleted context {context_id}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting context {context_id}: {e}")
            return False
    
    def search_contexts(
        self,
        agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[str]:
        """Search for context IDs based on agent or tags"""
        try:
            context_ids = set()
            
            if agent_id:
                agent_key = f"{self.AGENT_PREFIX}{agent_id}"
                agent_contexts = self.redis_client.smembers(agent_key)
                context_ids.update(ctx.decode('utf-8') for ctx in agent_contexts)
            
            if tags:
                for tag in tags:
                    tag_key = f"tag:{tag}"
                    tag_contexts = self.redis_client.smembers(tag_key)
                    tag_context_ids = {ctx.decode('utf-8') for ctx in tag_contexts}
                    
                    if context_ids:
                        context_ids &= tag_context_ids  # Intersection
                    else:
                        context_ids = tag_context_ids
            
            # Limit results
            return list(context_ids)[:limit]
            
        except Exception as e:
            logger.error(f"Error searching contexts: {e}")
            return []
    
    def get_context_metadata(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific context"""
        try:
            metadata_key = f"{self.METADATA_PREFIX}{context_id}"
            serialized_metadata = self.redis_client.get(metadata_key)
            
            if not serialized_metadata:
                return None
            
            return self._deserialize_data(serialized_metadata)
            
        except Exception as e:
            logger.error(f"Error getting metadata for context {context_id}: {e}")
            return None
    
    def cleanup_expired(self) -> int:
        """Clean up expired contexts"""
        try:
            current_time = int(time.time())
            cleaned_count = 0
            
            # Scan for all metadata keys
            for key in self.redis_client.scan_iter(match=f"{self.METADATA_PREFIX}*"):
                try:
                    serialized_metadata = self.redis_client.get(key)
                    if serialized_metadata:
                        metadata = self._deserialize_data(serialized_metadata)
                        if metadata.get('expiry_time', 0) < current_time:
                            context_id = key.decode('utf-8').replace(self.METADATA_PREFIX, '')
                            if self.delete_context(context_id):
                                cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Error cleaning up context {key}: {e}")
                    continue
            
            logger.info(f"Cleaned up {cleaned_count} expired contexts")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            info = self.redis_client.info('memory')
            
            return {
                "used_memory": info.get('used_memory', 0),
                "used_memory_human": info.get('used_memory_human', '0B'),
                "used_memory_peak": info.get('used_memory_peak', 0),
                "used_memory_peak_human": info.get('used_memory_peak_human', '0B'),
                "total_system_memory": info.get('total_system_memory', 0),
                "maxmemory": info.get('maxmemory', 0),
                "maxmemory_human": info.get('maxmemory_human', '0B'),
                "memory_usage_percentage": (info.get('used_memory', 0) / max(info.get('total_system_memory', 1), 1)) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        memory_info = self.get_memory_usage()
        
        return {
            "hit_rate": self.stats.hit_rate,
            "total_requests": self.stats.total_requests,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "avg_response_time_ms": self.stats.avg_response_time_ms,
            "memory_usage": memory_info,
            "connected_clients": self.redis_client.info().get('connected_clients', 0),
            "total_commands_processed": self.redis_client.info().get('total_commands_processed', 0)
        }
    
    def flush_all(self) -> bool:
        """Clear all cached data (use with caution)"""
        try:
            self.redis_client.flushdb()
            self.stats = CacheStats(0, 0, 0, 0, 0, 0.0)
            logger.warning("All cached data has been flushed")
            return True
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data with optional compression"""
        try:
            # Use pickle for Python objects
            serialized = pickle.dumps(data)
            
            # Apply compression if enabled and data is large enough
            if self.enable_compression and len(serialized) > 1024:
                compressed = gzip.compress(serialized)
                # Only use compression if it actually reduces size
                if len(compressed) < len(serialized):
                    return b'GZIP:' + compressed
            
            return b'RAW:' + serialized
            
        except Exception as e:
            logger.error(f"Error serializing data: {e}")
            raise
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data with automatic decompression"""
        try:
            if data.startswith(b'GZIP:'):
                # Decompress gzipped data
                compressed_data = data[5:]  # Remove 'GZIP:' prefix
                decompressed = gzip.decompress(compressed_data)
                return pickle.loads(decompressed)
            elif data.startswith(b'RAW:'):
                # Raw pickled data
                serialized_data = data[4:]  # Remove 'RAW:' prefix
                return pickle.loads(serialized_data)
            else:
                # Legacy format - assume raw pickle
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            raise
    
    def _update_stats(self, cache_hit: bool, processing_time: float):
        """Update cache statistics"""
        self.stats.total_requests += 1
        
        if cache_hit:
            self.stats.cache_hits += 1
        else:
            self.stats.cache_misses += 1
        
        # Update rolling average response time
        if self.stats.total_requests == 1:
            self.stats.avg_response_time_ms = processing_time
        else:
            alpha = 0.1  # Smoothing factor for rolling average
            self.stats.avg_response_time_ms = (
                alpha * processing_time + 
                (1 - alpha) * self.stats.avg_response_time_ms
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection"""
        try:
            start_time = time.time()
            
            # Test basic operations
            test_key = "health_check_test"
            test_value = f"test_{int(time.time())}"
            
            # Write test
            self.redis_client.setex(test_key, 10, test_value)
            
            # Read test
            retrieved_value = self.redis_client.get(test_key)
            
            # Cleanup
            self.redis_client.delete(test_key)
            
            response_time = (time.time() - start_time) * 1000
            
            # Check if read/write worked
            is_healthy = (
                retrieved_value is not None and 
                retrieved_value.decode('utf-8') == test_value
            )
            
            return {
                "healthy": is_healthy,
                "response_time_ms": response_time,
                "redis_info": self.redis_client.info(),
                "memory_usage": self.get_memory_usage(),
                "stats": self.get_cache_stats()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "response_time_ms": -1
            }

# Factory function for easy initialization
def create_redis_shared_memory(
    redis_url: Optional[str] = None,
    **kwargs
) -> RedisSharedMemory:
    """Factory function to create Redis shared memory instance"""
    
    if redis_url:
        # Parse Redis URL
        import urllib.parse
        parsed = urllib.parse.urlparse(redis_url)
        
        config = {
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 6379,
            'db': int(parsed.path.lstrip('/')) if parsed.path else 0,
            'password': parsed.password
        }
        config.update(kwargs)
        return RedisSharedMemory(**config)
    else:
        return RedisSharedMemory(**kwargs)

# Context manager for automatic cleanup
class RedisContextManager:
    """Context manager for Redis shared memory with automatic cleanup"""
    
    def __init__(self, redis_memory: RedisSharedMemory):
        self.redis_memory = redis_memory
        self.created_contexts = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up any contexts created during this session
        for context_id in self.created_contexts:
            self.redis_memory.delete_context(context_id)
    
    def store_context(self, context_id: str, *args, **kwargs) -> bool:
        """Store context and track for cleanup"""
        success = self.redis_memory.store_context(context_id, *args, **kwargs)
        if success:
            self.created_contexts.append(context_id)
        return success