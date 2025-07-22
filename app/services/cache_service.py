"""
Cache Service
Handles caching operations, cache management, and performance optimization
"""

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Union
from functools import wraps

from flask import current_app
import redis
from sqlalchemy import text

from app.extensions import db
from app.models.cache import AICacheEntry, AICacheStats
from app.utils.exceptions import CacheError, ServiceError


class CacheService:
    """Service for handling caching operations"""
    
    def __init__(self):
        self.redis_client = None
        self.default_ttl = 3600  # 1 hour
        self.enabled = True
        self.cache_prefix = "ml_router:"
    
    def initialize(self, app):
        """Initialize the cache service with app configuration"""
        try:
            redis_url = app.config.get('REDIS_URL', 'redis://localhost:6379/0')
            redis_host = app.config.get('REDIS_HOST', 'localhost')
            redis_port = app.config.get('REDIS_PORT', 6379)
            redis_db = app.config.get('REDIS_DB', 0)
            redis_password = app.config.get('REDIS_PASSWORD')
            
            try:
                if redis_url:
                    self.redis_client = redis.from_url(redis_url)
                else:
                    self.redis_client = redis.Redis(
                        host=redis_host,
                        port=redis_port,
                        db=redis_db,
                        password=redis_password,
                        decode_responses=True
                    )
                
                # Test connection
                self.redis_client.ping()
                current_app.logger.info("Redis cache connected successfully")
                
            except Exception as e:
                current_app.logger.warning(f"Redis connection failed: {e}. Using database cache only.")
                self.redis_client = None
            
            self.default_ttl = app.config.get('CACHE_DEFAULT_TTL', 3600)
            self.enabled = app.config.get('CACHE_ENABLED', True)
            self.cache_prefix = app.config.get('CACHE_PREFIX', 'ml_router:')
            
        except Exception as e:
            current_app.logger.error(f"Cache service initialization failed: {e}")
            self.enabled = False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        if not self.enabled:
            return default
        
        try:
            full_key = self._make_key(key)
            
            if self.redis_client:
                try:
                    value = self.redis_client.get(full_key)
                    if value is not None:
                        self._record_hit(key, 'redis')
                        return self._deserialize(value)
                except Exception as e:
                    current_app.logger.warning(f"Redis get error: {e}")
            
            cache_entry = AICacheEntry.query.filter_by(
                cache_key=full_key,
                is_valid=True
            ).first()
            
            if cache_entry:
                if cache_entry.expires_at and cache_entry.expires_at < datetime.utcnow():
                    self._expire_entry(cache_entry)
                    self._record_miss(key, 'database')
                    return default
                
                self._record_hit(key, 'database')
                return self._deserialize(cache_entry.response_data.get("value", ""))
            
            self._record_miss(key, 'database')
            return default
            
        except Exception as e:
            current_app.logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            full_key = self._make_key(key)
            ttl = ttl or self.default_ttl
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            serialized_value = self._serialize(value)
            
            if self.redis_client:
                try:
                    self.redis_client.setex(full_key, ttl, serialized_value)
                except Exception as e:
                    current_app.logger.warning(f"Redis set error: {e}")
            
            try:
                existing = AICacheEntry.query.filter_by(cache_key=full_key).first()
                if existing:
                    db.session.delete(existing)
                
                cache_entry = AICacheEntry(
                    cache_key=full_key,
                    query_hash=hashlib.sha256("cached_value".encode()).hexdigest(),
                    model_id="cache_service",
                    provider="internal",
                    query_text="cached_value",
                    response_data={"value": serialized_value},
                    ttl_seconds=ttl,
                    expires_at=expires_at
                )
                
                db.session.add(cache_entry)
                db.session.commit()
                
            except Exception as e:
                db.session.rollback()
                current_app.logger.warning(f"Database cache set error: {e}")
            
            return True
            
        except Exception as e:
            current_app.logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            full_key = self._make_key(key)
            
            if self.redis_client:
                try:
                    self.redis_client.delete(full_key)
                except Exception as e:
                    current_app.logger.warning(f"Redis delete error: {e}")
            
            try:
                AICacheEntry.query.filter_by(cache_key=full_key).delete()
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                current_app.logger.warning(f"Database cache delete error: {e}")
            
            return True
            
        except Exception as e:
            current_app.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries
        
        Args:
            pattern: Key pattern to match (optional, clears all if None)
            
        Returns:
            Number of entries cleared
        """
        if not self.enabled:
            return 0
        
        try:
            cleared_count = 0
            
            if self.redis_client:
                try:
                    if pattern:
                        keys = self.redis_client.keys(f"{self.cache_prefix}{pattern}")
                    else:
                        keys = self.redis_client.keys(f"{self.cache_prefix}*")
                    
                    if keys:
                        cleared_count += self.redis_client.delete(*keys)
                        
                except Exception as e:
                    current_app.logger.warning(f"Redis clear error: {e}")
            
            try:
                if pattern:
                    query = AICacheEntry.query.filter(
                        AICacheEntry.cache_key.like(f"{self.cache_prefix}{pattern}")
                    )
                else:
                    query = AICacheEntry.query.filter(
                        AICacheEntry.cache_key.like(f"{self.cache_prefix}%")
                    )
                
                db_cleared = query.delete(synchronize_session=False)
                db.session.commit()
                cleared_count = max(cleared_count, db_cleared)
                
            except Exception as e:
                db.session.rollback()
                current_app.logger.warning(f"Database cache clear error: {e}")
            
            current_app.logger.info(f"Cleared {cleared_count} cache entries")
            return cleared_count
            
        except Exception as e:
            current_app.logger.error(f"Cache clear error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict containing cache statistics
        """
        try:
            stats = {
                'enabled': self.enabled,
                'redis_connected': self.redis_client is not None,
                'total_entries': 0,
                'expired_entries': 0,
                'hit_rate': 0.0,
                'memory_usage': 0
            }
            
            try:
                total_entries = AICacheEntry.query.filter_by(is_valid=True).count()
                expired_entries = AICacheEntry.query.filter(
                    AICacheEntry.expires_at < datetime.utcnow(),
                    AICacheEntry.is_valid == True
                ).count()
                
                stats['total_entries'] = total_entries
                stats['expired_entries'] = expired_entries
                
            except Exception as e:
                current_app.logger.warning(f"Database stats error: {e}")
            
            if self.redis_client:
                try:
                    redis_info = self.redis_client.info()
                    stats['memory_usage'] = redis_info.get('used_memory', 0)
                    stats['redis_keys'] = len(self.redis_client.keys(f"{self.cache_prefix}*"))
                except Exception as e:
                    current_app.logger.warning(f"Redis stats error: {e}")
            
            try:
                cache_stats = AICacheStats.query.order_by(AICacheStats.created_at.desc()).limit(100).all()
                if cache_stats:
                    total_requests = sum(s.total_requests for s in cache_stats)
                    total_hits = sum(s.cache_hits for s in cache_stats)
                    stats['hit_rate'] = (total_hits / total_requests * 100) if total_requests > 0 else 0
            except Exception as e:
                current_app.logger.warning(f"Hit rate calculation error: {e}")
            
            return stats
            
        except Exception as e:
            current_app.logger.error(f"Cache stats error: {e}")
            return {'enabled': False, 'error': str(e)}
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries
        
        Returns:
            Number of entries cleaned up
        """
        try:
            expired_count = AICacheEntry.query.filter(
                AICacheEntry.expires_at < datetime.utcnow(),
                AICacheEntry.is_valid == True
            ).update({'is_valid': False})
            
            db.session.commit()
            
            current_app.logger.info(f"Cleaned up {expired_count} expired cache entries")
            return expired_count
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Cache cleanup error: {e}")
            return 0
    
    def cached(self, key_func=None, ttl=None):
        """
        Decorator for caching function results
        
        Args:
            key_func: Function to generate cache key (optional)
            ttl: Time to live in seconds (optional)
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
                
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator
    
    def _make_key(self, key: str) -> str:
        """Generate full cache key with prefix"""
        return f"{self.cache_prefix}{key}"
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage"""
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return pickle.dumps(value).hex()
    
    def _deserialize(self, value: Union[str, bytes]) -> Any:
        """Deserialize value from storage"""
        try:
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                return pickle.loads(bytes.fromhex(str(value)))
            except Exception:
                return value
    
    def _record_hit(self, key: str, cache_type: str):
        """Record cache hit for statistics"""
        try:
            today = datetime.utcnow()
            period_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1)
            
            stats = AICacheStats.query.filter_by(
                period_start=period_start,
                granularity="day"
            ).first()
            
            if not stats:
                stats = AICacheStats(
                    period_start=period_start,
                    period_end=period_end,
                    granularity="day",
                    total_requests=0,
                    cache_hits=0,
                    cache_misses=0
                )
                db.session.add(stats)
            
            stats.cache_hits += 1
            stats.total_requests += 1
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.warning(f"Cache hit recording error: {e}")
    
    def _record_miss(self, key: str, cache_type: str):
        """Record cache miss for statistics"""
        try:
            today = datetime.utcnow()
            period_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1)
            
            stats = AICacheStats.query.filter_by(
                period_start=period_start,
                granularity="day"
            ).first()
            
            if not stats:
                stats = AICacheStats(
                    period_start=period_start,
                    period_end=period_end,
                    granularity="day",
                    total_requests=0,
                    cache_hits=0,
                    cache_misses=0
                )
                db.session.add(stats)
            
            stats.cache_misses += 1
            stats.total_requests += 1
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.warning(f"Cache miss recording error: {e}")
    
    def _expire_entry(self, entry: AICacheEntry):
        """Mark cache entry as expired"""
        try:
            entry.is_valid = False
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            current_app.logger.warning(f"Cache entry expiration error: {e}")
            
def get_stats(self) -> Dict[str, Any]:
    """Get comprehensive cache statistics"""
    try:
        stats = {
            'enabled': self.enabled,
            'type': 'database' if self.enabled else 'disabled',
            'ttl_seconds': self.ttl,
            'max_size': self.max_size,
            'total_entries': 0,
            'valid_entries': 0,
            'expired_entries': 0,
            'hit_rate': 0.0,
            'memory_usage': 0,
            'redis_keys': 0,
            'cache_prefix': getattr(self, 'cache_prefix', 'ml_router_cache:'),
            'last_updated': datetime.utcnow().isoformat()
        }
        
        if not self.enabled:
            return stats
        
        try:
            # Import models properly
            from app.models.cache import AICacheEntry, AICacheStats
            from app.extensions import db
            
            # Use db.session.query instead of Model.query
            total_entries = db.session.query(AICacheEntry).filter_by(is_valid=True).count()
            
            # Count expired entries
            expired_entries = db.session.query(AICacheEntry).filter(
                AICacheEntry.expires_at < datetime.utcnow(),
                AICacheEntry.is_valid == True
            ).count()
            
            stats['total_entries'] = total_entries
            stats['expired_entries'] = expired_entries
            stats['valid_entries'] = total_entries - expired_entries
            
        except Exception as e:
            current_app.logger.warning(f"Database stats error: {e}")
        
        # Redis statistics
        if hasattr(self, 'redis_client') and self.redis_client:
            try:
                redis_info = self.redis_client.info()
                stats['memory_usage'] = redis_info.get('used_memory', 0)
                cache_keys = self.redis_client.keys(f"{stats['cache_prefix']}*")
                stats['redis_keys'] = len(cache_keys)
            except Exception as e:
                current_app.logger.warning(f"Redis stats error: {e}")
        
        # Hit rate calculation
        try:
            from app.models.cache import AICacheStats
            from sqlalchemy import func
            
            # Get recent cache statistics
            recent_stats = db.session.query(AICacheStats).order_by(
                AICacheStats.created_at.desc()
            ).limit(10).all()
            
            if recent_stats:
                total_requests = sum(s.total_requests for s in recent_stats)
                total_hits = sum(s.cache_hits for s in recent_stats)
                stats['hit_rate'] = round(
                    (total_hits / total_requests * 100) if total_requests > 0 else 0, 2
                )
                
        except Exception as e:
            current_app.logger.warning(f"Hit rate calculation error: {e}")
        
        return stats
        
    except Exception as e:
        current_app.logger.error(f"Cache stats error: {e}")
        return {
            'enabled': False, 
            'error': str(e),
            'type': 'error'
        }

# Also add this helper method to handle database operations safely

def _safe_db_query(self, query_func, default_value=0, error_context="database query"):
    """Safely execute database queries with error handling"""
    try:
        return query_func()
    except Exception as e:
        current_app.logger.warning(f"Safe DB query error in {error_context}: {e}")
        return default_value


# Singleton instance
_cache_service = None

def get_cache_manager() -> CacheService:
    """Get singleton cache service instance"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
        if current_app:
            _cache_service.initialize(current_app)
    return _cache_service

def init_cache_service(app):
    """Initialize cache service with Flask app"""
    service = get_cache_manager()
    service.initialize(app)
    return service
