"""
Async Utilities
Helper functions for async operations in Flask application
"""

import asyncio
import functools
import threading
import logging
from typing import Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Global thread pool for async operations
_thread_pool = None

def get_thread_pool():
    """Get or create the global thread pool"""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="async_helper")
    return _thread_pool

def run_async_in_thread(coro):
    """
    Run an async coroutine in a background thread
    
    Args:
        coro: Async coroutine to run
        
    Returns:
        Result of the coroutine
    """
    def run_in_thread():
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(coro)
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error running async in thread: {e}")
            raise
    
    # Run in thread pool
    thread_pool = get_thread_pool()
    future = thread_pool.submit(run_in_thread)
    return future.result()

def async_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to async functions
    
    Args:
        timeout_seconds: Timeout in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        return wrapper
    return decorator

def run_with_timeout(coro, timeout_seconds: float):
    """
    Run a coroutine with a timeout
    
    Args:
        coro: Coroutine to run
        timeout_seconds: Timeout in seconds
        
    Returns:
        Result of the coroutine
        
    Raises:
        TimeoutError: If the operation times out
    """
    async def run_with_timeout_async():
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    
    return run_async_in_thread(run_with_timeout_async())

def sync_to_async(func):
    """
    Convert a synchronous function to async
    
    Args:
        func: Synchronous function to convert
        
    Returns:
        Async wrapper function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        thread_pool = get_thread_pool()
        return await loop.run_in_executor(thread_pool, functools.partial(func, *args, **kwargs))
    
    return wrapper

def async_to_sync(coro_func):
    """
    Convert an async function to synchronous
    
    Args:
        coro_func: Async function to convert
        
    Returns:
        Synchronous wrapper function
    """
    @functools.wraps(coro_func)
    def wrapper(*args, **kwargs):
        coro = coro_func(*args, **kwargs)
        return run_async_in_thread(coro)
    
    return wrapper

class AsyncContextManager:
    """Helper for managing async context in sync code"""
    
    def __init__(self):
        self.loop = None
        self.thread = None
        
    def start(self):
        """Start the async context"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        # Wait for loop to be ready
        while self.loop is None:
            threading.Event().wait(0.01)
    
    def stop(self):
        """Stop the async context"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.thread:
                self.thread.join(timeout=1.0)
    
    def run_async(self, coro):
        """Run a coroutine in the managed loop"""
        if not self.loop:
            raise RuntimeError("Async context not started")
        
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

def ensure_async_context(func):
    """
    Decorator to ensure an async context exists
    
    Creates a temporary async context if none exists
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Try to get current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context
                return func(*args, **kwargs)
        except RuntimeError:
            pass
        
        # Create temporary async context
        context = AsyncContextManager()
        try:
            context.start()
            return func(*args, **kwargs)
        finally:
            context.stop()
    
    return wrapper

def safe_async_call(coro, default_value=None, log_errors=True):
    """
    Safely call an async function and return a default value on error
    
    Args:
        coro: Coroutine to call
        default_value: Value to return on error
        log_errors: Whether to log errors
        
    Returns:
        Result of the coroutine or default_value on error
    """
    try:
        return run_async_in_thread(coro)
    except Exception as e:
        if log_errors:
            logger.error(f"Safe async call failed: {e}")
        return default_value

# Cleanup function for graceful shutdown
def cleanup_async_resources():
    """Clean up async resources"""
    global _thread_pool
    
    if _thread_pool:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None
        logger.info("Cleaned up async thread pool")

# Export commonly used functions
__all__ = [
    'run_async_in_thread',
    'async_timeout',
    'run_with_timeout',
    'sync_to_async',
    'async_to_sync',
    'AsyncContextManager',
    'ensure_async_context',
    'safe_async_call',
    'cleanup_async_resources'
]