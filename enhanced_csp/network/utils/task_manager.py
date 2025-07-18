"""Async task management utilities."""
import asyncio
import logging
from typing import Dict, Set, Optional, Callable, Any
import time

logger = logging.getLogger(__name__)

class TaskManager:
    """Manage async tasks with proper cleanup."""
    
    def __init__(self):
        self.tasks: Set[asyncio.Task] = set()
        self.named_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
    async def start(self):
        """Start the task manager."""
        self.running = True
        logger.info("TaskManager started")
    
    async def stop(self):
        """Stop all tasks and cleanup."""
        logger.info("Stopping TaskManager...")
        self.running = False
        
        # Cancel all tasks
        all_tasks = list(self.tasks) + list(self.named_tasks.values())
        
        for task in all_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        self.tasks.clear()
        self.named_tasks.clear()
        logger.info("TaskManager stopped")
    
    def create_task(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """Create and track a task."""
        task = asyncio.create_task(coro)
        
        if name:
            self.named_tasks[name] = task
            task.set_name(name)
        else:
            self.tasks.add(task)
        
        # Add completion callback
        task.add_done_callback(self._task_done_callback)
        
        logger.debug(f"Created task: {name or 'unnamed'}")
        return task
    
    def _task_done_callback(self, task: asyncio.Task):
        """Handle task completion."""
        # Remove from tracking
        self.tasks.discard(task)
        
        # Remove from named tasks
        for name, named_task in list(self.named_tasks.items()):
            if named_task is task:
                del self.named_tasks[name]
                break
        
        # Log if task failed
        if task.cancelled():
            logger.debug(f"Task cancelled: {task.get_name()}")
        elif task.exception():
            logger.error(f"Task failed: {task.get_name()}: {task.exception()}")
        else:
            logger.debug(f"Task completed: {task.get_name()}")
    
    def get_task(self, name: str) -> Optional[asyncio.Task]:
        """Get a named task."""
        return self.named_tasks.get(name)
    
    def cancel_task(self, name: str) -> bool:
        """Cancel a named task."""
        task = self.named_tasks.get(name)
        if task and not task.done():
            task.cancel()
            logger.info(f"Cancelled task: {name}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        return {
            "total_tasks": len(self.tasks) + len(self.named_tasks),
            "unnamed_tasks": len(self.tasks),
            "named_tasks": len(self.named_tasks),
            "named_task_names": list(self.named_tasks.keys()),
            "running": self.running,
        }
