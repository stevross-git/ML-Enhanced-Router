"""Enhanced CSP Network Main Entry Point."""
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core components
try:
    from .core.config import NetworkConfig
    from .core.node import NetworkNode
    from .utils.task_manager import TaskManager
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    IMPORTS_AVAILABLE = False

class EnhancedCSPMain:
    """Main Enhanced CSP Network application."""
    
    def __init__(self):
        self.config = None
        self.node = None
        self.task_manager = None
        self.running = False
        
    async def initialize(self):
        """Initialize the application."""
        logger.info("Initializing Enhanced CSP Network...")
        
        if not IMPORTS_AVAILABLE:
            logger.error("Required imports not available")
            return False
        
        try:
            # Create configuration
            self.config = NetworkConfig()
            self.config.node_name = "enhanced-csp-main"
            self.config.node_type = "ai_service"
            
            # Create task manager
            self.task_manager = TaskManager()
            await self.task_manager.start()
            
            # Create network node
            self.node = NetworkNode(self.config)
            
            logger.info("Enhanced CSP Network initialized")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def start(self):
        """Start the application."""
        if not await self.initialize():
            return False
        
        logger.info("Starting Enhanced CSP Network...")
        self.running = True
        
        try:
            # Start network node
            node_task = self.task_manager.create_task(
                self.node.start(), 
                name="network_node"
            )
            
            # Wait for shutdown signal
            await self._wait_for_shutdown()
            
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the application."""
        logger.info("Stopping Enhanced CSP Network...")
        self.running = False
        
        if self.node:
            await self.node.stop()
        
        if self.task_manager:
            await self.task_manager.stop()
        
        logger.info("Enhanced CSP Network stopped")
    
    async def _wait_for_shutdown(self):
        """Wait for shutdown signal."""
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

async def main():
    """Main function."""
    app = EnhancedCSPMain()
    
    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
