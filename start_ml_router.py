#!/usr/bin/env python3
"""
Start ML Router with Enhanced CSP Network Integration
This script starts both the network and ML router together.
"""

import asyncio
import logging
import sys
import signal
import threading
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLRouterNetworkStarter:
    """Start ML Router with network integration."""
    
    def __init__(self):
        self.network_running = False
        self.ml_router_running = False
        self.shutdown_requested = False
        
    async def start_network_component(self):
        """Start the network component in background."""
        logger.info("🌐 Starting Enhanced CSP Network component...")
        
        try:
            from enhanced_csp.network.core.config import NetworkConfig
            from enhanced_csp.network.core.node import NetworkNode
            from enhanced_csp.network.utils.task_manager import TaskManager
            
            # Create network configuration
            config = NetworkConfig()
            config.node_name = "ml-router-network-node"
            config.node_type = "ai_service"
            
            # Create task manager
            task_manager = TaskManager()
            await task_manager.start()
            
            # Create network node
            network_node = NetworkNode(config)
            
            self.network_running = True
            logger.info("✅ Enhanced CSP Network component ready")
            
            # Start network node in background task
            network_task = asyncio.create_task(self._run_network_node(network_node))
            
            # Return the task so it can be managed
            return network_task, task_manager
            
        except ImportError as e:
            logger.warning(f"⚠️  Enhanced CSP imports failed: {e}")
            logger.info("🔄 Network will run in fallback mode")
            self.network_running = True
            return None, None
        except Exception as e:
            logger.error(f"❌ Network startup failed: {e}")
            return None, None
    
    async def _run_network_node(self, network_node):
        """Run network node with proper shutdown handling."""
        try:
            # Start the network node with a timeout check
            while not self.shutdown_requested:
                # Run for short periods and check for shutdown
                try:
                    await asyncio.wait_for(
                        network_node.start(), 
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    # This is expected - we want to check for shutdown periodically
                    if not self.shutdown_requested:
                        logger.debug("Network node running normally...")
                        continue
                    else:
                        break
                except Exception as e:
                    logger.error(f"Network node error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Network node failed: {e}")
        finally:
            await network_node.stop()
            logger.info("🛑 Network node stopped")
    
    def start_ml_router_flask(self):
        """Start the ML Router Flask application."""
        logger.info("🚀 Starting ML Router Flask application...")
        
        try:
            # Add project root to path
            project_root = Path.cwd()
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            # Import and start Flask app
            from app import app
            
            # Set environment for network mode
            import os
            os.environ['CSP_NETWORK_ENABLED'] = 'true'
            os.environ['ML_ROUTER_ENABLED'] = 'true'
            os.environ['NETWORK_BRIDGE_MODE'] = 'true'
            
            self.ml_router_running = True
            logger.info("✅ ML Router Flask application ready")
            
            # Start Flask in a separate thread
            flask_thread = threading.Thread(
                target=self._run_flask_app,
                args=(app,),
                daemon=True
            )
            flask_thread.start()
            
            return flask_thread
            
        except ImportError as e:
            logger.error(f"❌ Failed to import Flask app: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ ML Router startup failed: {e}")
            return None
    
    def _run_flask_app(self, app):
        """Run Flask app in thread."""
        try:
            app.run(
                host='0.0.0.0',
                port=5000,
                debug=False,
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            logger.error(f"Flask app error: {e}")
    
    async def start_combined_system(self):
        """Start both network and ML router together."""
        logger.info("🌟 Starting Combined ML Router + Network System")
        logger.info("=" * 60)
        
        try:
            # Start network component
            network_task, task_manager = await self.start_network_component()
            
            # Give network a moment to initialize
            await asyncio.sleep(2)
            
            # Start ML Router in background thread
            flask_thread = self.start_ml_router_flask()
            
            if flask_thread:
                logger.info("🎉 System startup complete!")
                logger.info("📡 Network endpoints:")
                logger.info("   - ML Router: http://localhost:5000")
                logger.info("   - Network Status: http://localhost:5000/api/network/status")
                logger.info("   - Network Dashboard: http://localhost:5000/network-dashboard")
                logger.info("📋 Press Ctrl+C to shutdown")
                
                # Wait for shutdown signal
                await self._wait_for_shutdown()
            else:
                logger.error("❌ Failed to start ML Router")
                
        except Exception as e:
            logger.error(f"❌ System startup failed: {e}")
        finally:
            await self._shutdown()
    
    async def _wait_for_shutdown(self):
        """Wait for shutdown signal."""
        try:
            while not self.shutdown_requested:
                await asyncio.sleep(1)
                
                # Check if both components are still running
                if not (self.network_running and self.ml_router_running):
                    logger.warning("⚠️  One or more components stopped unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            logger.info("📡 Shutdown signal received")
            self.shutdown_requested = True
    
    async def _shutdown(self):
        """Shutdown all components."""
        logger.info("⏹️  Shutting down ML Router + Network System...")
        self.shutdown_requested = True
        self.network_running = False
        self.ml_router_running = False
        
        # Give components time to shutdown gracefully
        await asyncio.sleep(2)
        
        logger.info("✅ Shutdown complete")

def setup_signal_handlers(starter):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"📡 Received signal {signum}")
        starter.shutdown_requested = True
    
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except AttributeError:
        # Windows doesn't have all signals
        signal.signal(signal.SIGINT, signal_handler)

async def main():
    """Main function."""
    print("🚀 ML Router + Enhanced CSP Network Starter")
    print("=" * 50)
    print("This will start your ML Router with full web4ai network integration.")
    print()
    
    starter = MLRouterNetworkStarter()
    setup_signal_handlers(starter)
    
    try:
        await starter.start_combined_system()
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
