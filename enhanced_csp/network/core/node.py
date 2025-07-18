"""Network node implementation."""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any

# Import types with proper error handling
try:
    from .config import NetworkConfig
    from .types import NodeID, NetworkMessage, MessageType, PeerInfo, NodeCapabilities
except ImportError as e:
    # Fallback imports for when modules aren't fully set up
    logging.warning(f"Import warning in node.py: {e}")
    # Create minimal stubs
    class NetworkConfig:
        def __init__(self):
            self.node_name = "default-node"
            self.node_type = "standard"
    
    class NodeID:
        def __init__(self, name="default"):
            self.id = name
        def __str__(self):
            return self.id
    
    class MessageType:
        PING = "ping"
        PONG = "pong"
        DISCOVERY = "discovery"
        HEARTBEAT = "heartbeat"
    
    class NetworkMessage:
        def __init__(self, type, source, destination, data):
            self.type = type
            self.source = source
            self.destination = destination
            self.data = data
    
    class PeerInfo:
        def __init__(self, node_id, address, port):
            self.node_id = node_id
            self.address = address
            self.port = port
            self.last_seen = time.time()
        
        def is_alive(self, timeout=300):
            return time.time() - self.last_seen < timeout
        
        def update_last_seen(self):
            self.last_seen = time.time()
    
    class NodeCapabilities:
        def __init__(self):
            self.capabilities = {}
        def to_dict(self):
            return self.capabilities

logger = logging.getLogger(__name__)

class NetworkNode:
    """Base network node implementation."""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.node_id = NodeID(config.node_name)
        self.capabilities = NodeCapabilities()
        
        # Node state
        self.running = False
        self.peers: Dict[str, PeerInfo] = {}
        self.message_handlers: Dict = {}
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.start_time = time.time()
        
        logger.info(f"NetworkNode initialized: {self.node_id}")
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        try:
            self.register_handler(MessageType.PING, self._handle_ping)
            self.register_handler(MessageType.PONG, self._handle_pong)
            self.register_handler(MessageType.DISCOVERY, self._handle_discovery)
            self.register_handler(MessageType.HEARTBEAT, self._handle_heartbeat)
        except Exception as e:
            logger.warning(f"Failed to register some handlers: {e}")
    
    def register_handler(self, message_type, handler: Callable):
        """Register a message handler."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type}")
    
    async def start(self):
        """Start the network node."""
        logger.info(f"Starting network node: {self.node_id}")
        self.running = True
        self.start_time = time.time()
        
        # Start background tasks
        tasks = []
        try:
            tasks.append(asyncio.create_task(self._heartbeat_loop()))
            tasks.append(asyncio.create_task(self._peer_maintenance_loop()))
            
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Node error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the network node."""
        logger.info(f"Stopping network node: {self.node_id}")
        self.running = False
    
    async def send_message(self, message) -> bool:
        """Send a message to the network."""
        try:
            self.messages_sent += 1
            logger.debug(f"Sending {getattr(message, 'type', 'unknown')} to {getattr(message, 'destination', 'unknown')}")
            # Implementation would send via transport layer
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def handle_message(self, message):
        """Handle incoming message."""
        self.messages_received += 1
        message_type = getattr(message, 'type', 'unknown')
        source = getattr(message, 'source', 'unknown')
        logger.debug(f"Received {message_type} from {source}")
        
        handler = self.message_handlers.get(message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Handler error for {message_type}: {e}")
        else:
            logger.warning(f"No handler for {message_type}")
    
    async def _handle_ping(self, message):
        """Handle ping message."""
        try:
            pong = NetworkMessage(
                type=MessageType.PONG,
                source=self.node_id,
                destination=message.source,
                data={"timestamp": time.time()}
            )
            await self.send_message(pong)
        except Exception as e:
            logger.error(f"Failed to handle ping: {e}")
    
    async def _handle_pong(self, message):
        """Handle pong message."""
        try:
            # Update peer latency
            source_str = str(message.source)
            if source_str in self.peers:
                peer = self.peers[source_str]
                peer.latency = time.time() - message.data.get("timestamp", 0)
                peer.update_last_seen()
        except Exception as e:
            logger.error(f"Failed to handle pong: {e}")
    
    async def _handle_discovery(self, message):
        """Handle peer discovery message."""
        try:
            peer_info = PeerInfo(
                node_id=message.source,
                address=message.data.get("address", "unknown"),
                port=message.data.get("port", 0),
            )
            # Add capabilities if available
            if hasattr(peer_info, 'capabilities'):
                peer_info.capabilities = message.data.get("capabilities", {})
            
            self.peers[str(message.source)] = peer_info
            logger.info(f"Discovered peer: {message.source}")
        except Exception as e:
            logger.error(f"Failed to handle discovery: {e}")
    
    async def _handle_heartbeat(self, message):
        """Handle heartbeat message."""
        try:
            source_str = str(message.source)
            if source_str in self.peers:
                self.peers[source_str].update_last_seen()
        except Exception as e:
            logger.error(f"Failed to handle heartbeat: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self.running:
            try:
                heartbeat = NetworkMessage(
                    type=MessageType.HEARTBEAT,
                    source=self.node_id,
                    destination=None,  # Broadcast
                    data={"timestamp": time.time()}
                )
                await self.send_message(heartbeat)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            
            await asyncio.sleep(30)  # Heartbeat every 30 seconds
    
    async def _peer_maintenance_loop(self):
        """Maintain peer list."""
        while self.running:
            try:
                # Remove stale peers
                current_time = time.time()
                stale_peers = []
                
                for peer_id, peer in list(self.peers.items()):
                    try:
                        if not peer.is_alive():
                            stale_peers.append(peer_id)
                    except Exception:
                        # If peer check fails, consider it stale
                        stale_peers.append(peer_id)
                
                for peer_id in stale_peers:
                    try:
                        logger.info(f"Removing stale peer: {peer_id}")
                        del self.peers[peer_id]
                    except Exception as e:
                        logger.error(f"Failed to remove peer {peer_id}: {e}")
                        
            except Exception as e:
                logger.error(f"Peer maintenance error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        try:
            uptime = time.time() - self.start_time
            return {
                "node_id": str(self.node_id),
                "uptime": uptime,
                "peer_count": len(self.peers),
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "capabilities": self.capabilities.to_dict() if hasattr(self.capabilities, 'to_dict') else {},
                "running": self.running,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "node_id": str(self.node_id),
                "error": str(e),
                "running": self.running,
            }

# Make sure NetworkNode is properly exported
__all__ = ["NetworkNode"]
