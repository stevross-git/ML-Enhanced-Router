"""Network type definitions."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, List
import time
import uuid
import json

class MessageType(Enum):
    """Network message types."""
    PING = "ping"
    PONG = "pong"
    DATA = "data"
    QUERY = "query"
    RESPONSE = "response"
    DISCOVERY = "discovery"
    ROUTING = "routing"
    ML_QUERY = "ml_query"
    ML_RESPONSE = "ml_response"
    HEARTBEAT = "heartbeat"

@dataclass
class NodeID:
    """Network node identifier."""
    id: str
    
    def __init__(self, id_str: Optional[str] = None):
        self.id = id_str or str(uuid.uuid4())
    
    def __str__(self) -> str:
        return self.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, NodeID):
            return self.id == other.id
        return str(self) == str(other)
    
    def short_id(self) -> str:
        """Return shortened ID for display."""
        return self.id[:8]

@dataclass
class NetworkMessage:
    """Network message structure."""
    type: MessageType
    source: NodeID
    destination: Optional[NodeID]
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ttl: int = 10
    route_path: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "type": self.type.value,
            "source": str(self.source),
            "destination": str(self.destination) if self.destination else None,
            "data": self.data,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "ttl": self.ttl,
            "route_path": self.route_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkMessage':
        """Create message from dictionary."""
        return cls(
            type=MessageType(data["type"]),
            source=NodeID(data["source"]),
            destination=NodeID(data["destination"]) if data.get("destination") else None,
            data=data["data"],
            timestamp=data.get("timestamp", time.time()),
            message_id=data.get("message_id", str(uuid.uuid4())),
            ttl=data.get("ttl", 10),
            route_path=data.get("route_path", []),
        )

@dataclass
class PeerInfo:
    """Peer information."""
    node_id: NodeID
    address: str
    port: int
    capabilities: Dict[str, Any] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)
    reputation: float = 1.0
    latency: Optional[float] = None
    
    def is_alive(self, timeout: float = 300) -> bool:
        """Check if peer is considered alive."""
        return time.time() - self.last_seen < timeout
    
    def update_last_seen(self):
        """Update last seen timestamp."""
        self.last_seen = time.time()

class NodeCapabilities:
    """Node capability definitions."""
    
    def __init__(self, capabilities: Optional[Dict[str, bool]] = None):
        self.capabilities = capabilities or {
            "routing": True,
            "storage": False,
            "compute": False,
            "ai_service": False,
            "ml_router": False,
            "dns_resolver": False,
        }
    
    def set_capability(self, name: str, enabled: bool):
        """Set a capability."""
        self.capabilities[name] = enabled
    
    def has_capability(self, name: str) -> bool:
        """Check if node has capability."""
        return self.capabilities.get(name, False)
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get all capabilities."""
        return self.capabilities.copy()
    
    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary."""
        return self.capabilities
    
    @classmethod
    def from_dict(cls, data: Dict[str, bool]) -> 'NodeCapabilities':
        """Create from dictionary."""
        return cls(data)
