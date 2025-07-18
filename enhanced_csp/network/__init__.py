"""Enhanced CSP Network Module."""
__version__ = "1.0.0"

# Import core components with error handling
try:
    from .core.config import NetworkConfig, P2PConfig, SecurityConfig
    from .core.types import NodeID, MessageType, NetworkMessage
    from .core.node import NetworkNode
    
    __all__ = [
        "NetworkConfig", "P2PConfig", "SecurityConfig",
        "NodeID", "MessageType", "NetworkMessage", "NetworkNode"
    ]
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Some imports failed: {e}")
    __all__ = []

# Lazy loading for advanced components
def __getattr__(name):
    if name == "NetworkConfig":
        from .core.config import NetworkConfig
        return NetworkConfig
    elif name == "NodeID":
        from .core.types import NodeID  
        return NodeID
    elif name == "MessageType":
        from .core.types import MessageType
        return MessageType
    elif name == "NetworkNode":
        from .core.node import NetworkNode
        return NetworkNode
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
