"""Network error definitions."""

class NetworkError(Exception):
    """Base network error."""
    pass

class ConnectionError(NetworkError):
    """Connection related error."""
    pass

class RoutingError(NetworkError):
    """Routing related error."""
    pass

class SecurityError(NetworkError):
    """Security related error."""
    pass

class ConfigurationError(NetworkError):
    """Configuration related error."""
    pass

class PeerError(NetworkError):
    """Peer related error."""
    pass

class MessageError(NetworkError):
    """Message related error."""
    pass

class TimeoutError(NetworkError):
    """Timeout related error."""
    pass
