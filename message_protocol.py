"""
Core message protocol implementation for AI-to-AI communication
Handles Protocol Buffers serialization/deserialization and message validation
"""

import time
import hashlib
import uuid
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

# Protocol Buffers imports (generated from .proto file)
try:
    import protocol_buffers_pb2 as pb
except ImportError:
    # Fallback implementation if protobuf not available
    print("Warning: Protocol Buffers not available, using fallback implementation")
    pb = None

logger = logging.getLogger(__name__)

class MessageType(Enum):
    UNKNOWN = 0
    QUERY = 1
    RESPONSE = 2
    CONTEXT_SHARE = 3
    SYSTEM_STATUS = 4
    ERROR = 5
    HEARTBEAT = 6
    BULK_TRANSFER = 7
    EMERGENCY = 8

class Priority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4

class ErrorSeverity(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3
    FATAL = 4

@dataclass
class AIMessage:
    """Core AI communication message structure"""
    msg_id: int
    type: MessageType
    payload: bytes
    checksum: int
    context_ref: int
    timestamp: int
    sender_id: str
    recipient_id: str
    priority: Priority = Priority.NORMAL
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class ContextReference:
    """Context reference metadata"""
    context_id: int
    context_key: str
    created_at: int
    expires_at: int
    access_count: int = 0
    associated_agents: List[str] = None
    
    def __post_init__(self):
        if self.associated_agents is None:
            self.associated_agents = []

@dataclass
class SystemStatus:
    """System status information"""
    agent_id: str
    is_healthy: bool
    cpu_usage: float
    memory_usage: float
    active_connections: int
    pending_messages: int
    status_message: str
    last_heartbeat: int

@dataclass
class ErrorMessage:
    """Error message structure"""
    error_code: str
    error_message: str
    stack_trace: str
    original_msg_id: int
    severity: ErrorSeverity
    is_recoverable: bool

class MessageProtocol:
    """Main protocol handler for AI communication messages"""
    
    def __init__(self):
        self.message_counter = 0
        self.protocol_version = "1.0"
        
    def create_message(
        self,
        msg_type: MessageType,
        payload: Union[str, bytes, Dict[str, Any]],
        sender_id: str,
        recipient_id: str,
        context_ref: int = 0,
        priority: Priority = Priority.NORMAL,
        tags: List[str] = None
    ) -> AIMessage:
        """Create a new AI communication message"""
        
        # Convert payload to bytes if needed
        if isinstance(payload, str):
            payload_bytes = payload.encode('utf-8')
        elif isinstance(payload, dict):
            payload_bytes = json.dumps(payload).encode('utf-8')
        else:
            payload_bytes = payload
            
        # Generate message ID
        self.message_counter += 1
        msg_id = self.message_counter
        
        # Calculate checksum
        checksum = self._calculate_checksum(payload_bytes)
        
        # Create message
        message = AIMessage(
            msg_id=msg_id,
            type=msg_type,
            payload=payload_bytes,
            checksum=checksum,
            context_ref=context_ref,
            timestamp=int(time.time() * 1000),  # milliseconds
            sender_id=sender_id,
            recipient_id=recipient_id,
            priority=priority,
            tags=tags or []
        )
        
        logger.info(f"Created message {msg_id} from {sender_id} to {recipient_id}")
        return message
    
    def serialize_message(self, message: AIMessage) -> bytes:
        """Serialize message to binary format"""
        if pb:
            return self._serialize_protobuf(message)
        else:
            return self._serialize_json(message)
    
    def deserialize_message(self, data: bytes) -> AIMessage:
        """Deserialize binary data to message"""
        if pb:
            return self._deserialize_protobuf(data)
        else:
            return self._deserialize_json(data)
    
    def validate_message(self, message: AIMessage) -> bool:
        """Validate message integrity and format"""
        try:
            # Check checksum
            calculated_checksum = self._calculate_checksum(message.payload)
            if calculated_checksum != message.checksum:
                logger.error(f"Checksum mismatch for message {message.msg_id}")
                return False
            
            # Check required fields
            if not message.sender_id or not message.recipient_id:
                logger.error(f"Missing sender or recipient for message {message.msg_id}")
                return False
            
            # Check timestamp is recent (within last hour for regular messages)
            current_time = int(time.time() * 1000)
            if message.type != MessageType.EMERGENCY:
                if abs(current_time - message.timestamp) > 3600000:  # 1 hour
                    logger.warning(f"Message {message.msg_id} has old timestamp")
            
            # Check payload size limits
            payload_size = len(message.payload)
            max_size = self._get_max_payload_size(message.type)
            if payload_size > max_size:
                logger.error(f"Message {message.msg_id} payload too large: {payload_size} > {max_size}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating message: {e}")
            return False
    
    def create_heartbeat(self, agent_id: str) -> AIMessage:
        """Create a heartbeat message"""
        return self.create_message(
            MessageType.HEARTBEAT,
            {"status": "alive", "timestamp": int(time.time())},
            agent_id,
            "system",
            priority=Priority.LOW
        )
    
    def create_error_response(
        self,
        original_msg_id: int,
        error_code: str,
        error_message: str,
        sender_id: str,
        recipient_id: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        is_recoverable: bool = True
    ) -> AIMessage:
        """Create an error response message"""
        
        error_data = ErrorMessage(
            error_code=error_code,
            error_message=error_message,
            stack_trace="",
            original_msg_id=original_msg_id,
            severity=severity,
            is_recoverable=is_recoverable
        )
        
        return self.create_message(
            MessageType.ERROR,
            {
                "error_code": error_data.error_code,
                "error_message": error_data.error_message,
                "original_msg_id": error_data.original_msg_id,
                "severity": error_data.severity.value,
                "is_recoverable": error_data.is_recoverable
            },
            sender_id,
            recipient_id,
            priority=Priority.HIGH
        )
    
    def _calculate_checksum(self, data: bytes) -> int:
        """Calculate CRC32 checksum for data integrity"""
        import zlib
        return zlib.crc32(data) & 0xffffffff
    
    def _get_max_payload_size(self, msg_type: MessageType) -> int:
        """Get maximum payload size based on message type"""
        size_limits = {
            MessageType.EMERGENCY: 512 * 4,  # 512 tokens * 4 bytes/token
            MessageType.HEARTBEAT: 256 * 4,
            MessageType.ERROR: 1024 * 4,
            MessageType.SYSTEM_STATUS: 1024 * 4,
            MessageType.QUERY: 2048 * 4,
            MessageType.RESPONSE: 2048 * 4,
            MessageType.CONTEXT_SHARE: 2048 * 4,
            MessageType.BULK_TRANSFER: 8192 * 4,
        }
        return size_limits.get(msg_type, 2048 * 4)
    
    def _serialize_protobuf(self, message: AIMessage) -> bytes:
        """Serialize using Protocol Buffers"""
        if not pb:
            raise RuntimeError("Protocol Buffers not available")
            
        pb_message = pb.AIMessage()
        pb_message.msg_id = message.msg_id
        pb_message.type = message.type.value
        pb_message.payload = message.payload
        pb_message.checksum = message.checksum
        pb_message.context_ref = message.context_ref
        pb_message.timestamp = message.timestamp
        pb_message.sender_id = message.sender_id
        pb_message.recipient_id = message.recipient_id
        pb_message.priority = message.priority.value
        pb_message.tags.extend(message.tags)
        
        return pb_message.SerializeToString()
    
    def _deserialize_protobuf(self, data: bytes) -> AIMessage:
        """Deserialize from Protocol Buffers"""
        if not pb:
            raise RuntimeError("Protocol Buffers not available")
            
        pb_message = pb.AIMessage()
        pb_message.ParseFromString(data)
        
        return AIMessage(
            msg_id=pb_message.msg_id,
            type=MessageType(pb_message.type),
            payload=pb_message.payload,
            checksum=pb_message.checksum,
            context_ref=pb_message.context_ref,
            timestamp=pb_message.timestamp,
            sender_id=pb_message.sender_id,
            recipient_id=pb_message.recipient_id,
            priority=Priority(pb_message.priority),
            tags=list(pb_message.tags)
        )
    
    def _serialize_json(self, message: AIMessage) -> bytes:
        """Fallback JSON serialization"""
        data = {
            "msg_id": message.msg_id,
            "type": message.type.value,
            "payload": message.payload.hex(),  # Convert bytes to hex string
            "checksum": message.checksum,
            "context_ref": message.context_ref,
            "timestamp": message.timestamp,
            "sender_id": message.sender_id,
            "recipient_id": message.recipient_id,
            "priority": message.priority.value,
            "tags": message.tags,
            "protocol_version": self.protocol_version
        }
        return json.dumps(data).encode('utf-8')
    
    def _deserialize_json(self, data: bytes) -> AIMessage:
        """Fallback JSON deserialization"""
        json_data = json.loads(data.decode('utf-8'))
        
        return AIMessage(
            msg_id=json_data["msg_id"],
            type=MessageType(json_data["type"]),
            payload=bytes.fromhex(json_data["payload"]),
            checksum=json_data["checksum"],
            context_ref=json_data["context_ref"],
            timestamp=json_data["timestamp"],
            sender_id=json_data["sender_id"],
            recipient_id=json_data["recipient_id"],
            priority=Priority(json_data["priority"]),
            tags=json_data["tags"]
        )

# Utility functions for message handling
def create_query_message(query: str, sender_id: str, recipient_id: str, context_ref: int = 0) -> AIMessage:
    """Convenience function to create query messages"""
    protocol = MessageProtocol()
    return protocol.create_message(
        MessageType.QUERY,
        {"query": query, "timestamp": int(time.time())},
        sender_id,
        recipient_id,
        context_ref=context_ref
    )

def create_response_message(response: str, sender_id: str, recipient_id: str, original_msg_id: int) -> AIMessage:
    """Convenience function to create response messages"""
    protocol = MessageProtocol()
    return protocol.create_message(
        MessageType.RESPONSE,
        {"response": response, "original_msg_id": original_msg_id, "timestamp": int(time.time())},
        sender_id,
        recipient_id
    )