"""
Decentralized P2P AI Routing System
Implements gossip protocol, fault-tolerant agent discovery, and network topology-based routing
"""

import os
import json
import time
import uuid
import socket
import threading
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
import sqlite3
from contextlib import contextmanager
import hashlib
import random
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    """Node status in the P2P network"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPECTED = "suspected"
    FAILED = "failed"
    JOINING = "joining"
    LEAVING = "leaving"

class MessageType(Enum):
    """P2P message types"""
    PING = "ping"
    PONG = "pong"
    GOSSIP = "gossip"
    AGENT_DISCOVERY = "agent_discovery"
    AGENT_RESPONSE = "agent_response"
    QUERY_ROUTE = "query_route"
    QUERY_RESPONSE = "query_response"
    HEARTBEAT = "heartbeat"
    JOIN_REQUEST = "join_request"
    JOIN_RESPONSE = "join_response"
    LEAVE_NOTIFICATION = "leave_notification"
    CIRCUIT_BREAK = "circuit_break"
    CIRCUIT_RESET = "circuit_reset"

class AgentCapability(Enum):
    """Agent capabilities in the network"""
    TEXT_GENERATION = "text_generation"
    IMAGE_ANALYSIS = "image_analysis"
    CODE_GENERATION = "code_generation"
    MATH_SOLVING = "math_solving"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    CREATIVE_WRITING = "creative_writing"
    TECHNICAL_ANALYSIS = "technical_analysis"
    RESEARCH = "research"

@dataclass
class P2PNode:
    """Represents a peer node in the network"""
    node_id: str
    address: str
    port: int
    status: NodeStatus
    last_seen: datetime
    capabilities: List[AgentCapability]
    load: float = 0.0
    latency: float = 0.0
    reliability: float = 1.0
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class P2PMessage:
    """P2P network message"""
    message_id: str
    message_type: MessageType
    sender_id: str
    receiver_id: Optional[str]
    payload: Dict[str, Any]
    timestamp: datetime
    ttl: int = 3
    signature: Optional[str] = None

@dataclass
class AgentDiscoveryInfo:
    """Agent discovery information"""
    agent_id: str
    node_id: str
    capabilities: List[AgentCapability]
    load: float
    performance_metrics: Dict[str, float]
    last_updated: datetime

@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    node_id: str
    failure_count: int = 0
    failure_threshold: int = 5
    timeout_duration: int = 60
    last_failure: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

class GossipProtocol:
    """Gossip protocol for distributed information sharing"""
    
    def __init__(self, node_id: str, max_peers: int = 3):
        self.node_id = node_id
        self.max_peers = max_peers
        self.gossip_interval = 5  # seconds
        self.message_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.running = False
        self.gossip_thread = None
    
    def start_gossip(self, peer_manager):
        """Start the gossip protocol"""
        self.running = True
        self.peer_manager = peer_manager
        self.gossip_thread = threading.Thread(target=self._gossip_loop, daemon=True)
        self.gossip_thread.start()
        logger.info(f"Gossip protocol started for node {self.node_id}")
    
    def stop_gossip(self):
        """Stop the gossip protocol"""
        self.running = False
        if self.gossip_thread:
            self.gossip_thread.join()
        logger.info(f"Gossip protocol stopped for node {self.node_id}")
    
    def _gossip_loop(self):
        """Main gossip loop"""
        while self.running:
            try:
                self._perform_gossip_round()
                time.sleep(self.gossip_interval)
            except Exception as e:
                logger.error(f"Error in gossip loop: {e}")
    
    def _perform_gossip_round(self):
        """Perform one round of gossip"""
        # Clean expired messages
        self._clean_message_cache()
        
        # Select random peers to gossip with
        active_peers = self.peer_manager.get_active_peers()
        if not active_peers:
            return
        
        gossip_peers = random.sample(
            active_peers, 
            min(self.max_peers, len(active_peers))
        )
        
        # Prepare gossip message
        gossip_data = {
            "node_status": self.peer_manager.get_node_status(),
            "agent_info": self.peer_manager.get_local_agent_info(),
            "network_topology": self.peer_manager.get_network_topology(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Send gossip to selected peers
        for peer in gossip_peers:
            try:
                self.peer_manager.send_message(
                    peer.node_id,
                    MessageType.GOSSIP,
                    gossip_data
                )
            except Exception as e:
                logger.error(f"Failed to gossip with {peer.node_id}: {e}")
    
    def process_gossip_message(self, message: P2PMessage):
        """Process incoming gossip message"""
        try:
            # Check if message already processed
            if message.message_id in self.message_cache:
                return
            
            # Cache the message
            self.message_cache[message.message_id] = {
                "timestamp": datetime.now(),
                "processed": True
            }
            
            # Process gossip data
            payload = message.payload
            self.peer_manager.update_network_info(payload)
            
            # Propagate to other peers (with decreased TTL)
            if message.ttl > 1:
                self._propagate_gossip(message)
                
        except Exception as e:
            logger.error(f"Error processing gossip message: {e}")
    
    def _propagate_gossip(self, original_message: P2PMessage):
        """Propagate gossip to other peers"""
        # Select peers to propagate to (excluding sender)
        active_peers = [
            peer for peer in self.peer_manager.get_active_peers()
            if peer.node_id != original_message.sender_id
        ]
        
        if not active_peers:
            return
        
        # Propagate to subset of peers
        propagation_peers = random.sample(
            active_peers,
            min(self.max_peers, len(active_peers))
        )
        
        for peer in propagation_peers:
            try:
                new_message = P2PMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.GOSSIP,
                    sender_id=self.node_id,
                    receiver_id=peer.node_id,
                    payload=original_message.payload,
                    timestamp=datetime.now(),
                    ttl=original_message.ttl - 1
                )
                
                self.peer_manager.send_message_direct(peer, new_message)
                
            except Exception as e:
                logger.error(f"Failed to propagate gossip to {peer.node_id}: {e}")
    
    def _clean_message_cache(self):
        """Clean expired messages from cache"""
        current_time = datetime.now()
        expired_messages = []
        
        for msg_id, msg_data in self.message_cache.items():
            if (current_time - msg_data["timestamp"]).total_seconds() > self.cache_ttl:
                expired_messages.append(msg_id)
        
        for msg_id in expired_messages:
            del self.message_cache[msg_id]

class FaultTolerantDiscovery:
    """Fault-tolerant agent discovery system"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.discovered_agents = {}
        self.circuit_breakers = {}
        self.discovery_interval = 30  # seconds
        self.running = False
        self.discovery_thread = None
    
    def start_discovery(self, peer_manager):
        """Start the discovery system"""
        self.running = True
        self.peer_manager = peer_manager
        self.discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
        self.discovery_thread.start()
        logger.info(f"Fault-tolerant discovery started for node {self.node_id}")
    
    def stop_discovery(self):
        """Stop the discovery system"""
        self.running = False
        if self.discovery_thread:
            self.discovery_thread.join()
        logger.info(f"Fault-tolerant discovery stopped for node {self.node_id}")
    
    def _discovery_loop(self):
        """Main discovery loop"""
        while self.running:
            try:
                self._perform_discovery_round()
                time.sleep(self.discovery_interval)
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
    
    def _perform_discovery_round(self):
        """Perform one round of agent discovery"""
        # Send discovery requests to active peers
        active_peers = self.peer_manager.get_active_peers()
        
        for peer in active_peers:
            # Check circuit breaker
            if self._is_circuit_open(peer.node_id):
                continue
            
            try:
                self.peer_manager.send_message(
                    peer.node_id,
                    MessageType.AGENT_DISCOVERY,
                    {"requesting_node": self.node_id}
                )
            except Exception as e:
                self._record_failure(peer.node_id)
                logger.error(f"Failed to send discovery request to {peer.node_id}: {e}")
    
    def process_discovery_request(self, message: P2PMessage):
        """Process agent discovery request"""
        try:
            # Get local agent information
            local_agents = self.peer_manager.get_local_agent_info()
            
            # Send response back to requester
            response_payload = {
                "agents": local_agents,
                "node_id": self.node_id,
                "timestamp": datetime.now().isoformat()
            }
            
            self.peer_manager.send_message(
                message.sender_id,
                MessageType.AGENT_RESPONSE,
                response_payload
            )
            
        except Exception as e:
            logger.error(f"Error processing discovery request: {e}")
    
    def process_discovery_response(self, message: P2PMessage):
        """Process agent discovery response"""
        try:
            payload = message.payload
            sender_id = message.sender_id
            
            # Update discovered agents
            for agent_info in payload.get("agents", []):
                agent_id = f"{sender_id}:{agent_info['agent_id']}"
                
                discovery_info = AgentDiscoveryInfo(
                    agent_id=agent_id,
                    node_id=sender_id,
                    capabilities=[AgentCapability(cap) for cap in agent_info.get("capabilities", [])],
                    load=agent_info.get("load", 0.0),
                    performance_metrics=agent_info.get("performance_metrics", {}),
                    last_updated=datetime.now()
                )
                
                self.discovered_agents[agent_id] = discovery_info
            
            # Reset circuit breaker on successful response
            self._reset_circuit_breaker(sender_id)
            
        except Exception as e:
            logger.error(f"Error processing discovery response: {e}")
    
    def _is_circuit_open(self, node_id: str) -> bool:
        """Check if circuit breaker is open for a node"""
        if node_id not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[node_id]
        
        if breaker.state == "CLOSED":
            return False
        elif breaker.state == "OPEN":
            # Check if timeout has passed
            if breaker.last_failure:
                time_since_failure = (datetime.now() - breaker.last_failure).total_seconds()
                if time_since_failure > breaker.timeout_duration:
                    breaker.state = "HALF_OPEN"
                    return False
            return True
        elif breaker.state == "HALF_OPEN":
            return False
        
        return False
    
    def _record_failure(self, node_id: str):
        """Record a failure for circuit breaker"""
        if node_id not in self.circuit_breakers:
            self.circuit_breakers[node_id] = CircuitBreaker(node_id)
        
        breaker = self.circuit_breakers[node_id]
        breaker.failure_count += 1
        breaker.last_failure = datetime.now()
        
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.state = "OPEN"
            logger.warning(f"Circuit breaker opened for node {node_id}")
    
    def _reset_circuit_breaker(self, node_id: str):
        """Reset circuit breaker on successful operation"""
        if node_id in self.circuit_breakers:
            breaker = self.circuit_breakers[node_id]
            breaker.failure_count = 0
            breaker.state = "CLOSED"
            logger.info(f"Circuit breaker reset for node {node_id}")
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[AgentDiscoveryInfo]:
        """Get agents by capability"""
        current_time = datetime.now()
        valid_agents = []
        
        for agent_info in self.discovered_agents.values():
            # Check if agent info is still valid (not too old)
            if (current_time - agent_info.last_updated).total_seconds() < 300:  # 5 minutes
                if capability in agent_info.capabilities:
                    valid_agents.append(agent_info)
        
        # Sort by load and performance
        valid_agents.sort(key=lambda x: (x.load, -x.performance_metrics.get("response_time", 1.0)))
        
        return valid_agents

class NetworkTopologyManager:
    """Manages network topology and routing decisions"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.topology = {}
        self.routing_table = {}
        self.network_lock = threading.Lock()
    
    def update_topology(self, topology_data: Dict[str, Any]):
        """Update network topology from gossip data"""
        with self.network_lock:
            try:
                # Update node information
                for node_id, node_info in topology_data.items():
                    self.topology[node_id] = {
                        "status": node_info.get("status", NodeStatus.ACTIVE.value),
                        "capabilities": node_info.get("capabilities", []),
                        "load": node_info.get("load", 0.0),
                        "latency": node_info.get("latency", 0.0),
                        "last_seen": datetime.fromisoformat(node_info.get("last_seen", datetime.now().isoformat()))
                    }
                
                # Rebuild routing table
                self._rebuild_routing_table()
                
            except Exception as e:
                logger.error(f"Error updating topology: {e}")
    
    def _rebuild_routing_table(self):
        """Rebuild routing table based on current topology"""
        self.routing_table = {}
        
        # Calculate shortest paths using simple distance algorithm
        for destination in self.topology:
            if destination != self.node_id:
                path = self._find_shortest_path(self.node_id, destination)
                if path:
                    self.routing_table[destination] = {
                        "next_hop": path[1] if len(path) > 1 else destination,
                        "distance": len(path) - 1,
                        "path": path
                    }
    
    def _find_shortest_path(self, source: str, destination: str) -> List[str]:
        """Find shortest path between two nodes"""
        if source == destination:
            return [source]
        
        # Simple BFS for shortest path
        queue = [(source, [source])]
        visited = {source}
        
        while queue:
            current_node, path = queue.pop(0)
            
            # Get neighbors (nodes with active status)
            neighbors = self._get_neighbors(current_node)
            
            for neighbor in neighbors:
                if neighbor == destination:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def _get_neighbors(self, node_id: str) -> List[str]:
        """Get active neighbors of a node"""
        neighbors = []
        
        for other_node_id, node_info in self.topology.items():
            if (other_node_id != node_id and 
                node_info.get("status") == NodeStatus.ACTIVE.value):
                neighbors.append(other_node_id)
        
        return neighbors
    
    def get_best_route(self, capability: AgentCapability) -> Optional[str]:
        """Get best route for a specific capability"""
        best_nodes = []
        
        for node_id, node_info in self.topology.items():
            if (node_id != self.node_id and 
                capability.value in node_info.get("capabilities", []) and
                node_info.get("status") == NodeStatus.ACTIVE.value):
                
                # Calculate routing score
                load_factor = 1.0 - node_info.get("load", 0.0)
                latency_factor = 1.0 / (1.0 + node_info.get("latency", 1.0))
                distance_factor = 1.0 / (1.0 + self.routing_table.get(node_id, {}).get("distance", 1))
                
                score = load_factor * latency_factor * distance_factor
                
                best_nodes.append((node_id, score))
        
        if best_nodes:
            best_nodes.sort(key=lambda x: x[1], reverse=True)
            return best_nodes[0][0]
        
        return None
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        with self.network_lock:
            active_nodes = sum(1 for node_info in self.topology.values() 
                             if node_info.get("status") == NodeStatus.ACTIVE.value)
            
            total_load = sum(node_info.get("load", 0.0) for node_info in self.topology.values())
            avg_load = total_load / len(self.topology) if self.topology else 0.0
            
            return {
                "total_nodes": len(self.topology),
                "active_nodes": active_nodes,
                "average_load": avg_load,
                "routing_table_size": len(self.routing_table),
                "network_diameter": self._calculate_network_diameter()
            }
    
    def _calculate_network_diameter(self) -> int:
        """Calculate network diameter (longest shortest path)"""
        max_distance = 0
        
        for route_info in self.routing_table.values():
            distance = route_info.get("distance", 0)
            max_distance = max(max_distance, distance)
        
        return max_distance

class P2PNetworkManager:
    """Main P2P network manager"""
    
    def __init__(self, node_id: str, port: int, capabilities: List[AgentCapability]):
        self.node_id = node_id
        self.port = port
        self.capabilities = capabilities
        self.peers = {}
        self.local_agents = {}
        
        # Initialize components
        self.gossip_protocol = GossipProtocol(node_id)
        self.discovery_system = FaultTolerantDiscovery(node_id)
        self.topology_manager = NetworkTopologyManager(node_id)
        
        # Network state
        self.running = False
        self.server_socket = None
        self.server_thread = None
        
        # Database for persistence
        self.db_path = f"p2p_network_{node_id}.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize P2P network database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS peers (
                    node_id TEXT PRIMARY KEY,
                    address TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    capabilities TEXT,
                    last_seen TIMESTAMP,
                    load REAL DEFAULT 0.0,
                    latency REAL DEFAULT 0.0,
                    reliability REAL DEFAULT 1.0
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    message_type TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    receiver_id TEXT,
                    payload TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS routing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    source_node TEXT NOT NULL,
                    destination_node TEXT NOT NULL,
                    capability TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    latency REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def start_network(self):
        """Start the P2P network"""
        try:
            self.running = True
            
            # Start network server
            self._start_server()
            
            # Start network components
            self.gossip_protocol.start_gossip(self)
            self.discovery_system.start_discovery(self)
            
            logger.info(f"P2P network started for node {self.node_id} on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start P2P network: {e}")
            self.stop_network()
    
    def stop_network(self):
        """Stop the P2P network"""
        self.running = False
        
        # Stop components
        self.gossip_protocol.stop_gossip()
        self.discovery_system.stop_discovery()
        
        # Stop server
        if self.server_socket:
            self.server_socket.close()
        
        if self.server_thread:
            self.server_thread.join()
        
        logger.info(f"P2P network stopped for node {self.node_id}")
    
    def _start_server(self):
        """Start the network server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(5)
        
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
    
    def _server_loop(self):
        """Main server loop"""
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
            except Exception as e:
                if self.running:
                    logger.error(f"Server error: {e}")
    
    def _handle_client(self, client_socket, address):
        """Handle client connection"""
        try:
            # Read message length
            length_data = client_socket.recv(4)
            if not length_data:
                return
            
            message_length = struct.unpack('!I', length_data)[0]
            
            # Read message data
            message_data = b''
            while len(message_data) < message_length:
                chunk = client_socket.recv(message_length - len(message_data))
                if not chunk:
                    break
                message_data += chunk
            
            # Parse message
            message_json = json.loads(message_data.decode('utf-8'))
            message = P2PMessage(
                message_id=message_json['message_id'],
                message_type=MessageType(message_json['message_type']),
                sender_id=message_json['sender_id'],
                receiver_id=message_json.get('receiver_id'),
                payload=message_json['payload'],
                timestamp=datetime.fromisoformat(message_json['timestamp']),
                ttl=message_json.get('ttl', 3)
            )
            
            # Process message
            self._process_message(message)
            
        except Exception as e:
            logger.error(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
    
    def _process_message(self, message: P2PMessage):
        """Process incoming P2P message"""
        try:
            if message.message_type == MessageType.GOSSIP:
                self.gossip_protocol.process_gossip_message(message)
            elif message.message_type == MessageType.AGENT_DISCOVERY:
                self.discovery_system.process_discovery_request(message)
            elif message.message_type == MessageType.AGENT_RESPONSE:
                self.discovery_system.process_discovery_response(message)
            elif message.message_type == MessageType.QUERY_ROUTE:
                self._process_query_route(message)
            elif message.message_type == MessageType.PING:
                self._process_ping(message)
            elif message.message_type == MessageType.JOIN_REQUEST:
                self._process_join_request(message)
            elif message.message_type == MessageType.HEARTBEAT:
                self._process_heartbeat(message)
            
        except Exception as e:
            logger.error(f"Error processing message {message.message_type}: {e}")
    
    def send_message(self, target_node_id: str, message_type: MessageType, payload: Dict[str, Any]):
        """Send message to target node"""
        if target_node_id not in self.peers:
            logger.error(f"Target node {target_node_id} not found in peers")
            return
        
        message = P2PMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=self.node_id,
            receiver_id=target_node_id,
            payload=payload,
            timestamp=datetime.now()
        )
        
        target_peer = self.peers[target_node_id]
        self.send_message_direct(target_peer, message)
    
    def send_message_direct(self, target_peer: P2PNode, message: P2PMessage):
        """Send message directly to peer"""
        try:
            # Serialize message
            message_data = {
                'message_id': message.message_id,
                'message_type': message.message_type.value,
                'sender_id': message.sender_id,
                'receiver_id': message.receiver_id,
                'payload': message.payload,
                'timestamp': message.timestamp.isoformat(),
                'ttl': message.ttl
            }
            
            message_json = json.dumps(message_data).encode('utf-8')
            message_length = len(message_json)
            
            # Send to peer
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # 10 second timeout
            sock.connect((target_peer.address, target_peer.port))
            
            # Send length first, then message
            sock.send(struct.pack('!I', message_length))
            sock.send(message_json)
            
            sock.close()
            
        except Exception as e:
            logger.error(f"Failed to send message to {target_peer.node_id}: {e}")
    
    def join_network(self, bootstrap_nodes: List[Tuple[str, int]]):
        """Join the P2P network using bootstrap nodes"""
        for address, port in bootstrap_nodes:
            try:
                # Send join request
                join_message = P2PMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.JOIN_REQUEST,
                    sender_id=self.node_id,
                    receiver_id=None,
                    payload={
                        "node_id": self.node_id,
                        "address": "localhost",  # This would be the actual address
                        "port": self.port,
                        "capabilities": [cap.value for cap in self.capabilities]
                    },
                    timestamp=datetime.now()
                )
                
                # Create temporary peer for bootstrap
                bootstrap_peer = P2PNode(
                    node_id=f"bootstrap_{address}_{port}",
                    address=address,
                    port=port,
                    status=NodeStatus.ACTIVE,
                    last_seen=datetime.now(),
                    capabilities=[]
                )
                
                self.send_message_direct(bootstrap_peer, join_message)
                logger.info(f"Sent join request to {address}:{port}")
                
            except Exception as e:
                logger.error(f"Failed to join via {address}:{port}: {e}")
    
    def route_query(self, query: str, capability: AgentCapability) -> Optional[str]:
        """Route query to best available agent"""
        try:
            # First check local agents
            local_agents = self.get_local_agents_by_capability(capability)
            if local_agents:
                return self._process_query_locally(query, local_agents[0])
            
            # Find best remote agent
            best_agents = self.discovery_system.get_agents_by_capability(capability)
            if not best_agents:
                return None
            
            # Route to best agent
            best_agent = best_agents[0]
            return self._route_query_to_agent(query, best_agent)
            
        except Exception as e:
            logger.error(f"Error routing query: {e}")
            return None
    
    def get_active_peers(self) -> List[P2PNode]:
        """Get list of active peers"""
        return [peer for peer in self.peers.values() if peer.status == NodeStatus.ACTIVE]
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get current node status"""
        return {
            "node_id": self.node_id,
            "status": NodeStatus.ACTIVE.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "load": self._calculate_current_load(),
            "peer_count": len(self.get_active_peers()),
            "uptime": time.time() - getattr(self, 'start_time', time.time())
        }
    
    def get_local_agent_info(self) -> List[Dict[str, Any]]:
        """Get local agent information"""
        return [
            {
                "agent_id": agent_id,
                "capabilities": [cap.value for cap in agent_info.get("capabilities", [])],
                "load": agent_info.get("load", 0.0),
                "performance_metrics": agent_info.get("performance_metrics", {})
            }
            for agent_id, agent_info in self.local_agents.items()
        ]
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get network topology information"""
        topology = {}
        
        for peer_id, peer in self.peers.items():
            topology[peer_id] = {
                "status": peer.status.value,
                "capabilities": [cap.value for cap in peer.capabilities],
                "load": peer.load,
                "latency": peer.latency,
                "last_seen": peer.last_seen.isoformat()
            }
        
        return topology
    
    def update_network_info(self, network_data: Dict[str, Any]):
        """Update network information from gossip"""
        try:
            # Update topology
            if "network_topology" in network_data:
                self.topology_manager.update_topology(network_data["network_topology"])
            
            # Update node status
            if "node_status" in network_data:
                node_status = network_data["node_status"]
                node_id = node_status.get("node_id")
                
                if node_id and node_id != self.node_id:
                    # Update or create peer
                    if node_id not in self.peers:
                        self.peers[node_id] = P2PNode(
                            node_id=node_id,
                            address="unknown",  # Would be resolved through other means
                            port=0,
                            status=NodeStatus.ACTIVE,
                            last_seen=datetime.now(),
                            capabilities=[]
                        )
                    
                    peer = self.peers[node_id]
                    peer.status = NodeStatus(node_status.get("status", NodeStatus.ACTIVE.value))
                    peer.load = node_status.get("load", 0.0)
                    peer.last_seen = datetime.now()
                    
                    # Update capabilities
                    caps = node_status.get("capabilities", [])
                    peer.capabilities = [AgentCapability(cap) for cap in caps if cap in [c.value for c in AgentCapability]]
        
        except Exception as e:
            logger.error(f"Error updating network info: {e}")
    
    def _calculate_current_load(self) -> float:
        """Calculate current node load"""
        # Simple load calculation based on active connections and processing
        return min(1.0, len(self.peers) / 100.0)  # Simplified
    
    def _process_query_locally(self, query: str, agent_info: Dict[str, Any]) -> str:
        """Process query using local agent"""
        # Placeholder for local query processing
        return f"Local response to: {query}"
    
    def _route_query_to_agent(self, query: str, agent_info: AgentDiscoveryInfo) -> Optional[str]:
        """Route query to remote agent"""
        try:
            query_message = P2PMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.QUERY_ROUTE,
                sender_id=self.node_id,
                receiver_id=agent_info.node_id,
                payload={
                    "query": query,
                    "agent_id": agent_info.agent_id,
                    "capability": agent_info.capabilities[0].value if agent_info.capabilities else None
                },
                timestamp=datetime.now()
            )
            
            if agent_info.node_id in self.peers:
                target_peer = self.peers[agent_info.node_id]
                self.send_message_direct(target_peer, query_message)
                return f"Query routed to {agent_info.agent_id}"
            
        except Exception as e:
            logger.error(f"Error routing query to agent: {e}")
        
        return None
    
    def get_local_agents_by_capability(self, capability: AgentCapability) -> List[Dict[str, Any]]:
        """Get local agents by capability"""
        return [
            agent_info for agent_info in self.local_agents.values()
            if capability in agent_info.get("capabilities", [])
        ]
    
    def _process_query_route(self, message: P2PMessage):
        """Process incoming query route message"""
        try:
            payload = message.payload
            query = payload.get("query")
            agent_id = payload.get("agent_id")
            
            # Process query locally
            response = self._process_query_locally(query, {"agent_id": agent_id})
            
            # Send response back
            response_message = P2PMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.QUERY_RESPONSE,
                sender_id=self.node_id,
                receiver_id=message.sender_id,
                payload={"response": response, "query_id": message.message_id},
                timestamp=datetime.now()
            )
            
            if message.sender_id in self.peers:
                sender_peer = self.peers[message.sender_id]
                self.send_message_direct(sender_peer, response_message)
            
        except Exception as e:
            logger.error(f"Error processing query route: {e}")
    
    def _process_ping(self, message: P2PMessage):
        """Process ping message"""
        try:
            # Send pong response
            pong_message = P2PMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.PONG,
                sender_id=self.node_id,
                receiver_id=message.sender_id,
                payload={"original_message_id": message.message_id},
                timestamp=datetime.now()
            )
            
            if message.sender_id in self.peers:
                sender_peer = self.peers[message.sender_id]
                self.send_message_direct(sender_peer, pong_message)
            
        except Exception as e:
            logger.error(f"Error processing ping: {e}")
    
    def _process_join_request(self, message: P2PMessage):
        """Process join request from new node"""
        try:
            payload = message.payload
            new_node_id = payload.get("node_id")
            address = payload.get("address")
            port = payload.get("port")
            capabilities = payload.get("capabilities", [])
            
            # Add new peer
            new_peer = P2PNode(
                node_id=new_node_id,
                address=address,
                port=port,
                status=NodeStatus.ACTIVE,
                last_seen=datetime.now(),
                capabilities=[AgentCapability(cap) for cap in capabilities if cap in [c.value for c in AgentCapability]]
            )
            
            self.peers[new_node_id] = new_peer
            
            # Send join response with current network info
            join_response = P2PMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.JOIN_RESPONSE,
                sender_id=self.node_id,
                receiver_id=new_node_id,
                payload={
                    "welcome": True,
                    "network_topology": self.get_network_topology(),
                    "bootstrap_peers": [
                        {"node_id": peer.node_id, "address": peer.address, "port": peer.port}
                        for peer in list(self.peers.values())[:10]  # Send up to 10 peers
                    ]
                },
                timestamp=datetime.now()
            )
            
            self.send_message_direct(new_peer, join_response)
            logger.info(f"New node {new_node_id} joined the network")
            
        except Exception as e:
            logger.error(f"Error processing join request: {e}")
    
    def _process_heartbeat(self, message: P2PMessage):
        """Process heartbeat message"""
        try:
            sender_id = message.sender_id
            
            if sender_id in self.peers:
                peer = self.peers[sender_id]
                peer.last_seen = datetime.now()
                peer.status = NodeStatus.ACTIVE
                
                # Update peer metrics from heartbeat
                payload = message.payload
                peer.load = payload.get("load", peer.load)
                peer.latency = payload.get("latency", peer.latency)
            
        except Exception as e:
            logger.error(f"Error processing heartbeat: {e}")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        topology_stats = self.topology_manager.get_network_stats()
        
        active_peers = self.get_active_peers()
        total_capabilities = set()
        
        for peer in active_peers:
            total_capabilities.update(peer.capabilities)
        
        return {
            "node_id": self.node_id,
            "network_topology": topology_stats,
            "peer_count": len(active_peers),
            "total_capabilities": len(total_capabilities),
            "discovered_agents": len(self.discovery_system.discovered_agents),
            "circuit_breakers": len(self.discovery_system.circuit_breakers),
            "gossip_cache_size": len(self.gossip_protocol.message_cache)
        }

# Global instance
_p2p_network_manager = None

def get_p2p_network_manager(node_id: str = None, port: int = None, capabilities: List[AgentCapability] = None) -> P2PNetworkManager:
    """Get or create global P2P network manager instance"""
    global _p2p_network_manager
    if _p2p_network_manager is None and node_id and port:
        _p2p_network_manager = P2PNetworkManager(
            node_id=node_id,
            port=port,
            capabilities=capabilities or [AgentCapability.TEXT_GENERATION]
        )
    return _p2p_network_manager