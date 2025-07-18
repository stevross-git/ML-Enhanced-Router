"""Network configuration classes."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import os

@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_encryption: bool = True
    enable_authentication: bool = True
    jwt_secret: str = field(default_factory=lambda: os.getenv("ROUTER_JWT_SECRET", ""))
    enable_tls: bool = False
    cert_file: Optional[str] = None
    key_file: Optional[str] = None

@dataclass 
class P2PConfig:
    """P2P configuration."""
    listen_address: str = "0.0.0.0"
    listen_port: int = 30301
    bootstrap_nodes: List[str] = field(default_factory=list)
    enable_mdns: bool = True
    enable_upnp: bool = True
    enable_nat_traversal: bool = True
    dns_seed_domain: str = "peoplesainetwork.com"
    connection_timeout: int = 30
    max_connections: int = 50
    enable_dht: bool = True
    dht_port: int = 30302

@dataclass
class MeshConfig:
    """Mesh topology configuration."""
    topology_type: str = "dynamic_partial"
    enable_super_peers: bool = True
    super_peer_capacity_threshold: float = 100.0
    max_peers: int = 20
    routing_update_interval: int = 10
    link_quality_threshold: float = 0.5
    enable_multi_hop: bool = True
    max_hop_count: int = 10

@dataclass
class DNSConfig:
    """DNS configuration."""
    root_domain: str = ".web4ai"
    enable_dnssec: bool = True
    default_ttl: int = 3600
    cache_size: int = 10000
    enable_recursive: bool = True
    upstream_dns: List[str] = field(default_factory=lambda: ["8.8.8.8", "1.1.1.1"])

@dataclass
class RoutingConfig:
    """Routing configuration."""
    enable_multipath: bool = True
    enable_ml_predictor: bool = True
    max_paths_per_destination: int = 3
    failover_threshold_ms: int = 500
    path_quality_update_interval: int = 30
    metric_update_interval: int = 30
    route_optimization_interval: int = 60
    ml_update_interval: int = 300
    enable_congestion_control: bool = True
    enable_qos: bool = True
    priority_levels: int = 4

@dataclass
class NetworkConfig:
    """Main network configuration."""
    node_name: str = "csp-node"
    node_type: str = "standard"
    data_dir: Path = field(default_factory=lambda: Path("./network_data"))
    
    # Component configurations
    security: SecurityConfig = field(default_factory=SecurityConfig)
    p2p: P2PConfig = field(default_factory=P2PConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    dns: DNSConfig = field(default_factory=DNSConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    
    # Legacy compatibility
    listen_port: int = 30301
    listen_address: str = "0.0.0.0"
    
    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sync legacy fields with P2P config
        self.p2p.listen_port = self.listen_port
        self.p2p.listen_address = self.listen_address
