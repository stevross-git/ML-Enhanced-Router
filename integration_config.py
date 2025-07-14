"""
AI Network Integration Configuration
Manages network-wide AI service integration and routing
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class NetworkServiceType(Enum):
    """Types of AI services in the network"""
    ML_ROUTER = "ml_router"
    INFERENCE_ENGINE = "inference_engine"
    MODEL_REGISTRY = "model_registry"
    VECTOR_DATABASE = "vector_database"
    CACHE_SERVICE = "cache_service"
    MONITORING = "monitoring"
    GATEWAY = "gateway"
    LOAD_BALANCER = "load_balancer"

@dataclass
class NetworkService:
    """Network service configuration"""
    id: str
    name: str
    type: NetworkServiceType
    host: str
    port: int
    protocol: str = "http"
    health_endpoint: str = "/health"
    api_version: str = "v1"
    authentication: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    priority: int = 1
    timeout: int = 30
    retry_count: int = 3
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def base_url(self) -> str:
        """Get the base URL for the service"""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def health_url(self) -> str:
        """Get the health check URL"""
        return f"{self.base_url}{self.health_endpoint}"

@dataclass
class NetworkIntegrationConfig:
    """Complete network integration configuration"""
    
    # Network Identity
    network_id: str = "ai_network"
    cluster_name: str = "ml_cluster"
    environment: str = "production"
    
    # Service Discovery
    service_discovery_enabled: bool = True
    service_registry_host: str = "localhost"
    service_registry_port: int = 8500
    service_registry_protocol: str = "http"
    auto_register: bool = True
    registration_interval: int = 30
    
    # Load Balancing
    load_balancer_enabled: bool = True
    load_balancer_algorithm: str = "round_robin"  # round_robin, least_connections, weighted
    health_check_interval: int = 10
    health_check_timeout: int = 5
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    
    # Security
    mutual_tls_enabled: bool = False
    api_key_enabled: bool = True
    jwt_enabled: bool = True
    network_encryption: bool = True
    allowed_networks: List[str] = field(default_factory=lambda: ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"])
    
    # Monitoring and Logging
    monitoring_enabled: bool = True
    metrics_endpoint: str = "/metrics"
    log_level: str = "INFO"
    distributed_tracing: bool = True
    telemetry_endpoint: str = ""
    
    # Caching
    distributed_cache_enabled: bool = True
    cache_ttl: int = 3600
    cache_replication_factor: int = 2
    
    # Resource Management
    max_concurrent_requests: int = 1000
    request_timeout: int = 120
    memory_limit: str = "2Gi"
    cpu_limit: str = "2000m"
    
    # Network Services
    services: Dict[str, NetworkService] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> 'NetworkIntegrationConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # Network Identity
        config.network_id = os.getenv("NETWORK_ID", config.network_id)
        config.cluster_name = os.getenv("CLUSTER_NAME", config.cluster_name)
        config.environment = os.getenv("ENVIRONMENT", config.environment)
        
        # Service Discovery
        config.service_discovery_enabled = os.getenv("SERVICE_DISCOVERY_ENABLED", "true").lower() == "true"
        config.service_registry_host = os.getenv("SERVICE_REGISTRY_HOST", config.service_registry_host)
        config.service_registry_port = int(os.getenv("SERVICE_REGISTRY_PORT", str(config.service_registry_port)))
        config.auto_register = os.getenv("AUTO_REGISTER", "true").lower() == "true"
        
        # Load Balancing
        config.load_balancer_enabled = os.getenv("LOAD_BALANCER_ENABLED", "true").lower() == "true"
        config.load_balancer_algorithm = os.getenv("LOAD_BALANCER_ALGORITHM", config.load_balancer_algorithm)
        config.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", str(config.health_check_interval)))
        config.circuit_breaker_enabled = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
        
        # Security
        config.mutual_tls_enabled = os.getenv("MUTUAL_TLS_ENABLED", "false").lower() == "true"
        config.api_key_enabled = os.getenv("API_KEY_ENABLED", "true").lower() == "true"
        config.jwt_enabled = os.getenv("JWT_ENABLED", "true").lower() == "true"
        config.network_encryption = os.getenv("NETWORK_ENCRYPTION", "true").lower() == "true"
        
        # Monitoring
        config.monitoring_enabled = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
        config.log_level = os.getenv("LOG_LEVEL", config.log_level)
        config.distributed_tracing = os.getenv("DISTRIBUTED_TRACING", "true").lower() == "true"
        config.telemetry_endpoint = os.getenv("TELEMETRY_ENDPOINT", config.telemetry_endpoint)
        
        # Caching
        config.distributed_cache_enabled = os.getenv("DISTRIBUTED_CACHE_ENABLED", "true").lower() == "true"
        config.cache_ttl = int(os.getenv("CACHE_TTL", str(config.cache_ttl)))
        
        # Resource Management
        config.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", str(config.max_concurrent_requests)))
        config.request_timeout = int(os.getenv("REQUEST_TIMEOUT", str(config.request_timeout)))
        config.memory_limit = os.getenv("MEMORY_LIMIT", config.memory_limit)
        config.cpu_limit = os.getenv("CPU_LIMIT", config.cpu_limit)
        
        # Load services from environment
        config._load_services_from_env()
        
        return config
    
    def _load_services_from_env(self):
        """Load service configurations from environment variables"""
        services_config = os.getenv("NETWORK_SERVICES")
        if services_config:
            try:
                services_data = json.loads(services_config)
                for service_data in services_data:
                    service = NetworkService(
                        id=service_data.get("id"),
                        name=service_data.get("name"),
                        type=NetworkServiceType(service_data.get("type")),
                        host=service_data.get("host"),
                        port=service_data.get("port"),
                        protocol=service_data.get("protocol", "http"),
                        health_endpoint=service_data.get("health_endpoint", "/health"),
                        api_version=service_data.get("api_version", "v1"),
                        authentication=service_data.get("authentication", {}),
                        capabilities=service_data.get("capabilities", []),
                        priority=service_data.get("priority", 1),
                        timeout=service_data.get("timeout", 30),
                        retry_count=service_data.get("retry_count", 3),
                        is_active=service_data.get("is_active", True),
                        metadata=service_data.get("metadata", {})
                    )
                    self.services[service.id] = service
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse NETWORK_SERVICES: {e}")
    
    def add_service(self, service: NetworkService):
        """Add a network service"""
        self.services[service.id] = service
        logger.info(f"Added service: {service.name} ({service.id})")
    
    def remove_service(self, service_id: str):
        """Remove a network service"""
        if service_id in self.services:
            service = self.services.pop(service_id)
            logger.info(f"Removed service: {service.name} ({service_id})")
    
    def get_service(self, service_id: str) -> Optional[NetworkService]:
        """Get a service by ID"""
        return self.services.get(service_id)
    
    def get_services_by_type(self, service_type: NetworkServiceType) -> List[NetworkService]:
        """Get services by type"""
        return [service for service in self.services.values() if service.type == service_type]
    
    def get_active_services(self) -> List[NetworkService]:
        """Get all active services"""
        return [service for service in self.services.values() if service.is_active]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "network_id": self.network_id,
            "cluster_name": self.cluster_name,
            "environment": self.environment,
            "service_discovery_enabled": self.service_discovery_enabled,
            "service_registry_host": self.service_registry_host,
            "service_registry_port": self.service_registry_port,
            "auto_register": self.auto_register,
            "load_balancer_enabled": self.load_balancer_enabled,
            "load_balancer_algorithm": self.load_balancer_algorithm,
            "health_check_interval": self.health_check_interval,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "mutual_tls_enabled": self.mutual_tls_enabled,
            "api_key_enabled": self.api_key_enabled,
            "jwt_enabled": self.jwt_enabled,
            "network_encryption": self.network_encryption,
            "allowed_networks": self.allowed_networks,
            "monitoring_enabled": self.monitoring_enabled,
            "log_level": self.log_level,
            "distributed_tracing": self.distributed_tracing,
            "distributed_cache_enabled": self.distributed_cache_enabled,
            "cache_ttl": self.cache_ttl,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout": self.request_timeout,
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit,
            "services": {
                service_id: {
                    "id": service.id,
                    "name": service.name,
                    "type": service.type.value,
                    "host": service.host,
                    "port": service.port,
                    "protocol": service.protocol,
                    "health_endpoint": service.health_endpoint,
                    "api_version": service.api_version,
                    "authentication": service.authentication,
                    "capabilities": service.capabilities,
                    "priority": service.priority,
                    "timeout": service.timeout,
                    "retry_count": service.retry_count,
                    "is_active": service.is_active,
                    "metadata": service.metadata
                }
                for service_id, service in self.services.items()
            }
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'NetworkIntegrationConfig':
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # Load basic configuration
        for key, value in data.items():
            if key != "services" and hasattr(config, key):
                setattr(config, key, value)
        
        # Load services
        for service_id, service_data in data.get("services", {}).items():
            service = NetworkService(
                id=service_data["id"],
                name=service_data["name"],
                type=NetworkServiceType(service_data["type"]),
                host=service_data["host"],
                port=service_data["port"],
                protocol=service_data.get("protocol", "http"),
                health_endpoint=service_data.get("health_endpoint", "/health"),
                api_version=service_data.get("api_version", "v1"),
                authentication=service_data.get("authentication", {}),
                capabilities=service_data.get("capabilities", []),
                priority=service_data.get("priority", 1),
                timeout=service_data.get("timeout", 30),
                retry_count=service_data.get("retry_count", 3),
                is_active=service_data.get("is_active", True),
                metadata=service_data.get("metadata", {})
            )
            config.services[service_id] = service
        
        logger.info(f"Configuration loaded from {filepath}")
        return config


# Default network integration configuration
def get_default_config() -> NetworkIntegrationConfig:
    """Get default network integration configuration"""
    config = NetworkIntegrationConfig()
    
    # Add default services
    config.add_service(NetworkService(
        id="ml_router_main",
        name="ML Router Main",
        type=NetworkServiceType.ML_ROUTER,
        host="localhost",
        port=5000,
        capabilities=["query_routing", "model_management", "collaborative_ai"],
        priority=1
    ))
    
    config.add_service(NetworkService(
        id="redis_cache",
        name="Redis Cache",
        type=NetworkServiceType.CACHE_SERVICE,
        host="localhost",
        port=6379,
        protocol="redis",
        health_endpoint="/info",
        capabilities=["caching", "session_storage"],
        priority=2
    ))
    
    config.add_service(NetworkService(
        id="postgresql_db",
        name="PostgreSQL Database",
        type=NetworkServiceType.VECTOR_DATABASE,
        host="localhost",
        port=5432,
        protocol="postgresql",
        health_endpoint="/health",
        capabilities=["data_storage", "vector_search"],
        priority=3
    ))
    
    return config


# Global configuration instance
_config_instance = None

def get_network_config() -> NetworkIntegrationConfig:
    """Get global network configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = NetworkIntegrationConfig.from_env()
    return _config_instance

def set_network_config(config: NetworkIntegrationConfig):
    """Set global network configuration instance"""
    global _config_instance
    _config_instance = config