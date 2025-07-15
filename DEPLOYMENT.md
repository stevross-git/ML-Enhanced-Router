# Network Deployment Guide

## Overview
This guide explains how to deploy the ML Router application to your AI network infrastructure with full network integration capabilities including service discovery, load balancing, distributed caching, and enterprise features including Cross-Persona Memory Inference, Cognitive Loop Debugging, and Temporal Memory Weighting.

## Deployment Options

### 1. Docker Deployment (Recommended)

#### Prerequisites
- Docker and Docker Compose installed
- Access to your network environment
- SSL certificates (optional, for HTTPS)
- Network access to service registry (if using external Consul)

#### Quick Start with Automated Setup
```bash
# Clone or copy the application to your server
git clone <your-repo> ml-router
cd ml-router

# Run automated setup script
chmod +x setup.sh
./setup.sh

# Choose option 1 for Docker Compose deployment
# The script will:
# 1. Create necessary directories
# 2. Generate secrets
# 3. Configure environment variables
# 4. Build and start Docker containers
```

#### Manual Docker Setup
```bash
# Create .env file from template
cp .env.example .env
# Edit .env with your configuration

# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

#### Docker Compose Configuration
Edit `docker-compose.yml` to customize:
- Database connection (PostgreSQL recommended for production)
- Redis cache (optional but recommended)
- API keys and secrets
- Network ports and domains
- Service discovery endpoints
- Network security settings

## Network Integration Startup Methods

### Using Network Integration Scripts

#### Option 1: Network-Enabled ML Router
```bash
# Start with network integration
python ml_router_network.py

# Or use the startup script
./startup_script.sh start

# Check network status
./startup_script.sh status

# View network logs
./startup_script.sh logs
```

#### Option 2: Network Bridge Mode
```bash
# Start network bridge directly
python -c "
import asyncio
from network_bridge import NetworkBridge

async def start_bridge():
    bridge = NetworkBridge()
    await bridge.start_bridge()
    print('Network bridge started successfully')

asyncio.run(start_bridge())
"
```

#### Network Configuration Files
- **`integration_config.py`**: Network service configuration
- **`network_bridge.py`**: Network bridge implementation
- **`ml_router_network.py`**: Network-enabled ML router
- **`startup_script.sh`**: Automated startup script

### Network API Endpoints
Once deployed, the following network integration endpoints are available:

#### Status and Health
- `GET /api/network/status` - Network integration status
- `GET /api/network/health` - Network health check
- `GET /network-dashboard` - Network integration dashboard

#### Query Processing
- `POST /api/network/query` - Submit query through network
- `GET /api/network/peers` - Get connected peers
- `GET /api/network/metrics` - Network performance metrics

#### Management
- `POST /api/network/shutdown` - Gracefully shutdown network components

### 2. Direct Server Deployment

#### Prerequisites
- Python 3.11+
- PostgreSQL database
- Redis (optional)
- Nginx (recommended for production)

#### Installation Steps
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:password@localhost/mlrouter"
export SESSION_SECRET="your-secret-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
# ... other API keys

# Initialize database
python -c "from app import db; db.create_all()"

# Run with Gunicorn (standard mode)
gunicorn --bind 0.0.0.0:5000 --workers 4 main:app

# Or run with network integration
python ml_router_network.py
```

#### Network-Enabled Direct Deployment
```bash
# Set additional network environment variables
export NETWORK_ID=ai_network
export SERVICE_DISCOVERY_ENABLED=true
export LOAD_BALANCER_ENABLED=true
export MONITORING_ENABLED=true

# Start with network integration
python ml_router_network.py

# Or use the startup script
./startup_script.sh start --environment=production
```

### 3. Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster
- kubectl configured
- Container registry access

#### Deploy to Kubernetes
```bash
# Create namespace
kubectl create namespace ml-router

# Create secrets (edit with your values)
kubectl create secret generic ml-router-secrets \
  --from-literal=DATABASE_URL="postgresql://user:pass@db:5432/mlrouter" \
  --from-literal=SESSION_SECRET="your-secret-key" \
  --from-literal=OPENAI_API_KEY="your-openai-key" \
  --from-literal=ANTHROPIC_API_KEY="your-anthropic-key" \
  -n ml-router

# Apply Kubernetes manifests
kubectl apply -f k8s/ -n ml-router

# Check deployment status
kubectl get pods -n ml-router
kubectl get svc -n ml-router

# Access logs
kubectl logs -f deployment/ml-router -n ml-router
```

#### Network Integration in Kubernetes
The Kubernetes deployment includes:
- Service mesh integration
- Network policies for secure communication
- Health checks and readiness probes
- Horizontal pod autoscaling
- Network service discovery
- Load balancing across pods

## Enterprise Features Configuration

### Cross-Persona Memory Inference
- **Database Setup**: Creates dedicated tables for persona linkages and memory bridges
- **Configuration**: Automatic initialization with Personal AI router
- **Memory Management**: Handles persona compatibility analysis and memory bridging
- **API Integration**: Provides REST endpoints for linkage analysis and insights

### Cognitive Loop Debugging
- **Decision Tracking**: Logs all AI routing decisions with confidence scores
- **Session Management**: Supports session-based debugging with unique session IDs
- **Performance Monitoring**: Tracks decision patterns and optimization opportunities
- **Real-time Analysis**: Provides immediate feedback on AI decision-making processes

### Temporal Memory Weighting
- **Memory Prioritization**: Manages memory importance based on temporal context
- **Intelligent Decay**: Automatically reduces memory relevance over time
- **Access Pattern Learning**: Optimizes memory storage based on usage patterns
- **Dynamic Relevance**: Calculates time-based relevance scores for retrieval

### Enterprise Configuration Variables
```bash
# Cross-Persona Memory Inference
CROSS_PERSONA_ENABLED=true
CROSS_PERSONA_DB_PATH=cross_persona_memory.db
PERSONA_ANALYSIS_ENABLED=true
MEMORY_BRIDGE_SUGGESTIONS=true

# Cognitive Loop Debugging
COGNITIVE_DEBUGGING_ENABLED=true
COGNITIVE_DEBUG_DB_PATH=cognitive_decisions.db
DECISION_TRACKING_ENABLED=true
SESSION_DEBUGGING_ENABLED=true

# Temporal Memory Weighting
TEMPORAL_MEMORY_ENABLED=true
TEMPORAL_MEMORY_DB_PATH=temporal_memory.db
MEMORY_DECAY_ENABLED=true
DYNAMIC_RELEVANCE_ENABLED=true
```

## Network Integration Features

### Service Discovery
- **Automatic Registration**: Services automatically register with the network
- **Health Monitoring**: Continuous health checks and service availability tracking
- **Service Registry**: Consul or similar service discovery backend support

### Load Balancing
- **Circuit Breaker**: Automatic failover when services become unhealthy
- **Round Robin**: Distributes requests evenly across available services
- **Weighted Routing**: Priority-based routing for optimal performance

### Security
- **Mutual TLS**: Secure service-to-service communication
- **JWT Authentication**: Token-based authentication across network services
- **Network Encryption**: End-to-end encryption for all network traffic

### Monitoring
- **Distributed Tracing**: Track requests across multiple services
- **Network Metrics**: Real-time performance and health metrics
- **Telemetry**: Comprehensive logging and monitoring integration

## Network Configuration

### Port Configuration
- **Web Interface**: Port 5000 (HTTP)
- **Network Bridge**: Port 8080 (Network Integration)
- **Service Registry**: Port 8500 (Consul)
- **Database**: Port 5432 (PostgreSQL)
- **Cache**: Port 6379 (Redis)

### Network Environment Variables
```bash
# Network Identity
NETWORK_ID=ai_network
CLUSTER_NAME=ml_cluster
ENVIRONMENT=production

# Service Discovery
SERVICE_DISCOVERY_ENABLED=true
SERVICE_REGISTRY_HOST=localhost
SERVICE_REGISTRY_PORT=8500
AUTO_REGISTER=true

# Load Balancing
LOAD_BALANCER_ENABLED=true
LOAD_BALANCER_ALGORITHM=round_robin
HEALTH_CHECK_INTERVAL=10
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_THRESHOLD=5

# Security
MUTUAL_TLS_ENABLED=false
JWT_ENABLED=true
NETWORK_ENCRYPTION=true
API_KEY_ENABLED=true

# Monitoring
MONITORING_ENABLED=true
DISTRIBUTED_TRACING=true
LOG_LEVEL=INFO
TELEMETRY_ENDPOINT=http://monitoring:8080/metrics

# Caching
DISTRIBUTED_CACHE_ENABLED=true
CACHE_TTL=3600
CACHE_REPLICATION_FACTOR=2

# Resource Management
MAX_CONCURRENT_REQUESTS=1000
REQUEST_TIMEOUT=120
MEMORY_LIMIT=2Gi
CPU_LIMIT=2000m
```
- **API Endpoints**: Port 5000 (same as web)
- **Health Check**: GET /api/health
- **Metrics**: Port 9090 (optional)

### Load Balancer Setup
Configure your load balancer to:
- Forward traffic to port 5000
- Enable health checks on `/api/health`
- Set up SSL termination (recommended)

### Firewall Rules
Allow inbound traffic on:
- Port 80 (HTTP) - if using reverse proxy
- Port 443 (HTTPS) - if using SSL
- Port 5000 (direct access)

## Environment Variables

### Required Variables
```bash
# Database
DATABASE_URL="postgresql://user:password@host:5432/database"

# Security
SESSION_SECRET="your-long-random-secret"
ROUTER_JWT_SECRET="jwt-signing-secret"

# AI Provider Keys (at least one required)
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GOOGLE_API_KEY="AIza..."
XAI_API_KEY="xai-..."
PERPLEXITY_API_KEY="pplx-..."
```

### Optional Variables
```bash
# Cache Configuration
REDIS_URL="redis://localhost:6379"
ROUTER_CACHE_TTL="3600"
ROUTER_CACHE_SIZE="10000"

# Performance
ROUTER_MAX_CONCURRENT="50"
ROUTER_RATE_LIMIT="1000"
ROUTER_GLOBAL_RATE_LIMIT="5000"

# Features
ROUTER_USE_ML="true"
ROUTER_AUTH_ENABLED="true"
```

## Security Considerations

### Authentication
- Enable API authentication: `ROUTER_AUTH_ENABLED=true`
- Use strong JWT secrets
- Implement rate limiting
- Use HTTPS in production

### Network Security
- Place behind firewall
- Use VPN for internal access
- Implement IP whitelisting if needed
- Regular security updates

### API Key Management
- Store API keys securely (environment variables or secrets manager)
- Rotate keys regularly
- Monitor usage and costs
- Use least privilege principle

## Monitoring and Maintenance

### Health Monitoring
- Endpoint: `GET /api/health`
- Returns system status and component health
- Use for load balancer health checks

### Metrics Dashboard
- Access: `http://your-server:5000/dashboard`
- Shows real-time performance metrics
- Monitor query routing, cache performance, API usage

### Log Management
- Application logs via Python logging
- Gunicorn access logs
- Error tracking and alerting
- Log rotation and retention

### Backup Strategy
- Regular database backups
- Configuration backup
- Document recovery procedures
- Test restore procedures

## Scaling Considerations

### Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use shared PostgreSQL database
- Use Redis for session storage
- Implement sticky sessions if needed

### Vertical Scaling
- Increase CPU and memory
- Optimize database performance
- Tune cache settings
- Monitor resource usage

### Performance Optimization
- Enable Redis caching
- Optimize database queries
- Use connection pooling
- Enable gzip compression

## Troubleshooting

### Common Issues
1. **Database Connection**: Check DATABASE_URL and network connectivity
2. **API Key Errors**: Verify API keys are set and valid
3. **Cache Issues**: Check Redis connection and configuration
4. **High Response Times**: Monitor database and external API performance

### Debug Mode
For development only:
```bash
export FLASK_ENV=development
export FLASK_DEBUG=true
python main.py
```

### Logs Location
- Application logs: Check console output or logging configuration
- Access logs: Gunicorn access logs
- Error logs: stderr output

## Support

### Documentation
- API Documentation: `http://your-server:5000/docs`
- Configuration Reference: See `config.py`
- Database Schema: See `models.py`

### Monitoring Endpoints
- Health Check: `/api/health`
- Statistics: `/api/stats`
- Cache Status: `/api/cache/stats`
- External LLM Metrics: `/api/external-llm/metrics`

For additional support, check the application logs and monitoring dashboard.