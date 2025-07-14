# Network Deployment Guide

## Overview
This guide explains how to deploy the ML Router application to your network infrastructure.

## Deployment Options

### 1. Docker Deployment (Recommended)

#### Prerequisites
- Docker and Docker Compose installed
- Access to your network environment
- SSL certificates (optional, for HTTPS)

#### Quick Start
```bash
# Clone or copy the application to your server
git clone <your-repo> ml-router
cd ml-router

# Build and run with Docker Compose
docker-compose up -d
```

#### Configuration
Edit `docker-compose.yml` to customize:
- Database connection (PostgreSQL recommended for production)
- Redis cache (optional but recommended)
- API keys and secrets
- Network ports and domains

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

# Run with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 main:app
```

### 3. Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster
- kubectl configured
- Container registry access

#### Deploy to Kubernetes
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/
```

## Network Configuration

### Port Configuration
- **Web Interface**: Port 5000 (HTTP)
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