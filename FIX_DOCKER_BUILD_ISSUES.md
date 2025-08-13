# 🔧 Fix Docker Build Hanging on Package Downloads

The Docker build is hanging while downloading Python packages. Here are multiple solutions to fix this:

---

## 🛑 Immediate Fix - Stop and Restart with Better Configuration

### Step 1: Stop Current Build
```bash
# Stop hanging containers
docker-compose down

# Clean up any stuck processes
docker system prune -f

# Remove stuck images (if any)
docker images | grep none
docker rmi $(docker images -f "dangling=true" -q) 2>/dev/null || true
```

### Step 2: Optimize Dockerfile for Faster Build

Create an optimized Dockerfile:

```bash
cd /data/apps/MLEnhancedRouter
cp Dockerfile Dockerfile.backup

# Create optimized Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages with optimized settings
RUN pip install --no-cache-dir \
    --timeout=300 \
    --retries=5 \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/api/health || exit 1

# Run application
CMD ["python", "main.py"]
EOF
```

### Step 3: Create Optimized requirements.txt

```bash
# Backup original requirements
cp requirements.txt requirements.txt.backup

# Create minimal requirements for faster install
cat > requirements.txt << 'EOF'
# Core Flask
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
Flask-Migrate==4.0.5
Flask-CORS==4.0.0
Flask-JWT-Extended==4.5.3

# Database
psycopg2-binary==2.9.7

# AI Libraries (specific versions for faster install)
openai==1.35.0
anthropic==0.30.0
google-generativeai==0.7.2

# Async support
aiohttp==3.8.6
asyncio==3.4.3

# Utilities
redis==5.0.0
celery==5.3.1
python-dotenv==1.0.0
pydantic==2.3.0
requests==2.31.0

# Security
bcrypt==4.0.1
cryptography==41.0.4

# Monitoring
prometheus-client==0.17.1

# Development
gunicorn==21.2.0
EOF
```

### Step 4: Update Docker Compose with Build Options

```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - /data/postgres:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - /data/redis:/data
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  app:
    build: 
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - .env:/app/.env
      - /data/logs:/app/logs
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  pgadmin:
    image: dpage/pgadmin4
    environment:
      PGADMIN_DEFAULT_EMAIL: ${PGADMIN_DEFAULT_EMAIL}
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_DEFAULT_PASSWORD}
    volumes:
      - /data/pgadmin:/var/lib/pgadmin
    ports:
      - "8080:80"
    depends_on:
      - db
EOF
```

---

## 🚀 Alternative Quick Fix - Use Pre-built Image

If build still hangs, use a pre-built Python image with packages:

```bash
# Create simple Dockerfile using pre-built image
cat > Dockerfile.simple << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install only essential packages
RUN pip install --no-cache-dir \
    flask==2.3.3 \
    flask-sqlalchemy==3.0.5 \
    psycopg2-binary==2.9.7 \
    redis==5.0.0 \
    python-dotenv==1.0.0 \
    requests==2.31.0 \
    gunicorn==21.2.0

# Copy app
COPY . .

# Run
CMD ["python", "main.py"]
EOF

# Use the simple dockerfile temporarily
cp Dockerfile.simple Dockerfile
```

---

## 🔧 Network/Timeout Solutions

### Option 1: Increase Docker Build Timeout
```bash
# Build with increased timeout
DOCKER_BUILDKIT=0 docker-compose build --no-cache \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --progress=plain

# If that fails, build manually
docker build --no-cache --network=host -t mlrouter-app .
```

### Option 2: Use Different Package Index
```bash
# Add to Dockerfile before pip install
RUN pip config set global.index-url https://pypi.org/simple/
RUN pip config set global.trusted-host pypi.org
```

### Option 3: Build in Stages
```bash
# Build without AI packages first
cat > requirements-base.txt << 'EOF'
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
psycopg2-binary==2.9.7
redis==5.0.0
python-dotenv==1.0.0
requests==2.31.0
gunicorn==21.2.0
EOF

# Install base packages first
pip install -r requirements-base.txt

# Then add AI packages later
```

---

## 🛠️ Emergency Workaround - Run Without Docker

If Docker keeps failing, run directly on the system:

```bash
# Install Python and packages directly
sudo apt install python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv /data/mlrouter-env
source /data/mlrouter-env/bin/activate

# Install packages manually
pip install flask flask-sqlalchemy psycopg2-binary redis python-dotenv

# Run application directly
cd /data/apps/MLEnhancedRouter
python main.py
```

---

## 📊 Monitor Build Progress

```bash
# Monitor Docker build logs in real-time
docker-compose up --build | tee /data/logs/docker-build.log

# Monitor network activity
sudo netstat -tuln

# Monitor disk I/O
iostat -x 1

# Check available memory
free -h
```

---

## 🔍 Debug Current State

```bash
# Check what's currently running
docker ps -a

# Check Docker logs
docker-compose logs app

# Check system resources
htop

# Check disk space
df -h /data

# Check network connectivity
ping -c 4 pypi.org
curl -I https://pypi.org/simple/
```

---

## ✅ Recommended Quick Solution

**Try this first - it's the fastest fix:**

```bash
# 1. Stop everything
docker-compose down
docker system prune -f

# 2. Use minimal requirements
cat > requirements.txt << 'EOF'
Flask==2.3.3
psycopg2-binary==2.9.7
redis==5.0.0
python-dotenv==1.0.0
gunicorn==21.2.0
EOF

# 3. Build with timeout and no cache
DOCKER_BUILDKIT=0 docker-compose build --no-cache

# 4. Start services
docker-compose up -d

# 5. Add AI packages later if needed
# docker exec -it mlrouter_app_1 pip install openai anthropic
```

This should get your application running quickly. You can add the AI libraries later once the basic application is working.

Try the quick solution first and let me know if it works!