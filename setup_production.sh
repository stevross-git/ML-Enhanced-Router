#!/bin/bash

# Production Setup Script for MLEnhancedRouter
# This script sets up the database with Docker and local virtual environment for development

set -e  # Exit on error

echo "🚀 MLEnhancedRouter Production Setup"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3.11 or later."
    exit 1
fi

echo "✅ All prerequisites met!"

# Step 1: Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  IMPORTANT: Edit .env file and set secure passwords before proceeding!"
    echo "   Press Enter after you've updated the .env file..."
    read -r
else
    echo "✅ .env file already exists"
fi

# Step 2: Set up Python virtual environment
echo ""
echo "🐍 Setting up Python virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Python dependencies installed"

# Step 3: Start Docker services (Database and Redis only)
echo ""
echo "🐳 Starting Docker services (PostgreSQL and Redis)..."

# Create a docker-compose override for database-only setup
cat > docker-compose.override.yml << 'EOF'
# Override to run only database and cache services
# The application will run locally in virtual environment

services:
  # Disable the app service - we'll run it locally
  app:
    deploy:
      replicas: 0
    command: echo "App service disabled - running locally"
    
  # pgAdmin is optional, comment out if not needed
  pgadmin:
    profiles:
      - with-pgadmin
EOF

# Start only database and redis services
docker-compose up -d db redis

echo "⏳ Waiting for database to be ready..."
sleep 10

# Check if database is ready
until docker-compose exec -T db pg_isready -U ${POSTGRES_USER:-ml_router_user} -d ${POSTGRES_DB:-ml_router_db}; do
    echo "⏳ Waiting for database..."
    sleep 2
done

echo "✅ Database services are running"

# Step 4: Initialize database
echo ""
echo "🗄️ Initializing database..."

# Set environment variables for local development to connect to Docker database
export DATABASE_URL="postgresql://${POSTGRES_USER:-ml_router_user}:${POSTGRES_PASSWORD:-ml_router_password}@localhost:5432/${POSTGRES_DB:-ml_router_db}"
export SQLALCHEMY_DATABASE_URI=$DATABASE_URL
export REDIS_URL="redis://localhost:6379/0"
export FLASK_ENV="development"

# Initialize the database
python init_db.py

echo "✅ Database initialized"

# Step 5: Create run script for local development
cat > run_local.sh << 'EOF'
#!/bin/bash

# Script to run the application locally with virtual environment

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Activate virtual environment
source venv/bin/activate

# Set database connection to Docker containers
export DATABASE_URL="postgresql://${POSTGRES_USER:-ml_router_user}:${POSTGRES_PASSWORD:-ml_router_password}@localhost:5432/${POSTGRES_DB:-ml_router_db}"
export SQLALCHEMY_DATABASE_URI=$DATABASE_URL
export REDIS_URL="redis://localhost:6379/0"
export FLASK_ENV="${FLASK_ENV:-development}"

# Run the application
echo "🚀 Starting MLEnhancedRouter..."
echo "📍 Application: http://localhost:5000"
echo "🗄️ Database: PostgreSQL (Docker) on localhost:5432"
echo "💾 Cache: Redis (Docker) on localhost:6379"
echo ""
python main.py
EOF

chmod +x run_local.sh

# Step 6: Create management scripts
cat > manage_services.sh << 'EOF'
#!/bin/bash

# Service management script

case "$1" in
    start)
        echo "Starting database services..."
        docker-compose up -d db redis
        echo "✅ Services started"
        echo "Run ./run_local.sh to start the application"
        ;;
    stop)
        echo "Stopping database services..."
        docker-compose stop db redis
        echo "✅ Services stopped"
        ;;
    restart)
        echo "Restarting database services..."
        docker-compose restart db redis
        echo "✅ Services restarted"
        ;;
    status)
        echo "Service status:"
        docker-compose ps db redis
        ;;
    logs)
        docker-compose logs -f db redis
        ;;
    pgadmin)
        echo "Starting pgAdmin..."
        docker-compose --profile with-pgadmin up -d pgadmin
        echo "✅ pgAdmin available at http://localhost:8080"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|pgadmin}"
        exit 1
        ;;
esac
EOF

chmod +x manage_services.sh

# Step 7: Display summary
echo ""
echo "✨ Setup Complete! ✨"
echo "===================="
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🎯 Next Steps:"
echo "1. Database is running in Docker on localhost:5432"
echo "2. Redis is running in Docker on localhost:6379"
echo "3. To start the application locally:"
echo "   ./run_local.sh"
echo ""
echo "📝 Management Commands:"
echo "   ./manage_services.sh start    - Start database services"
echo "   ./manage_services.sh stop     - Stop database services"
echo "   ./manage_services.sh status   - Check service status"
echo "   ./manage_services.sh logs     - View service logs"
echo "   ./manage_services.sh pgadmin  - Start pgAdmin (optional)"
echo ""
echo "🔧 Configuration:"
echo "   - Edit .env file for API keys and settings"
echo "   - Database: PostgreSQL (Docker)"
echo "   - Cache: Redis (Docker)"
echo "   - Application: Python virtual environment (local)"
echo ""
echo "🌐 Access Points:"
echo "   - Application: http://localhost:5000"
echo "   - pgAdmin: http://localhost:8080 (if enabled)"
echo ""
echo "⚠️  For production deployment:"
echo "   1. Set strong passwords in .env file"
echo "   2. Configure SSL/TLS certificates"
echo "   3. Set FLASK_ENV=production"
echo "   4. Use 'docker-compose up -d' to run full stack in Docker"