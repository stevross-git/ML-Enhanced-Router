#!/bin/bash
# setup_docker.sh - Docker setup script for ML Router

set -e  # Exit on any error

echo "🚀 Setting up ML Router with Docker..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

print_header "Step 1: Cleaning up previous containers"
print_status "Stopping and removing existing containers..."
docker-compose down -v 2>/dev/null || docker compose down -v 2>/dev/null || true

print_header "Step 2: Creating environment file"
if [ ! -f .env ]; then
    print_status "Creating .env file from example..."
    cp .env.example .env
    print_warning "Please edit .env file and set your API keys!"
else
    print_status ".env file already exists"
fi

print_header "Step 3: Creating necessary directories"
mkdir -p instance logs instance/uploads
print_status "Created directories: instance, logs, instance/uploads"

print_header "Step 4: Building and starting services"
print_status "Building ML Router application..."
if command -v docker-compose &> /dev/null; then
    docker-compose build --no-cache
    print_status "Starting services..."
    docker-compose up -d
else
    docker compose build --no-cache
    print_status "Starting services..."
    docker compose up -d
fi

print_header "Step 5: Waiting for services to be ready"
print_status "Waiting for database to be ready..."
sleep 10

# Wait for database to be healthy
for i in {1..30}; do
    if docker-compose ps | grep -q "healthy" || docker compose ps | grep -q "healthy"; then
        print_status "Database is ready!"
        break
    fi
    print_status "Waiting for database... ($i/30)"
    sleep 2
done

print_header "Step 6: Running database migrations"
print_status "Initializing database schema..."
if command -v docker-compose &> /dev/null; then
    docker-compose exec app python -c "
from app import create_app
from app.extensions import db
app = create_app('production')
with app.app_context():
    db.create_all()
    print('Database tables created successfully!')
" || print_warning "Database initialization may have failed - check logs"
else
    docker compose exec app python -c "
from app import create_app
from app.extensions import db
app = create_app('production')
with app.app_context():
    db.create_all()
    print('Database tables created successfully!')
" || print_warning "Database initialization may have failed - check logs"
fi

print_header "Setup Complete!"
print_status "Services started successfully!"
echo ""
echo -e "${GREEN}🎉 ML Router is now running!${NC}"
echo ""
echo "📋 Service URLs:"
echo "   • ML Router App:  http://localhost:5000"
echo "   • pgAdmin:        http://localhost:8080"
echo "   • PostgreSQL:     localhost:5432"
echo "   • Redis:          localhost:6379"
echo ""
echo "🔐 Database Credentials:"
echo "   • Username: ml_router_user"
echo "   • Password: ml_router_password"
echo "   • Database: ml_router_db"
echo ""
echo "🔧 pgAdmin Credentials:"
echo "   • Email:    admin@peoplesainetwork.com"
echo "   • Password: admin_password"
echo ""
print_warning "Remember to:"
print_warning "1. Edit .env file and set your API keys"
print_warning "2. Change default passwords in production"
print_warning "3. Set proper SECRET_KEY values"
echo ""
print_status "To view logs: docker-compose logs -f"
print_status "To stop services: docker-compose down"
print_status "To restart: docker-compose restart"