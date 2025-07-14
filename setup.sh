#!/bin/bash

# ML Router Network Deployment Setup Script
# This script helps set up the ML Router for network deployment

set -e

echo "🚀 ML Router Network Deployment Setup"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to generate random secret
generate_secret() {
    openssl rand -hex 32
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    
    # Generate random secrets
    SESSION_SECRET=$(generate_secret)
    JWT_SECRET=$(generate_secret)
    
    # Replace secrets in .env file
    if command_exists sed; then
        sed -i "s/your-very-long-random-secret-key-here/$SESSION_SECRET/" .env
        sed -i "s/jwt-signing-secret-here/$JWT_SECRET/" .env
        echo "✅ Generated random secrets in .env file"
    else
        echo "⚠️  Please manually update the secrets in .env file"
    fi
    
    echo "📄 Please edit .env file and add your API keys:"
    echo "   - OPENAI_API_KEY"
    echo "   - ANTHROPIC_API_KEY"
    echo "   - GOOGLE_API_KEY"
    echo "   - Other AI provider keys as needed"
    echo ""
    read -p "Press Enter when you've updated the API keys in .env file..."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p rag_data models instance ssl

# Set proper permissions
chmod 755 rag_data models instance
echo "✅ Directories created and permissions set"

# Choose deployment method
echo ""
echo "🔧 Choose deployment method:"
echo "1. Docker Compose (recommended)"
echo "2. Kubernetes"
echo "3. Direct server installation"
echo "4. Systemd service"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🐳 Setting up Docker Compose deployment..."
        
        # Build and start services
        echo "Building Docker images..."
        docker-compose build
        
        echo "Starting services..."
        docker-compose up -d
        
        echo "✅ Docker Compose deployment completed!"
        echo "🌐 Application will be available at: http://localhost"
        echo "📊 Dashboard: http://localhost/dashboard"
        echo "📚 API Documentation: http://localhost/docs"
        echo ""
        echo "📋 To check status: docker-compose ps"
        echo "📋 To view logs: docker-compose logs -f"
        echo "📋 To stop: docker-compose down"
        ;;
        
    2)
        echo "☸️  Setting up Kubernetes deployment..."
        
        if ! command_exists kubectl; then
            echo "❌ kubectl is not installed. Please install kubectl first."
            exit 1
        fi
        
        # Create namespace
        kubectl create namespace ml-router --dry-run=client -o yaml | kubectl apply -f -
        
        # Create secrets from .env file
        kubectl create secret generic ml-router-secrets --from-env-file=.env -n ml-router --dry-run=client -o yaml | kubectl apply -f -
        
        # Apply Kubernetes manifests
        kubectl apply -f k8s/ -n ml-router
        
        echo "✅ Kubernetes deployment completed!"
        echo "📋 To check status: kubectl get pods -n ml-router"
        echo "📋 To get service URL: kubectl get svc -n ml-router"
        ;;
        
    3)
        echo "🖥️  Setting up direct server installation..."
        
        # Install Python dependencies
        if command_exists pip; then
            pip install -r requirements.txt
        elif command_exists pip3; then
            pip3 install -r requirements.txt
        else
            echo "❌ pip is not installed. Please install Python pip first."
            exit 1
        fi
        
        # Set up database
        echo "Setting up database..."
        python -c "from app import db; db.create_all()"
        
        echo "✅ Direct installation completed!"
        echo "🚀 To start the application:"
        echo "   gunicorn --bind 0.0.0.0:5000 --workers 4 main:app"
        echo ""
        echo "🌐 Application will be available at: http://localhost:5000"
        ;;
        
    4)
        echo "🔧 Setting up systemd service..."
        
        # Create user and directories
        sudo useradd -r -s /bin/false mlrouter || true
        sudo mkdir -p /opt/ml-router
        sudo cp -r * /opt/ml-router/
        sudo chown -R mlrouter:mlrouter /opt/ml-router
        
        # Install service file
        sudo cp systemd/ml-router.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable ml-router
        
        echo "✅ Systemd service setup completed!"
        echo "🚀 To start the service:"
        echo "   sudo systemctl start ml-router"
        echo "📋 To check status: sudo systemctl status ml-router"
        echo "📋 To view logs: sudo journalctl -u ml-router -f"
        ;;
        
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Important URLs:"
echo "   • Main Application: http://your-server/"
echo "   • Dashboard: http://your-server/dashboard"
echo "   • API Documentation: http://your-server/docs"
echo "   • Health Check: http://your-server/api/health"
echo ""
echo "📋 Next steps:"
echo "   1. Configure your firewall to allow traffic on port 80/443"
echo "   2. Set up SSL certificates for HTTPS (optional)"
echo "   3. Configure monitoring and alerting"
echo "   4. Set up backup procedures"
echo ""
echo "📄 See DEPLOYMENT.md for detailed configuration options"
echo "🔧 Check the dashboard for system status and performance metrics"