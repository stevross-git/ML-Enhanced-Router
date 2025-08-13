#!/bin/bash

# MLEnhancedRouter AWS Quick Setup Script
# For Ubuntu 22.04 LTS on EC2
# Domain: peoplesainetwork.com

set -e

echo "========================================="
echo "MLEnhancedRouter AWS Setup Script"
echo "Domain: peoplesainetwork.com"
echo "========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as ubuntu user
if [ "$USER" != "ubuntu" ]; then
    echo -e "${RED}Please run this script as ubuntu user${NC}"
    exit 1
fi

# Get Elastic IP
echo -e "${YELLOW}Enter your AWS Elastic IP address:${NC}"
read ELASTIC_IP

# Confirm domain
echo -e "${YELLOW}This will setup for peoplesainetwork.com subdomains.${NC}"
echo -e "${YELLOW}Make sure DNS A records point to: ${ELASTIC_IP}${NC}"
echo -e "${YELLOW}Required records:${NC}"
echo "  - mlrouter.peoplesainetwork.com -> ${ELASTIC_IP}"
echo "  - api.peoplesainetwork.com -> ${ELASTIC_IP}"
echo "  - admin.peoplesainetwork.com -> ${ELASTIC_IP}"
echo -e "${YELLOW}Press Enter to continue or Ctrl+C to cancel...${NC}"
read

# Update system
echo -e "${GREEN}Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Install required packages
echo -e "${GREEN}Installing Docker, Nginx, and Certbot...${NC}"
sudo apt install -y docker.io docker-compose git nginx certbot python3-certbot-nginx ufw

# Configure Docker
echo -e "${GREEN}Configuring Docker...${NC}"
sudo usermod -aG docker ubuntu
sudo systemctl start docker
sudo systemctl enable docker

# Configure firewall
echo -e "${GREEN}Configuring firewall...${NC}"
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# Create application directory
echo -e "${GREEN}Setting up application directory...${NC}"
cd ~
if [ ! -d "MLEnhancedRouter" ]; then
    echo -e "${YELLOW}Please clone your repository first:${NC}"
    echo "git clone <your-repo-url> MLEnhancedRouter"
    echo "Then run this script again."
    exit 1
fi

cd MLEnhancedRouter

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}.env file not found!${NC}"
    echo -e "${YELLOW}Please create .env file with your production configuration${NC}"
    echo "Then run this script again."
    exit 1
fi

# Create necessary directories
mkdir -p logs
mkdir -p data

# Configure Nginx (initial HTTP)
echo -e "${GREEN}Configuring Nginx...${NC}"
sudo tee /etc/nginx/sites-available/mlrouter > /dev/null <<EOF
server {
    listen 80;
    server_name mlrouter.peoplesainetwork.com api.peoplesainetwork.com admin.peoplesainetwork.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/mlrouter /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test and reload Nginx
sudo nginx -t
sudo systemctl restart nginx

# Start Docker containers
echo -e "${GREEN}Starting Docker containers...${NC}"
docker-compose down 2>/dev/null || true
docker-compose up -d

# Wait for application to start
echo -e "${YELLOW}Waiting for application to start...${NC}"
sleep 10

# Check if application is running
if curl -f http://localhost:5000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}Application is running!${NC}"
else
    echo -e "${RED}Application failed to start. Check logs with: docker-compose logs${NC}"
    exit 1
fi

# Get SSL certificate
echo -e "${GREEN}Getting SSL certificate from Let's Encrypt...${NC}"
echo -e "${YELLOW}Enter your email for Let's Encrypt notifications:${NC}"
read EMAIL

sudo certbot --nginx \
    -d mlrouter.peoplesainetwork.com \
    -d api.peoplesainetwork.com \
    -d admin.peoplesainetwork.com \
    --non-interactive \
    --agree-tos \
    --email $EMAIL \
    --redirect

# Update Nginx for production
echo -e "${GREEN}Updating Nginx for production HTTPS...${NC}"
sudo tee /etc/nginx/sites-available/mlrouter > /dev/null <<'EOF'
server {
    listen 80;
    server_name mlrouter.peoplesainetwork.com api.peoplesainetwork.com admin.peoplesainetwork.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name mlrouter.peoplesainetwork.com api.peoplesainetwork.com admin.peoplesainetwork.com;

    ssl_certificate /etc/letsencrypt/live/mlrouter.peoplesainetwork.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mlrouter.peoplesainetwork.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health {
        access_log off;
        proxy_pass http://localhost:5000/api/health;
    }

    client_max_body_size 10M;
}
EOF

# Reload Nginx with new config
sudo nginx -t
sudo systemctl reload nginx

# Setup auto-renewal
echo -e "${GREEN}Setting up SSL auto-renewal...${NC}"
(crontab -l 2>/dev/null; echo "0 0,12 * * * certbot renew --quiet --post-hook 'systemctl reload nginx'") | crontab -

# Create systemd service
echo -e "${GREEN}Creating systemd service for auto-start...${NC}"
sudo tee /etc/systemd/system/mlrouter.service > /dev/null <<EOF
[Unit]
Description=ML Enhanced Router
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/MLEnhancedRouter
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0
User=ubuntu
Group=docker

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlrouter.service

# Final checks
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "${GREEN}Your application is now available at:${NC}"
echo "  - https://mlrouter.peoplesainetwork.com"
echo "  - https://api.peoplesainetwork.com"
echo "  - https://admin.peoplesainetwork.com"
echo ""
echo -e "${YELLOW}Testing HTTPS endpoints...${NC}"

# Test endpoints
if curl -f https://mlrouter.peoplesainetwork.com/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ mlrouter.peoplesainetwork.com is working!${NC}"
else
    echo -e "${RED}✗ mlrouter.peoplesainetwork.com is not responding${NC}"
fi

if curl -f https://api.peoplesainetwork.com/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ api.peoplesainetwork.com is working!${NC}"
else
    echo -e "${RED}✗ api.peoplesainetwork.com is not responding${NC}"
fi

if curl -f https://admin.peoplesainetwork.com/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ admin.peoplesainetwork.com is working!${NC}"
else
    echo -e "${RED}✗ admin.peoplesainetwork.com is not responding${NC}"
fi

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Monitor logs: docker-compose logs -f"
echo "  2. Check SSL: https://www.ssllabs.com/ssltest/analyze.html?d=mlrouter.peoplesainetwork.com"
echo "  3. Setup monitoring and backups"
echo ""
echo -e "${GREEN}Deployment complete! 🚀${NC}"