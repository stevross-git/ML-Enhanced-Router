# 🚀 AWS Deployment Guide for MLEnhancedRouter
## Domain: peoplesainetwork.com | HTTPS with Free SSL Certificate

---

## 📋 AWS Services Required

1. **EC2** - Ubuntu 22.04 LTS instance for hosting
2. **Route 53** - DNS management (or your existing DNS provider)
3. **Let's Encrypt** - Free SSL certificate
4. **Security Groups** - Firewall rules
5. **Elastic IP** - Static IP address

---

## 🔧 Step 1: Launch EC2 Instance

### Launch Instance:
```bash
# Recommended Instance Types:
# - t3.medium (2 vCPU, 4GB RAM) - Minimum for production
# - t3.large (2 vCPU, 8GB RAM) - Recommended
# - t3.xlarge (4 vCPU, 16GB RAM) - High performance

# AMI: Ubuntu Server 22.04 LTS (ami-0c7217cdde317cfec)
# Region: us-east-1 (or your preferred region)
```

### Security Group Configuration:
```bash
# Inbound Rules:
SSH         TCP  22     Your IP          # For management
HTTP        TCP  80     0.0.0.0/0        # For Let's Encrypt
HTTPS       TCP  443    0.0.0.0/0        # For application
Custom TCP  TCP  5000   0.0.0.0/0        # Docker app (temporary)

# Outbound Rules:
All Traffic All  All    0.0.0.0/0        # Allow all outbound
```

### Allocate Elastic IP:
```bash
# In EC2 Console:
# 1. Go to Elastic IPs
# 2. Allocate Elastic IP address
# 3. Associate with your EC2 instance
# Note the IP: YOUR_ELASTIC_IP
```

---

## 🔧 Step 2: Connect and Setup Server

### SSH to your server:
```bash
# Download your .pem key file
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_ELASTIC_IP
```

### Initial Server Setup:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y docker.io docker-compose git nginx certbot python3-certbot-nginx

# Add ubuntu user to docker group
sudo usermod -aG docker ubuntu
newgrp docker

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Verify installation
docker --version
docker-compose --version
nginx -v
```

---

## 🔧 Step 3: Configure DNS

### Option A: Using Route 53
```bash
# Create A Records in Route 53:
mlrouter.peoplesainetwork.com    A    YOUR_ELASTIC_IP
api.peoplesainetwork.com         A    YOUR_ELASTIC_IP
admin.peoplesainetwork.com       A    YOUR_ELASTIC_IP
```

### Option B: Using External DNS Provider
```bash
# Add these A records in your DNS provider:
mlrouter    A    YOUR_ELASTIC_IP    TTL: 3600
api         A    YOUR_ELASTIC_IP    TTL: 3600
admin       A    YOUR_ELASTIC_IP    TTL: 3600
```

### Verify DNS (wait 5-10 minutes):
```bash
# From your local machine:
nslookup mlrouter.peoplesainetwork.com
ping mlrouter.peoplesainetwork.com
```

---

## 🔧 Step 4: Clone and Configure Application

### On EC2 Instance:
```bash
# Clone repository
cd ~
git clone <your-repo-url> MLEnhancedRouter
cd MLEnhancedRouter

# Create production environment file
nano .env
# Paste your production .env content here
# Save and exit (Ctrl+X, Y, Enter)

# Create necessary directories
mkdir -p logs
mkdir -p data
```

---

## 🔧 Step 5: Configure Nginx (Initial HTTP)

### Create Nginx configuration:
```bash
sudo nano /etc/nginx/sites-available/mlrouter
```

### Paste this configuration:
```nginx
# Initial HTTP configuration for Let's Encrypt
server {
    listen 80;
    server_name mlrouter.peoplesainetwork.com api.peoplesainetwork.com admin.peoplesainetwork.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Enable site:
```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/mlrouter /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

---

## 🔧 Step 6: Get Free SSL Certificate with Let's Encrypt

### Install certificate:
```bash
# Get certificate for all subdomains
sudo certbot --nginx -d mlrouter.peoplesainetwork.com -d api.peoplesainetwork.com -d admin.peoplesainetwork.com

# Follow prompts:
# - Enter email: admin@peoplesainetwork.com
# - Agree to terms: A
# - Share email: N (optional)
# - Redirect HTTP to HTTPS: 2 (redirect)
```

### Verify SSL installation:
```bash
# Check certificate
sudo certbot certificates

# Test auto-renewal
sudo certbot renew --dry-run
```

---

## 🔧 Step 7: Update Nginx for Production HTTPS

### Edit Nginx configuration:
```bash
sudo nano /etc/nginx/sites-available/mlrouter
```

### Update with full HTTPS configuration:
```nginx
# HTTP to HTTPS redirect
server {
    listen 80;
    server_name mlrouter.peoplesainetwork.com api.peoplesainetwork.com admin.peoplesainetwork.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS Server
server {
    listen 443 ssl http2;
    server_name mlrouter.peoplesainetwork.com api.peoplesainetwork.com admin.peoplesainetwork.com;

    # SSL Configuration (managed by Certbot)
    ssl_certificate /etc/letsencrypt/live/mlrouter.peoplesainetwork.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mlrouter.peoplesainetwork.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Proxy Configuration
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port 443;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://localhost:5000/api/health;
    }

    # Client body size (for file uploads)
    client_max_body_size 10M;
}
```

### Apply changes:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🔧 Step 8: Deploy Application with Docker

### Start the application:
```bash
cd ~/MLEnhancedRouter

# Build and start containers
docker-compose up -d

# Check container status
docker-compose ps

# View logs
docker-compose logs -f
```

---

## 🔧 Step 9: Setup Auto-renewal for SSL

### Create renewal script:
```bash
sudo nano /etc/cron.d/certbot-renewal
```

### Add cron job:
```cron
# Renew certificates twice daily
0 0,12 * * * root certbot renew --quiet --post-hook "systemctl reload nginx"
```

---

## 🔧 Step 10: Configure AWS Security Best Practices

### Update Security Group (remove port 5000):
```bash
# After verifying HTTPS works, remove the temporary rule:
# Remove: Custom TCP  TCP  5000  0.0.0.0/0

# Final rules should be:
SSH    TCP  22   Your IP
HTTP   TCP  80   0.0.0.0/0  # For redirect to HTTPS
HTTPS  TCP  443  0.0.0.0/0  # Main application
```

### Setup CloudWatch Monitoring:
```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb

# Configure monitoring for:
# - CPU, Memory, Disk usage
# - Application logs
# - Nginx access/error logs
```

---

## 🔧 Step 11: Create Systemd Service (Optional)

### Create service for auto-start:
```bash
sudo nano /etc/systemd/system/mlrouter.service
```

```ini
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
```

### Enable service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mlrouter.service
sudo systemctl start mlrouter.service
```

---

## ✅ Verification Steps

### 1. Test HTTPS access:
```bash
# From your local machine:
curl https://mlrouter.peoplesainetwork.com/api/health
curl https://api.peoplesainetwork.com/api/health
curl https://admin.peoplesainetwork.com/api/health
```

### 2. Check SSL certificate:
```bash
# Check SSL grade
# Visit: https://www.ssllabs.com/ssltest/analyze.html?d=mlrouter.peoplesainetwork.com
```

### 3. Monitor logs:
```bash
# Application logs
docker-compose logs -f app

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## 🚨 Troubleshooting

### DNS Issues:
```bash
# Check DNS propagation
nslookup mlrouter.peoplesainetwork.com 8.8.8.8
dig mlrouter.peoplesainetwork.com
```

### Certificate Issues:
```bash
# Check certificate status
sudo certbot certificates

# Manually renew
sudo certbot renew

# Check Nginx SSL config
sudo nginx -t
```

### Docker Issues:
```bash
# Restart containers
docker-compose restart

# Check container logs
docker-compose logs app
docker-compose logs db

# Check disk space
df -h
```

---

## 📊 Monitoring URLs

Once deployed, access your application at:

- **Main App**: https://mlrouter.peoplesainetwork.com
- **API**: https://api.peoplesainetwork.com
- **Admin**: https://admin.peoplesainetwork.com
- **Health Check**: https://mlrouter.peoplesainetwork.com/api/health

---

## 💰 AWS Cost Estimate

### Monthly costs (US East 1):
- **EC2 t3.medium**: ~$30/month
- **Elastic IP**: Free (when attached)
- **Data Transfer**: ~$9/GB after 1GB free
- **Route 53**: $0.50/hosted zone + $0.40/million queries
- **SSL Certificate**: FREE (Let's Encrypt)

**Total**: ~$40-60/month

---

## 🔐 Security Checklist

- [x] HTTPS enabled with valid SSL certificate
- [x] HTTP redirects to HTTPS
- [x] Security headers configured
- [x] Firewall rules restricted
- [x] Auto-renewal for SSL certificates
- [x] Docker running as non-root
- [x] Sensitive data in environment variables
- [x] Regular security updates scheduled

---

## 🎯 Next Steps

1. **Setup Backups**: Configure automated RDS backups
2. **Setup Monitoring**: CloudWatch or Datadog
3. **Setup Alerts**: For downtime, high CPU, errors
4. **Setup CI/CD**: GitHub Actions for automated deployment
5. **Setup WAF**: AWS WAF for additional security

**Your MLEnhancedRouter is now ready for production on AWS with HTTPS! 🚀**