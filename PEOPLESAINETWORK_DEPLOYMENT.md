# 🚀 MLEnhancedRouter Deployment for peoplesainetwork.com

**Domain:** peoplesainetwork.com  
**Security Status:** ✅ Production Approved (9/10)  
**Date:** 2025-08-09  

---

## 🌐 Subdomain Strategy for peoplesainetwork.com

### Recommended Subdomains:

1. **`mlrouter.peoplesainetwork.com`** - Main ML Router application
2. **`api.peoplesainetwork.com`** - API endpoints (can point to same server)
3. **`admin.peoplesainetwork.com`** - Admin panel and management interface
4. **`dash.peoplesainetwork.com`** - Optional: Monitoring dashboard

### DNS Configuration Needed:

```dns
# A Records (replace with your server IP)
mlrouter.peoplesainetwork.com.    A    YOUR_SERVER_IP
api.peoplesainetwork.com.         A    YOUR_SERVER_IP
admin.peoplesainetwork.com.       A    YOUR_SERVER_IP

# CNAME Records (alternative)
mlrouter.peoplesainetwork.com.    CNAME    your-server.example.com
api.peoplesainetwork.com.         CNAME    your-server.example.com
admin.peoplesainetwork.com.       CNAME    your-server.example.com
```

---

## 🔐 SSL Certificate Requirements

### Option 1: Wildcard Certificate (Recommended)
```bash
# Get wildcard cert for *.peoplesainetwork.com
# Covers all subdomains with single certificate
```

### Option 2: Multi-domain Certificate
```bash
# Certificate for specific subdomains:
# - mlrouter.peoplesainetwork.com
# - api.peoplesainetwork.com  
# - admin.peoplesainetwork.com
```

### Let's Encrypt Example:
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get wildcard certificate
sudo certbot --manual --preferred-challenges=dns certonly -d *.peoplesainetwork.com -d peoplesainetwork.com

# Or specific subdomains
sudo certbot --nginx -d mlrouter.peoplesainetwork.com -d api.peoplesainetwork.com -d admin.peoplesainetwork.com
```

---

## ⚙️ Nginx Configuration

Create `/etc/nginx/sites-available/mlrouter.peoplesainetwork.com`:

```nginx
# MLRouter for peoplesainetwork.com
server {
    listen 80;
    server_name mlrouter.peoplesainetwork.com api.peoplesainetwork.com admin.peoplesainetwork.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name mlrouter.peoplesainetwork.com api.peoplesainetwork.com admin.peoplesainetwork.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/peoplesainetwork.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/peoplesainetwork.com/privkey.pem;
    
    # SSL Security
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    # Proxy to Docker container
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://localhost:5000/api/health;
    }

    # Static files (if served by nginx)
    location /static {
        alias /path/to/mlrouter/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

---

## 📧 Office 365 Email Configuration

Since you're using `peoplesainetwork.com`, configure Office 365 SMTP:

```env
# Email Configuration
MAIL_SERVER=smtp.office365.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=noreply@peoplesainetwork.com
MAIL_PASSWORD=your_office365_app_password

# Make sure to create app password in Office 365 admin
```

### Office 365 Setup Steps:
1. Go to **Microsoft 365 Admin Center**
2. Navigate to **Users** → **Active users**
3. Select service account user
4. Go to **Mail** tab → **Manage email apps**
5. Generate **App password** for SMTP authentication

---

## 🔑 Azure OAuth Configuration

Update your Azure App Registration:

```env
# OAuth Configuration
OFFICE365_CLIENT_ID=your_azure_app_client_id
OFFICE365_CLIENT_SECRET=your_azure_app_client_secret
OFFICE365_REDIRECT_URI=https://mlrouter.peoplesainetwork.com/auth/office365/callback
```

### Azure Portal Steps:
1. Go to **Azure Portal** → **App Registrations**
2. Select your app or create new one
3. **Redirect URIs**: Add `https://mlrouter.peoplesainetwork.com/auth/office365/callback`
4. **API Permissions**: Ensure proper Microsoft Graph permissions
5. **Certificates & Secrets**: Create client secret

---

## 🚀 Deployment Commands

```bash
# 1. Copy peoplesainetwork environment
cp .env.production.peoplesainetwork .env

# 2. Update with your actual secrets
nano .env
# Update:
# - Database passwords
# - JWT secrets  
# - AI API keys
# - Office 365 credentials
# - Email passwords

# 3. Run production security check
python final_production_verification.py

# 4. Deploy with Docker
docker-compose -f docker-compose.yml up -d

# 5. Verify deployment
curl https://mlrouter.peoplesainetwork.com/api/health
```

---

## 🔍 Domain-Specific Testing

### Health Check URLs:
- **Main App**: `https://mlrouter.peoplesainetwork.com/api/health`
- **Database**: `https://mlrouter.peoplesainetwork.com/api/health/db`
- **Authentication**: `https://mlrouter.peoplesainetwork.com/auth/login`
- **API**: `https://api.peoplesainetwork.com/query` (if using separate subdomain)

### CORS Testing:
```bash
# Test CORS from browser console
fetch('https://mlrouter.peoplesainetwork.com/api/health', {
  method: 'GET',
  mode: 'cors'
}).then(response => console.log(response.status));
```

### SSL Certificate Verification:
```bash
# Check SSL certificate
openssl s_client -connect mlrouter.peoplesainetwork.com:443 -servername mlrouter.peoplesainetwork.com

# Check certificate expiry
curl -vI https://mlrouter.peoplesainetwork.com 2>&1 | grep -i expire
```

---

## 📊 Monitoring & Analytics

### Recommended Monitoring Setup:

1. **Uptime Monitoring**: Monitor all subdomains
   - `mlrouter.peoplesainetwork.com`
   - `api.peoplesainetwork.com`
   - `admin.peoplesainetwork.com`

2. **SSL Certificate Monitoring**: 
   - Expiry alerts (30, 7, 1 days before expiry)
   - Certificate chain validation

3. **DNS Monitoring**:
   - A record resolution
   - Subdomain availability

### Health Check Endpoints:
```bash
# Application health
GET https://mlrouter.peoplesainetwork.com/api/health

# Database connectivity
GET https://mlrouter.peoplesainetwork.com/api/health/db

# Authentication service
GET https://mlrouter.peoplesainetwork.com/api/health/auth
```

---

## 🛡️ Security Considerations for peoplesainetwork.com

### Domain-Specific Security:
- **HSTS**: Enforced for all peoplesainetwork.com subdomains
- **CORS**: Strictly limited to peoplesainetwork.com subdomains
- **CSP**: Configured for your domain resources
- **Email**: Using Office 365 with app passwords (more secure)

### Security Headers:
```nginx
# Add to nginx config
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
add_header Content-Security-Policy "default-src 'self' *.peoplesainetwork.com; script-src 'self' 'unsafe-inline' *.peoplesainetwork.com" always;
```

---

## 🎯 Go-Live Checklist for peoplesainetwork.com

- [ ] **DNS records** configured for subdomains
- [ ] **SSL certificates** installed and validated
- [ ] **Nginx** configured with proper proxy settings
- [ ] **Environment file** updated with domain-specific settings
- [ ] **Azure OAuth** redirect URIs updated
- [ ] **Office 365 email** configured and tested
- [ ] **CORS origins** set to peoplesainetwork.com subdomains
- [ ] **Security verification** passed (18/18 tests)
- [ ] **Health checks** responding from all subdomains
- [ ] **Monitoring** configured for uptime and SSL
- [ ] **Backup strategy** implemented

**Status: Ready for peoplesainetwork.com deployment! 🚀**