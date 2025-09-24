# Deployment Guide

This guide covers various deployment options for the Elizabeth Andrews Bank Statement Parser API.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- API keys for Llama Cloud and Google Gemini AI
- (Optional) Docker and Docker Compose

## Local Development

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/elizabeth-andrews-bank-parser.git
cd elizabeth-andrews-bank-parser
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp env.example .env
# Edit .env with your API keys
```

### 5. Run the Application

```bash
python start_api_clean.py
```

## Docker Deployment

### 1. Build Docker Image

```bash
docker build -t elizabeth-andrews-bank-parser .
```

### 2. Run with Docker Compose

```bash
# Set environment variables
export LLAMA_CLOUD_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here

# Start the service
docker-compose up -d
```

### 3. Check Status

```bash
docker-compose ps
docker-compose logs -f
```

## Production Deployment

### Option 1: Traditional Server

#### 1. Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Install nginx (optional, for reverse proxy)
sudo apt install nginx -y
```

#### 2. Application Setup

```bash
# Create application directory
sudo mkdir -p /opt/bank-parser
sudo chown $USER:$USER /opt/bank-parser

# Clone repository
cd /opt/bank-parser
git clone https://github.com/yourusername/elizabeth-andrews-bank-parser.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
nano .env  # Add your API keys
```

#### 3. Create Systemd Service

Create `/etc/systemd/system/bank-parser.service`:

```ini
[Unit]
Description=Elizabeth Andrews Bank Parser API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/bank-parser
Environment=PATH=/opt/bank-parser/venv/bin
ExecStart=/opt/bank-parser/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

#### 4. Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable bank-parser
sudo systemctl start bank-parser
sudo systemctl status bank-parser
```

#### 5. Configure Nginx (Optional)

Create `/etc/nginx/sites-available/bank-parser`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/bank-parser /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Option 2: Cloud Deployment

#### AWS EC2

1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS or newer
   - Select appropriate instance type (t3.medium or larger)
   - Configure security group to allow HTTP/HTTPS traffic

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv nginx -y
   ```

3. **Deploy Application**
   ```bash
   git clone https://github.com/yourusername/elizabeth-andrews-bank-parser.git
   cd elizabeth-andrews-bank-parser
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Configure and Start**
   ```bash
   cp env.example .env
   # Add your API keys to .env
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

#### Google Cloud Platform

1. **Create App Engine Application**
   ```bash
   gcloud app create --region=us-central
   ```

2. **Create app.yaml**
   ```yaml
   runtime: python39
   
   env_variables:
     LLAMA_CLOUD_API_KEY: "your_key_here"
     GOOGLE_API_KEY: "your_key_here"
   
   handlers:
   - url: /.*
     script: auto
   ```

3. **Deploy**
   ```bash
   gcloud app deploy
   ```

#### Heroku

1. **Install Heroku CLI**
   ```bash
   # Follow instructions at https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set LLAMA_CLOUD_API_KEY=your_key_here
   heroku config:set GOOGLE_API_KEY=your_key_here
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

### Option 3: Kubernetes Deployment

#### 1. Create Kubernetes Manifests

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bank-parser-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bank-parser-api
  template:
    metadata:
      labels:
        app: bank-parser-api
    spec:
      containers:
      - name: bank-parser-api
        image: elizabeth-andrews-bank-parser:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLAMA_CLOUD_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: llama-cloud-api-key
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: google-api-key
```

**service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: bank-parser-service
spec:
  selector:
    app: bank-parser-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

**secret.yaml**:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
data:
  llama-cloud-api-key: <base64-encoded-key>
  google-api-key: <base64-encoded-key>
```

#### 2. Deploy to Kubernetes

```bash
kubectl apply -f secret.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LLAMA_CLOUD_API_KEY` | Llama Cloud API key | `llx-...` |
| `GOOGLE_API_KEY` | Google Gemini AI API key | `AIza...` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | Host to bind to | `0.0.0.0` |
| `API_PORT` | Port to listen on | `8000` |
| `MAX_FILE_SIZE` | Maximum file size in bytes | `52428800` |
| `PROCESSING_TIMEOUT` | Processing timeout in seconds | `600` |

## Monitoring and Logging

### Health Checks

The API provides built-in health checks:

```bash
curl http://your-server:8000/health
```

### Logging

Logs are written to stdout and can be captured by your process manager:

```bash
# View logs with systemd
sudo journalctl -u bank-parser -f

# View logs with Docker
docker-compose logs -f

# View logs with Kubernetes
kubectl logs -f deployment/bank-parser-api
```

### Monitoring

Consider setting up monitoring with:

- **Prometheus + Grafana**: For metrics and dashboards
- **ELK Stack**: For log aggregation and analysis
- **Uptime monitoring**: For availability checks

## Security Considerations

### 1. API Keys
- Store API keys in environment variables or secret management systems
- Never commit API keys to version control
- Rotate keys regularly

### 2. Network Security
- Use HTTPS in production
- Configure proper firewall rules
- Consider VPN or private networks for internal APIs

### 3. File Security
- Validate all uploaded files
- Implement file size limits
- Scan files for malware if needed

### 4. Access Control
- Implement authentication if needed
- Use rate limiting
- Monitor for suspicious activity

## Scaling Considerations

### Horizontal Scaling
- Use load balancers for multiple instances
- Implement session affinity if needed
- Consider database clustering for job storage

### Vertical Scaling
- Monitor CPU and memory usage
- Adjust instance sizes based on load
- Implement auto-scaling policies

### Performance Optimization
- Use CDN for static assets
- Implement caching strategies
- Optimize database queries
- Use connection pooling

## Troubleshooting

### Common Issues

1. **API not starting**
   - Check environment variables
   - Verify Python version
   - Check port availability

2. **Processing failures**
   - Verify API keys are valid
   - Check file format and size
   - Review error logs

3. **Memory issues**
   - Increase instance memory
   - Implement file cleanup
   - Monitor memory usage

### Debug Mode

Enable debug logging:

```bash
uvicorn main:app --log-level debug
```

### Support

For deployment issues:

- Check the [GitHub Issues](https://github.com/yourusername/elizabeth-andrews-bank-parser/issues)
- Review the [API Documentation](API_DOCUMENTATION.md)
- Contact support@example.com
