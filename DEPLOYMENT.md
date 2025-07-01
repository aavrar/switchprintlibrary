# Deployment Guide

Comprehensive deployment guide for SwitchPrint in production environments.

## 🚀 Quick Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Setup
```yaml
version: '3.8'

services:
  codeswitch-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CODESWITCH_DB_PATH=/data/conversations.db
      - CODESWITCH_LOG_LEVEL=INFO
      - CODESWITCH_SECURITY_LEVEL=strict
    volumes:
      - ./data:/data
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: codeswitch
      POSTGRES_USER: codeswitch
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

## 🏗️ Production Architecture

### Microservices Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │  Authentication │
│    (nginx)      │────│   (FastAPI)     │────│    Service      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │                 │
              ┌─────────────────┐  ┌─────────────────┐
              │   Detection     │  │   Security      │
              │   Service       │  │   Service       │
              └─────────────────┘  └─────────────────┘
                       │                 │
              ┌─────────────────┐  ┌─────────────────┐
              │   Memory        │  │   Monitoring    │
              │   Service       │  │   Service       │
              └─────────────────┘  └─────────────────┘
                       │
              ┌─────────────────┐
              │   Database      │
              │   (PostgreSQL)  │
              └─────────────────┘
```

### FastAPI Production Application

```python
# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import logging
from typing import List, Optional

from codeswitch_ai import (
    EnsembleDetector, PrivacyProtector, SecurityMonitor,
    InputValidator, ModelSecurityAuditor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SwitchPrint API",
    description="Production API for multilingual code-switching detection with SwitchPrint",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware
security = HTTPBearer()
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "*.yourdomain.com"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize AI components
detector = EnsembleDetector(
    use_fasttext=True,
    use_transformer=True,
    ensemble_strategy="weighted_average"
)

privacy_protector = PrivacyProtector()
security_monitor = SecurityMonitor(log_file="/app/logs/security.log")
input_validator = InputValidator()

class DetectionRequest(BaseModel):
    text: str
    user_languages: Optional[List[str]] = None
    user_id: Optional[str] = None
    apply_privacy_protection: bool = True

class DetectionResponse(BaseModel):
    detected_languages: List[str]
    confidence: float
    switch_points: List[dict]
    phrases: List[dict]
    processing_time: float
    privacy_applied: bool
    security_events: int

@app.post("/detect", response_model=DetectionResponse)
async def detect_language(
    request: DetectionRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Detect code-switching in text with security and privacy protection."""
    
    try:
        # 1. Authentication (implement your auth logic)
        user_id = validate_token(credentials.credentials)
        
        # 2. Input validation
        validation_result = input_validator.validate(request.text)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid input: {validation_result.threats_detected}"
            )
        
        # 3. Privacy protection
        protected_text = request.text
        privacy_applied = False
        
        if request.apply_privacy_protection:
            privacy_result = privacy_protector.protect_text(
                validation_result.sanitized_text,
                source_id=request.user_id or "anonymous"
            )
            protected_text = privacy_result['protected_text']
            privacy_applied = privacy_result['protection_applied']
        
        # 4. Language detection
        detection_result = detector.detect_language(
            protected_text,
            user_languages=request.user_languages
        )
        
        # 5. Security monitoring
        security_events = security_monitor.process_request(
            source_id="api_detection",
            request_data={
                'text_size': len(request.text),
                'detected_languages': detection_result.detected_languages,
                'user_languages': request.user_languages
            },
            user_id=user_id,
            success=True
        )
        
        return DetectionResponse(
            detected_languages=detection_result.detected_languages,
            confidence=detection_result.confidence,
            switch_points=[
                {"position": p[0], "from": p[1], "to": p[2]} 
                for p in detection_result.switch_points
            ],
            phrases=detection_result.phrases,
            processing_time=detection_result.processing_time,
            privacy_applied=privacy_applied,
            security_events=len(security_events)
        )
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        
        # Log security event for failed requests
        security_monitor.process_request(
            source_id="api_detection",
            request_data={'error': str(e)},
            user_id=user_id if 'user_id' in locals() else None,
            success=False
        )
        
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": time.time()
    }

@app.get("/metrics")
async def get_metrics(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get system metrics for monitoring."""
    user_id = validate_token(credentials.credentials)
    
    if not is_admin_user(user_id):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "system": get_system_metrics(),
        "security": security_monitor.get_monitoring_status(),
        "detector": detector.get_performance_stats()
    }

def validate_token(token: str) -> str:
    """Validate JWT token and return user ID."""
    # Implement your JWT validation logic
    # Return user_id if valid, raise HTTPException if invalid
    pass

def is_admin_user(user_id: str) -> bool:
    """Check if user has admin privileges."""
    # Implement your authorization logic
    pass

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True
    )
```

## 🔧 Configuration Management

### Environment-based Configuration

```python
# app/config.py
import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Application settings
    app_name: str = "Code-Switch AI API"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Security settings
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    security_level: str = "strict"
    
    # Database settings
    database_url: str = "sqlite:///./conversations.db"
    redis_url: str = "redis://localhost:6379"
    
    # AI Model settings
    model_path: Optional[str] = None
    use_gpu: bool = True
    cache_size: int = 1000
    
    # Monitoring settings
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Performance settings
    max_workers: int = 4
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit: int = 100  # requests per minute
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Production Environment Variables

```bash
# .env.production
APP_NAME="SwitchPrint Production API"
DEBUG=false
SECRET_KEY="your-super-secret-key-here"
SECURITY_LEVEL=strict

# Database
DATABASE_URL="postgresql://user:password@postgres:5432/codeswitch"
REDIS_URL="redis://redis:6379"

# Performance
USE_GPU=true
MAX_WORKERS=8
CACHE_SIZE=5000
RATE_LIMIT=1000

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090

# Security
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

## 📊 Monitoring and Observability

### Prometheus Metrics

```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
request_count = Counter('codeswitch_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('codeswitch_request_duration_seconds', 'Request duration')
active_connections = Gauge('codeswitch_active_connections', 'Active connections')
detection_accuracy = Gauge('codeswitch_detection_accuracy', 'Detection accuracy')
security_events = Counter('codeswitch_security_events_total', 'Security events', ['event_type'])

def setup_metrics():
    """Start Prometheus metrics server."""
    start_http_server(settings.metrics_port)

@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    """Add Prometheus metrics to all requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    request_count.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    request_duration.observe(time.time() - start_time)
    
    return response
```

### Logging Configuration

```python
# app/logging_config.py
import logging
import logging.handlers
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

def setup_logging():
    """Setup structured logging for production."""
    
    # Create formatter
    formatter = JSONFormatter()
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/app.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        handlers=[file_handler, console_handler]
    )
```

## 🔒 Production Security

### SSL/TLS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/yourdomain.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://codeswitch-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Security Best Practices

1. **Input Validation**: Always validate and sanitize inputs
2. **Authentication**: Use JWT tokens with proper expiration
3. **Authorization**: Implement role-based access control
4. **Rate Limiting**: Prevent abuse with rate limiting
5. **Monitoring**: Monitor security events and anomalies
6. **Updates**: Keep dependencies updated
7. **Secrets**: Use secure secret management
8. **Network**: Use private networks and firewalls

## 📈 Performance Optimization

### Database Optimization

```python
# app/database.py
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker

# Production database configuration
engine = create_engine(
    settings.database_url,
    poolclass=pool.QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Set to True for debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Caching Strategy

```python
# app/cache.py
import redis
import json
import pickle
from typing import Optional, Any

class CacheManager:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cached value with TTL."""
        try:
            self.redis_client.setex(
                key, 
                ttl, 
                pickle.dumps(value)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete cached value."""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")

# Initialize cache
cache = CacheManager(settings.redis_url)

# Usage in API endpoints
@app.post("/detect")
async def detect_language_cached(request: DetectionRequest):
    # Check cache first
    cache_key = f"detection:{hash(request.text)}:{hash(str(request.user_languages))}"
    cached_result = cache.get(cache_key)
    
    if cached_result:
        return cached_result
    
    # Perform detection
    result = await detect_language(request)
    
    # Cache result
    cache.set(cache_key, result, ttl=1800)  # 30 minutes
    
    return result
```

## 🚀 Kubernetes Deployment

### Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codeswitch-api
  labels:
    app: codeswitch-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: codeswitch-api
  template:
    metadata:
      labels:
        app: codeswitch-api
    spec:
      containers:
      - name: api
        image: codeswitch-ai:2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: codeswitch-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: codeswitch-service
spec:
  selector:
    app: codeswitch-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: codeswitch-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: codeswitch-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 📋 Deployment Checklist

### Pre-deployment

- [ ] Security audit completed
- [ ] Performance testing done
- [ ] Load testing passed
- [ ] SSL certificates configured
- [ ] Environment variables set
- [ ] Database migrations applied
- [ ] Monitoring configured
- [ ] Backup strategy in place

### Post-deployment

- [ ] Health checks passing
- [ ] Metrics collection working
- [ ] Logs aggregation configured
- [ ] Alerts configured
- [ ] Performance baseline established
- [ ] Security monitoring active
- [ ] Documentation updated

### Monitoring Endpoints

- Health: `GET /health`
- Metrics: `GET /metrics` (authenticated)
- Ready: `GET /ready`
- Version: `GET /version`

This deployment guide provides a comprehensive foundation for running SwitchPrint in production environments with proper security, monitoring, and scalability considerations.