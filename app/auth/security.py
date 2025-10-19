"""
Production Security Layer
Authentication, authorization, rate limiting, and security middleware
"""

import os
try:
    import jwt
except ImportError:
    jwt = None
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Request, Depends, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from passlib.context import CryptContext
import redis
from functools import wraps
import logging

# Import monitoring components for metrics
try:
    from monitoring.metrics import metrics
    from monitoring.logging import get_logger, with_correlation
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    metrics = None
    def get_logger(name): return logging.getLogger(name)
    def with_correlation(): return None

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Redis for rate limiting and session management
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/1"))

# Security scheme
security = HTTPBearer()

logger = get_logger('app.security') if MONITORING_AVAILABLE else logging.getLogger(__name__)


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics and performance data"""
    
    async def dispatch(self, request: Request, call_next):
        # Start timing
        start_time = time.time()
        
        # Extract request info
        method = request.method
        path = request.url.path
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics if available
            if MONITORING_AVAILABLE and metrics:
                # Sanitize endpoint for metrics (remove IDs and query params)
                endpoint = self._sanitize_endpoint(path)
                
                # Record HTTP request metrics
                metrics.record_http_request(method, endpoint, status_code, duration)
                
                # Log slow requests
                if duration > 2.0:  # Slow request threshold
                    logger.warning(
                        f"Slow request detected: {method} {path}",
                        extra={
                            'method': method,
                            'path': path,
                            'status_code': status_code,
                            'duration_seconds': duration,
                            'performance_issue': True
                        }
                    )
            
            return response
            
        except Exception as e:
            # Calculate duration for failed requests
            duration = time.time() - start_time
            
            # Record error metrics
            if MONITORING_AVAILABLE and metrics:
                endpoint = self._sanitize_endpoint(path)
                metrics.record_http_request(method, endpoint, 500, duration)
            
            # Log error
            logger.error(
                f"Request failed: {method} {path}",
                extra={
                    'method': method,
                    'path': path,
                    'duration_seconds': duration,
                    'error': str(e)
                },
                exc_info=True
            )
            
            raise
    
    def _sanitize_endpoint(self, path: str) -> str:
        """Sanitize endpoint path for metrics"""
        # Replace IDs with placeholder
        import re
        
        # Replace UUIDs
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{id}', path)
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Limit path length for metrics
        if len(path) > 50:
            path = path[:47] + "..."
        
        return path


class SecurityHeaders(BaseHTTPMiddleware):
    """Security headers middleware"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class SecurityManager:
    """Centralized security management"""
    
    def __init__(self):
        self.admin_users = set(os.getenv("ADMIN_USERS", "admin@company.com").split(","))
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment or database"""
        api_keys = {}
        
        # Load from environment (format: API_KEY_NAME=key_value)
        for key, value in os.environ.items():
            if key.startswith("API_KEY_"):
                name = key.replace("API_KEY_", "").lower()
                api_keys[value] = name
        
        # Default admin API key
        if "ADMIN_API_KEY" in os.environ:
            api_keys[os.environ["ADMIN_API_KEY"]] = "admin"
            
        return api_keys
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def verify_api_key(self, api_key: str) -> Optional[str]:
        """Verify API key and return user type"""
        return self.api_keys.get(api_key)


# Global security manager instance
security_manager = SecurityManager()


class RateLimiter:
    """Redis-based rate limiting"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def is_allowed(self, key: str, limit: int, window: int) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit
        
        Args:
            key: Unique identifier (IP, user_id, etc.)
            limit: Number of requests allowed
            window: Time window in seconds
        
        Returns:
            (allowed, info) tuple
        """
        try:
            current_time = int(time.time())
            window_start = current_time - window
            
            # Clean old entries
            self.redis.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
            
            # Count current requests
            current_requests = self.redis.zcard(f"rate_limit:{key}")
            
            if current_requests < limit:
                # Add current request
                self.redis.zadd(f"rate_limit:{key}", {str(current_time): current_time})
                self.redis.expire(f"rate_limit:{key}", window)
                
                return True, {
                    "allowed": True,
                    "requests": current_requests + 1,
                    "limit": limit,
                    "reset_time": current_time + window
                }
            else:
                # Rate limit exceeded
                oldest_request = self.redis.zrange(f"rate_limit:{key}", 0, 0, withscores=True)
                reset_time = int(oldest_request[0][1]) + window if oldest_request else current_time + window
                
                return False, {
                    "allowed": False,
                    "requests": current_requests,
                    "limit": limit,
                    "reset_time": reset_time
                }
                
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request if rate limiting fails
            return True, {"allowed": True, "error": str(e)}


# Global rate limiter
rate_limiter = RateLimiter(redis_client)


def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    # Check for forwarded headers (behind proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"


def rate_limit(requests_per_minute: int = 60):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = get_client_ip(request)
            
            allowed, info = rate_limiter.is_allowed(
                key=f"ip:{client_ip}",
                limit=requests_per_minute,
                window=60
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Try again in {info['reset_time'] - int(time.time())} seconds",
                    headers={
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": str(max(0, info["limit"] - info["requests"])),
                        "X-RateLimit-Reset": str(info["reset_time"])
                    }
                )
            
            # Add rate limit headers to response
            response = await func(request, *args, **kwargs)
            if hasattr(response, 'headers'):
                response.headers["X-RateLimit-Limit"] = str(info["limit"])
                response.headers["X-RateLimit-Remaining"] = str(max(0, info["limit"] - info["requests"]))
                response.headers["X-RateLimit-Reset"] = str(info["reset_time"])
            
            return response
        return wrapper
    return decorator


async def verify_token_dependency(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """FastAPI dependency for token verification"""
    token = credentials.credentials
    payload = security_manager.verify_token(token)
    return payload


async def verify_api_key_dependency(request: Request):
    """FastAPI dependency for API key verification"""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    user_type = security_manager.verify_api_key(api_key)
    if not user_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return {"user_type": user_type, "api_key": api_key}


async def require_admin(auth_data: dict = Depends(verify_api_key_dependency)):
    """Require admin privileges"""
    if auth_data.get("user_type") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return auth_data


def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for logging/storage"""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class SecurityHeaders:
    """Security headers middleware"""
    
    @staticmethod
    def add_security_headers(response):
        """Add security headers to response"""
        response.headers.update({
            # Prevent XSS attacks
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            
            # HTTPS enforcement
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdnjs.cloudflare.com; "
                "img-src 'self' data: https:; "
                "font-src 'self' https://cdnjs.cloudflare.com"
            ),
            
            # Prevent information leakage
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        })
        
        return response


# Input validation helpers
def validate_input_length(data: str, max_length: int = 1000) -> str:
    """Validate input string length"""
    if len(data) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input too long. Maximum {max_length} characters allowed."
        )
    return data


def sanitize_input(data: str) -> str:
    """Basic input sanitization"""
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', '\x00']
    for char in dangerous_chars:
        data = data.replace(char, '')
    
    return data.strip()


# Audit logging
class AuditLogger:
    """Security audit logging"""
    
    def __init__(self):
        self.logger = logging.getLogger("security_audit")
        
        # Configure audit logger
        handler = logging.FileHandler("logs/security_audit.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_auth_attempt(self, request: Request, success: bool, user_info: str = "unknown"):
        """Log authentication attempt"""
        client_ip = get_client_ip(request)
        self.logger.info(
            f"AUTH_ATTEMPT - IP: {client_ip}, Success: {success}, User: {hash_sensitive_data(user_info)}, "
            f"User-Agent: {request.headers.get('User-Agent', 'unknown')}"
        )
    
    def log_api_access(self, request: Request, endpoint: str, user_type: str = "unknown"):
        """Log API access"""
        client_ip = get_client_ip(request)
        self.logger.info(
            f"API_ACCESS - IP: {client_ip}, Endpoint: {endpoint}, UserType: {user_type}, "
            f"Method: {request.method}"
        )
    
    def log_security_event(self, event_type: str, details: str, request: Request = None):
        """Log security events"""
        client_ip = get_client_ip(request) if request else "internal"
        self.logger.warning(
            f"SECURITY_EVENT - Type: {event_type}, IP: {client_ip}, Details: {details}"
        )


# Global audit logger
audit_logger = AuditLogger()


# Error handling for security
class SecurityError(Exception):
    """Custom security exception"""
    def __init__(self, message: str, error_code: str = "SECURITY_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


def handle_security_error(error: SecurityError, request: Request):
    """Handle security errors with proper logging"""
    audit_logger.log_security_event(
        event_type=error.error_code,
        details=error.message,
        request=request
    )
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Security violation detected"
    )