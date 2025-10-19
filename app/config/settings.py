"""
Production Configuration Management
Environment-based configuration with validation and secrets management
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from functools import lru_cache
import secrets


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False


@dataclass
class RedisConfig:
    """Redis configuration settings"""
    url: str
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str
    api_keys: Dict[str, str] = field(default_factory=dict)
    admin_users: list = field(default_factory=list)
    jwt_expire_minutes: int = 30
    rate_limit_per_minute: int = 60
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    allowed_hosts: list = field(default_factory=lambda: ["localhost", "127.0.0.1"])


@dataclass
class NotificationConfig:
    """Notification channels configuration"""
    wecom_webhook_url: Optional[str] = None
    wecom_key: Optional[str] = None
    
    email_host: str = "smtp.gmail.com"
    email_port: int = 587
    email_user: Optional[str] = None
    email_password: Optional[str] = None
    email_use_tls: bool = True
    email_from: Optional[str] = None
    email_to: list = field(default_factory=list)
    
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None


@dataclass
class AIConfig:
    """AI/LLM service configuration"""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    llm_provider: str = "openai"
    max_tokens: int = 1000
    temperature: float = 0.3
    timeout: int = 30


@dataclass
class ProcessingConfig:
    """Article processing configuration"""
    max_articles_per_run: int = 100
    duplicate_threshold: float = 0.85
    keyword_score_threshold: float = 0.3
    ml_score_threshold: float = 0.6
    notification_score_threshold: float = 0.8
    content_max_length: int = 10000
    user_agent: str = "TariffRadar/1.0"
    request_timeout: int = 30
    max_retries: int = 3


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    sentry_dsn: Optional[str] = None
    metrics_enabled: bool = True
    health_check_interval: int = 300  # 5 minutes


@dataclass
class AppConfig:
    """Main application configuration"""
    environment: str = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    
    database: DatabaseConfig = None
    redis: RedisConfig = None
    security: SecurityConfig = None
    notifications: NotificationConfig = None
    ai: AIConfig = None
    processing: ProcessingConfig = None
    monitoring: MonitoringConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided"""
        if self.database is None:
            self.database = DatabaseConfig(url=os.getenv("DATABASE_URL", ""))
        
        if self.redis is None:
            self.redis = RedisConfig(url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        
        if self.security is None:
            self.security = SecurityConfig(
                secret_key=os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
            )
        
        if self.notifications is None:
            self.notifications = NotificationConfig()
        
        if self.ai is None:
            self.ai = AIConfig()
        
        if self.processing is None:
            self.processing = ProcessingConfig()
        
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()


class ConfigManager:
    """Configuration manager with environment-based loading"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("config")
        self.config_path.mkdir(exist_ok=True)
        
        # Environment detection
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Load configuration
        self._config = self._load_config()
        self._validate_config()
        self._setup_logging()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from files and environment"""
        
        # Start with default configuration
        config_data = {
            "environment": self.environment,
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "workers": int(os.getenv("WORKERS", "4"))
        }
        
        # Load from YAML files (base + environment-specific)
        base_config_file = self.config_path / "config.yaml"
        env_config_file = self.config_path / f"config.{self.environment}.yaml"
        
        for config_file in [base_config_file, env_config_file]:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        file_config = yaml.safe_load(f) or {}
                        config_data.update(file_config)
                except Exception as e:
                    logging.warning(f"Failed to load config file {config_file}: {e}")
        
        # Override with environment variables
        config_data.update(self._load_from_environment())
        
        # Create configuration objects
        return AppConfig(
            environment=config_data.get("environment", "development"),
            debug=config_data.get("debug", False),
            host=config_data.get("host", "0.0.0.0"),
            port=config_data.get("port", 8000),
            workers=config_data.get("workers", 4),
            reload=config_data.get("reload", False),
            
            database=self._create_database_config(config_data),
            redis=self._create_redis_config(config_data),
            security=self._create_security_config(config_data),
            notifications=self._create_notification_config(config_data),
            ai=self._create_ai_config(config_data),
            processing=self._create_processing_config(config_data),
            monitoring=self._create_monitoring_config(config_data)
        )
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_mapping = {
            # App settings
            "ENVIRONMENT": "environment",
            "DEBUG": ("debug", lambda x: x.lower() == "true"),
            "HOST": "host",
            "PORT": ("port", int),
            "WORKERS": ("workers", int),
            
            # Database
            "DATABASE_URL": "database_url",
            "DB_POOL_SIZE": ("db_pool_size", int),
            "DB_MAX_OVERFLOW": ("db_max_overflow", int),
            
            # Redis
            "REDIS_URL": "redis_url",
            "REDIS_MAX_CONNECTIONS": ("redis_max_connections", int),
            
            # Security
            "SECRET_KEY": "secret_key",
            "JWT_EXPIRE_MINUTES": ("jwt_expire_minutes", int),
            "RATE_LIMIT_PER_MINUTE": ("rate_limit_per_minute", int),
            "ADMIN_USERS": ("admin_users", lambda x: x.split(",")),
            
            # Notifications
            "WECOM_WEBHOOK_URL": "wecom_webhook_url",
            "EMAIL_HOST": "email_host",
            "EMAIL_PORT": ("email_port", int),
            "EMAIL_USER": "email_user",
            "EMAIL_PASSWORD": "email_password",
            "EMAIL_FROM": "email_from",
            "EMAIL_TO": ("email_to", lambda x: x.split(",")),
            "TELEGRAM_BOT_TOKEN": "telegram_bot_token",
            "TELEGRAM_CHAT_ID": "telegram_chat_id",
            
            # AI
            "OPENAI_API_KEY": "openai_api_key",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "PERPLEXITY_API_KEY": "perplexity_api_key",
            "LLM_PROVIDER": "llm_provider",
            
            # Processing
            "MAX_ARTICLES_PER_RUN": ("max_articles_per_run", int),
            "DUPLICATE_THRESHOLD": ("duplicate_threshold", float),
            "KEYWORD_SCORE_THRESHOLD": ("keyword_score_threshold", float),
            "NOTIFICATION_SCORE_THRESHOLD": ("notification_score_threshold", float),
            
            # Monitoring
            "LOG_LEVEL": "log_level",
            "SENTRY_DSN": "sentry_dsn"
        }
        
        config = {}
        for env_key, config_key in env_mapping.items():
            value = os.getenv(env_key)
            if value is not None:
                if isinstance(config_key, tuple):
                    config_key, converter = config_key
                    try:
                        value = converter(value)
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Failed to convert {env_key}={value}: {e}")
                        continue
                
                config[config_key] = value
        
        return config
    
    def _create_database_config(self, config_data: Dict) -> DatabaseConfig:
        """Create database configuration"""
        return DatabaseConfig(
            url=config_data.get("database_url", os.getenv("DATABASE_URL", "")),
            pool_size=config_data.get("db_pool_size", 10),
            max_overflow=config_data.get("db_max_overflow", 20),
            echo=config_data.get("debug", False)
        )
    
    def _create_redis_config(self, config_data: Dict) -> RedisConfig:
        """Create Redis configuration"""
        return RedisConfig(
            url=config_data.get("redis_url", os.getenv("REDIS_URL", "redis://localhost:6379/0")),
            max_connections=config_data.get("redis_max_connections", 50)
        )
    
    def _create_security_config(self, config_data: Dict) -> SecurityConfig:
        """Create security configuration"""
        # Load API keys
        api_keys = {}
        for key, value in os.environ.items():
            if key.startswith("API_KEY_"):
                name = key.replace("API_KEY_", "").lower()
                api_keys[value] = name
        
        return SecurityConfig(
            secret_key=config_data.get("secret_key", os.getenv("SECRET_KEY", secrets.token_urlsafe(32))),
            api_keys=api_keys,
            admin_users=config_data.get("admin_users", []),
            jwt_expire_minutes=config_data.get("jwt_expire_minutes", 30),
            rate_limit_per_minute=config_data.get("rate_limit_per_minute", 60),
            allowed_hosts=config_data.get("allowed_hosts", ["localhost", "127.0.0.1"])
        )
    
    def _create_notification_config(self, config_data: Dict) -> NotificationConfig:
        """Create notification configuration"""
        return NotificationConfig(
            wecom_webhook_url=config_data.get("wecom_webhook_url"),
            email_host=config_data.get("email_host", "smtp.gmail.com"),
            email_port=config_data.get("email_port", 587),
            email_user=config_data.get("email_user"),
            email_password=config_data.get("email_password"),
            email_from=config_data.get("email_from"),
            email_to=config_data.get("email_to", []),
            telegram_bot_token=config_data.get("telegram_bot_token"),
            telegram_chat_id=config_data.get("telegram_chat_id")
        )
    
    def _create_ai_config(self, config_data: Dict) -> AIConfig:
        """Create AI configuration"""
        return AIConfig(
            openai_api_key=config_data.get("openai_api_key"),
            anthropic_api_key=config_data.get("anthropic_api_key"),
            perplexity_api_key=config_data.get("perplexity_api_key"),
            llm_provider=config_data.get("llm_provider", "openai")
        )
    
    def _create_processing_config(self, config_data: Dict) -> ProcessingConfig:
        """Create processing configuration"""
        return ProcessingConfig(
            max_articles_per_run=config_data.get("max_articles_per_run", 100),
            duplicate_threshold=config_data.get("duplicate_threshold", 0.85),
            keyword_score_threshold=config_data.get("keyword_score_threshold", 0.3),
            notification_score_threshold=config_data.get("notification_score_threshold", 0.8)
        )
    
    def _create_monitoring_config(self, config_data: Dict) -> MonitoringConfig:
        """Create monitoring configuration"""
        return MonitoringConfig(
            log_level=config_data.get("log_level", "INFO"),
            sentry_dsn=config_data.get("sentry_dsn")
        )
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Required settings
        if not self._config.database.url:
            errors.append("DATABASE_URL is required")
        
        if not self._config.security.secret_key:
            errors.append("SECRET_KEY is required")
        
        # Validate thresholds
        if not 0 <= self._config.processing.duplicate_threshold <= 1:
            errors.append("DUPLICATE_THRESHOLD must be between 0 and 1")
        
        if not 0 <= self._config.processing.keyword_score_threshold <= 1:
            errors.append("KEYWORD_SCORE_THRESHOLD must be between 0 and 1")
        
        # Environment-specific validations
        if self._config.environment == "production":
            if self._config.debug:
                errors.append("DEBUG should be False in production")
            
            if self._config.security.secret_key == "change-me":
                errors.append("SECRET_KEY must be changed in production")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self._config.monitoring.log_level.upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self._config.monitoring.log_file or "logs/app.log")
            ]
        )
        
        # Setup Sentry if configured
        if self._config.monitoring.sentry_dsn:
            try:
                import sentry_sdk
                from sentry_sdk.integrations.logging import LoggingIntegration
                
                sentry_logging = LoggingIntegration(
                    level=logging.INFO,
                    event_level=logging.ERROR
                )
                
                sentry_sdk.init(
                    dsn=self._config.monitoring.sentry_dsn,
                    integrations=[sentry_logging],
                    environment=self._config.environment,
                    traces_sample_rate=0.1 if self._config.environment == "production" else 1.0
                )
                
                logging.info("Sentry integration enabled")
            except ImportError:
                logging.warning("Sentry SDK not installed, skipping integration")
    
    @property
    def config(self) -> AppConfig:
        """Get the current configuration"""
        return self._config
    
    def get_database_url(self) -> str:
        """Get database URL with connection parameters"""
        return self._config.database.url
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self._config.environment == "production"
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled"""
        return self._config.debug
    
    def create_config_template(self):
        """Create configuration file templates"""
        
        # Base configuration template
        base_config = {
            "app": {
                "name": "Tariff Radar",
                "version": "1.0.0",
                "description": "US-China Trade Monitoring System"
            },
            "database": {
                "pool_size": 10,
                "max_overflow": 20,
                "echo": False
            },
            "redis": {
                "max_connections": 50,
                "socket_timeout": 5
            },
            "processing": {
                "max_articles_per_run": 100,
                "duplicate_threshold": 0.85,
                "keyword_score_threshold": 0.3
            },
            "monitoring": {
                "log_level": "INFO",
                "metrics_enabled": True
            }
        }
        
        # Production overrides
        production_config = {
            "debug": False,
            "database": {
                "pool_size": 20,
                "max_overflow": 40,
                "echo": False
            },
            "monitoring": {
                "log_level": "WARNING"
            }
        }
        
        # Development overrides
        development_config = {
            "debug": True,
            "reload": True,
            "database": {
                "echo": True
            },
            "monitoring": {
                "log_level": "DEBUG"
            }
        }
        
        # Write configuration files
        configs = {
            "config.yaml": base_config,
            "config.production.yaml": production_config,
            "config.development.yaml": development_config
        }
        
        for filename, config in configs.items():
            config_file = self.config_path / filename
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logging.info(f"Created configuration template: {config_file}")


# Global configuration instance
@lru_cache()
def get_config() -> ConfigManager:
    """Get singleton configuration manager"""
    return ConfigManager()


# Convenience functions
def get_database_url() -> str:
    """Get database connection URL"""
    return get_config().get_database_url()


def is_production() -> bool:
    """Check if running in production environment"""
    return get_config().is_production()


def is_debug() -> bool:
    """Check if debug mode is enabled"""
    return get_config().is_debug()


if __name__ == "__main__":
    # Create configuration templates
    config_manager = ConfigManager()
    config_manager.create_config_template()
    
    print("Configuration templates created:")
    print("- config/config.yaml (base configuration)")
    print("- config/config.production.yaml (production overrides)")
    print("- config/config.development.yaml (development overrides)")
    print("\nEdit these files to customize your deployment.")