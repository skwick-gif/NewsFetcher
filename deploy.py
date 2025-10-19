#!/usr/bin/env python3
"""
Production Deployment Script for Tariff Radar System
Handles Docker Compose deployment, initialization, and monitoring
"""

import subprocess
import sys
import os
import time
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional


class DeploymentManager:
    """Manages production deployment of the Tariff Radar system"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.compose_file = self.project_root / "docker-compose.yml"
        self.env_file = self.project_root / ".env"
        
    def check_prerequisites(self) -> bool:
        """Check if Docker and Docker Compose are available"""
        try:
            # Check Docker
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Docker found: {result.stdout.strip()}")
            
            # Check Docker Compose
            result = subprocess.run(["docker", "compose", "version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Docker Compose found: {result.stdout.strip()}")
            
            # Check if Docker daemon is running
            result = subprocess.run(["docker", "info"], 
                                  capture_output=True, text=True, check=True)
            print("✅ Docker daemon is running")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker check failed: {e}")
            return False
        except FileNotFoundError:
            print("❌ Docker or Docker Compose not found in PATH")
            return False
    
    def create_env_file(self) -> bool:
        """Create or update .env file with necessary configuration"""
        try:
            env_content = """# Tariff Radar Environment Configuration
# Database
POSTGRES_DB=tariff_radar
POSTGRES_USER=tariff_user
POSTGRES_PASSWORD=secure_password_123
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql://tariff_user:secure_password_123@postgres:5432/tariff_radar

# Redis
REDIS_URL=redis://redis:6379/0

# Qdrant Vector DB
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_API_KEY=

# FastAPI
SECRET_KEY=your-super-secret-key-change-in-production
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# Notification Settings
# WeChat Work (WeCom) - Enterprise WeChat
WECOM_WEBHOOK_URL=
WECOM_KEY=

# Email Notifications
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=
EMAIL_PASSWORD=
EMAIL_USE_TLS=true
EMAIL_FROM=tariff-radar@yourcompany.com
EMAIL_TO=alerts@yourcompany.com

# Telegram Bot
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# OpenAI/Anthropic for LLM Analysis  
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
LLM_PROVIDER=openai

# RSS and Scraping
USER_AGENT=TariffRadar/1.0 (+https://yourcompany.com/contact)
REQUEST_TIMEOUT=30
MAX_RETRIES=3

# Celery Configuration
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Monitoring and Logging
LOG_LEVEL=INFO
SENTRY_DSN=

# Application Settings
MAX_ARTICLES_PER_RUN=100
DUPLICATE_THRESHOLD=0.85
KEYWORD_SCORE_THRESHOLD=0.3
ML_SCORE_THRESHOLD=0.6
NOTIFICATION_SCORE_THRESHOLD=0.8

# Production Settings
WORKERS=4
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=50
TIMEOUT=120
"""
            
            with open(self.env_file, 'w') as f:
                f.write(env_content)
            
            print(f"✅ Created {self.env_file}")
            print("⚠️  Please update the .env file with your actual credentials!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    
    def build_images(self) -> bool:
        """Build Docker images"""
        try:
            print("🔨 Building Docker images...")
            
            cmd = ["docker", "compose", "build", "--no-cache"]
            result = subprocess.run(cmd, cwd=self.project_root, check=True)
            
            print("✅ Docker images built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker build failed: {e}")
            return False
    
    def start_services(self) -> bool:
        """Start all services using Docker Compose"""
        try:
            print("🚀 Starting services...")
            
            # Start infrastructure services first
            infrastructure_services = ["postgres", "redis", "qdrant"]
            cmd = ["docker", "compose", "up", "-d"] + infrastructure_services
            subprocess.run(cmd, cwd=self.project_root, check=True)
            
            print("⏳ Waiting for infrastructure services to be ready...")
            time.sleep(10)
            
            # Initialize database
            if not self.init_database():
                return False
            
            # Start application services
            app_services = ["app", "worker", "scheduler"]
            cmd = ["docker", "compose", "up", "-d"] + app_services
            subprocess.run(cmd, cwd=self.project_root, check=True)
            
            print("✅ All services started successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start services: {e}")
            return False
    
    def init_database(self) -> bool:
        """Initialize database tables"""
        try:
            print("🗄️ Initializing database...")
            
            cmd = [
                "docker", "compose", "exec", "-T", "postgres", 
                "psql", "-U", "tariff_user", "-d", "tariff_radar",
                "-c", "SELECT 1;"  # Simple connectivity test
            ]
            
            # Wait for PostgreSQL to be ready
            max_retries = 30
            for attempt in range(max_retries):
                try:
                    subprocess.run(cmd, cwd=self.project_root, 
                                 check=True, capture_output=True)
                    break
                except subprocess.CalledProcessError:
                    if attempt < max_retries - 1:
                        print(f"⏳ Waiting for PostgreSQL... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2)
                    else:
                        raise
            
            # Run database initialization
            init_cmd = [
                "docker", "compose", "run", "--rm", "app",
                "python", "-c", 
                "from storage.database import init_db; init_db(); print('Database initialized')"
            ]
            subprocess.run(init_cmd, cwd=self.project_root, check=True)
            
            print("✅ Database initialized successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Database initialization failed: {e}")
            return False
    
    def check_health(self) -> Dict[str, Any]:
        """Check health of all services"""
        health_status = {
            "services": {},
            "overall": False
        }
        
        try:
            # Check Docker Compose services
            result = subprocess.run(
                ["docker", "compose", "ps", "--format", "json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            services = json.loads(result.stdout) if result.stdout.strip() else []
            
            for service in services:
                service_name = service.get("Service", "unknown")
                status = service.get("State", "unknown")
                health = service.get("Health", "unknown")
                
                health_status["services"][service_name] = {
                    "status": status,
                    "health": health,
                    "running": status == "running"
                }
            
            # Check application endpoints
            try:
                response = requests.get("http://localhost:8000/api/system/health", timeout=5)
                if response.status_code == 200:
                    health_status["api"] = response.json()
                else:
                    health_status["api"] = {"error": f"HTTP {response.status_code}"}
            except requests.RequestException as e:
                health_status["api"] = {"error": str(e)}
            
            # Overall health assessment
            running_services = sum(1 for s in health_status["services"].values() 
                                 if s.get("running", False))
            total_services = len(health_status["services"])
            
            health_status["overall"] = (
                running_services >= total_services * 0.8 and  # 80% services running
                health_status.get("api", {}).get("overall_healthy", False)
            )
            
            return health_status
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return health_status
    
    def show_logs(self, service: Optional[str] = None, follow: bool = False):
        """Show logs for services"""
        try:
            cmd = ["docker", "compose", "logs"]
            if follow:
                cmd.append("-f")
            if service:
                cmd.append(service)
            
            subprocess.run(cmd, cwd=self.project_root)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to show logs: {e}")
    
    def stop_services(self):
        """Stop all services"""
        try:
            print("🛑 Stopping services...")
            cmd = ["docker", "compose", "down"]
            subprocess.run(cmd, cwd=self.project_root, check=True)
            print("✅ Services stopped")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to stop services: {e}")
    
    def deploy(self, skip_build: bool = False) -> bool:
        """Full deployment process"""
        print("🎯 Starting Tariff Radar deployment...")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Create environment file
        if not self.env_file.exists():
            if not self.create_env_file():
                return False
        
        # Build images (unless skipped)
        if not skip_build:
            if not self.build_images():
                return False
        
        # Start services
        if not self.start_services():
            return False
        
        # Wait a bit for services to stabilize
        print("⏳ Waiting for services to stabilize...")
        time.sleep(15)
        
        # Health check
        health = self.check_health()
        
        print("\n" + "=" * 50)
        print("🏥 DEPLOYMENT HEALTH CHECK")
        print("=" * 50)
        
        for service_name, service_info in health.get("services", {}).items():
            status_icon = "✅" if service_info.get("running") else "❌"
            print(f"{status_icon} {service_name}: {service_info.get('status', 'unknown')}")
        
        if health.get("api"):
            api_icon = "✅" if not health["api"].get("error") else "❌"
            print(f"{api_icon} API Health: {health['api']}")
        
        overall_icon = "✅" if health["overall"] else "❌"
        print(f"\n{overall_icon} Overall Status: {'HEALTHY' if health['overall'] else 'ISSUES DETECTED'}")
        
        if health["overall"]:
            print("\n🎉 Deployment successful!")
            print("🌐 Application available at: http://localhost:8000")
            print("📊 Dashboard: http://localhost:8000/")
            print("📋 API Documentation: http://localhost:8000/docs")
            
            print("\nUseful commands:")
            print("• View logs: python deploy.py logs")
            print("• Health check: python deploy.py health")
            print("• Stop services: python deploy.py stop")
            
            return True
        else:
            print("\n⚠️  Deployment completed with issues. Check logs for details.")
            return False


def main():
    """Main deployment script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tariff Radar Deployment Manager")
    parser.add_argument("action", choices=["deploy", "build", "start", "stop", "health", "logs", "init-env"],
                       help="Action to perform")
    parser.add_argument("--skip-build", action="store_true", 
                       help="Skip Docker image building")
    parser.add_argument("--service", help="Specific service for logs")
    parser.add_argument("--follow", "-f", action="store_true", 
                       help="Follow logs output")
    
    args = parser.parse_args()
    
    deployer = DeploymentManager()
    
    try:
        if args.action == "deploy":
            success = deployer.deploy(skip_build=args.skip_build)
            sys.exit(0 if success else 1)
            
        elif args.action == "build":
            success = deployer.build_images()
            sys.exit(0 if success else 1)
            
        elif args.action == "start":
            success = deployer.start_services()
            sys.exit(0 if success else 1)
            
        elif args.action == "stop":
            deployer.stop_services()
            
        elif args.action == "health":
            health = deployer.check_health()
            print(json.dumps(health, indent=2))
            sys.exit(0 if health["overall"] else 1)
            
        elif args.action == "logs":
            deployer.show_logs(service=args.service, follow=args.follow)
            
        elif args.action == "init-env":
            success = deployer.create_env_file()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()