"""
Secrets management abstraction.
Supports multiple backends: environment variables, file-based, and cloud providers.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
from pathlib import Path

class SecretsManager(ABC):
    """Abstract base class for secrets management."""
    
    @abstractmethod
    def get_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret by name."""
        pass
    
    @abstractmethod
    def get_all_secrets(self) -> Dict[str, str]:
        """Retrieve all available secrets."""
        pass

class EnvSecretsManager(SecretsManager):
    """Environment variable-based secrets manager."""
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
    
    def get_secret(self, name: str) -> Optional[str]:
        """Get secret from environment variable."""
        full_name = f"{self.prefix}{name}" if self.prefix else name
        return os.getenv(full_name)
    
    def get_all_secrets(self) -> Dict[str, str]:
        """Get all secrets with the configured prefix."""
        secrets = {}
        for key, value in os.environ.items():
            if self.prefix and key.startswith(self.prefix):
                secrets[key[len(self.prefix):]] = value
            elif not self.prefix and any(s in key.lower() for s in ['key', 'secret', 'password', 'token']):
                secrets[key] = value
        return secrets

class FileSecretsManager(SecretsManager):
    """File-based secrets manager (reads from .env or similar files)."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._secrets: Dict[str, str] = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """Load secrets from file."""
        if not self.file_path.exists():
            return
        
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    self._secrets[key.strip()] = value
    
    def get_secret(self, name: str) -> Optional[str]:
        """Get secret from loaded file."""
        return self._secrets.get(name)
    
    def get_all_secrets(self) -> Dict[str, str]:
        """Get all loaded secrets."""
        return self._secrets.copy()

class AWSSecretsManager(SecretsManager):
    """AWS Secrets Manager integration (placeholder)."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of AWS client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('secretsmanager', region_name=self.region)
            except ImportError:
                raise ImportError("boto3 required for AWS Secrets Manager")
        return self._client
    
    def get_secret(self, name: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        client = self._get_client()
        try:
            response = client.get_secret_value(SecretId=name)
            return response.get('SecretString')
        except Exception:
            return None
    
    def get_all_secrets(self) -> Dict[str, str]:
        """Get all secrets (not recommended for AWS - list specific secrets instead)."""
        # AWS doesn't support listing all secrets efficiently
        # This should be implemented based on your specific needs
        return {}

# Global secrets manager instance (configured via environment)
def get_secrets_manager() -> SecretsManager:
    """Factory function to get configured secrets manager."""
    backend = os.getenv("SECRETS_BACKEND", "env").lower()
    
    if backend == "aws":
        return AWSSecretsManager(region=os.getenv("AWS_REGION", "us-east-1"))
    elif backend == "file":
        file_path = os.getenv("SECRETS_FILE", ".env.secrets")
        return FileSecretsManager(file_path)
    else:
        # Default to environment variables
        prefix = os.getenv("SECRETS_PREFIX", "")
        return EnvSecretsManager(prefix=prefix)

# Convenience functions
_secrets_manager: Optional[SecretsManager] = None

def get_secret(name: str) -> Optional[str]:
    """Get a secret using the configured secrets manager."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = get_secrets_manager()
    return _secrets_manager.get_secret(name)

def get_database_url() -> str:
    """Get database URL from secrets."""
    url = get_secret("DATABASE_URL")
    if not url:
        # Fallback to individual components
        host = get_secret("DB_HOST", "localhost")
        port = get_secret("DB_PORT", "5432")
        name = get_secret("DB_NAME", "football_predictions")
        user = get_secret("DB_USER", "postgres")
        password = get_secret("DB_PASSWORD", "")
        url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
    return url

def get_secret_with_default(name: str, default: str) -> str:
    """Get a secret with a default fallback."""
    value = get_secret(name)
    return value if value is not None else default
