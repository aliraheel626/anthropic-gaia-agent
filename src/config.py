"""Configuration management for GAIA Agent."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """LLM model configuration."""
    
    name: str = "claude-sonnet-4-20250514"
    max_turns: int = 50
    max_tokens: int = 8192
    temperature: float = 0.0  # Deterministic for benchmark reproducibility


@dataclass
class SandboxConfig:
    """E2B sandbox configuration."""
    
    timeout: int = 300  # seconds
    memory_mb: int = 2048
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class RAGConfig:
    """RAG system configuration."""
    
    persist_dir: Path = field(default_factory=lambda: Path("./data/chroma"))
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5


@dataclass
class LogConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    file: Optional[Path] = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Config:
    """Main configuration container."""
    
    # API Keys
    anthropic_api_key: str = ""
    e2b_api_key: str = ""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    log: LogConfig = field(default_factory=LogConfig)
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    gaia_dir: Path = field(default_factory=lambda: Path("./data/gaia"))
    
    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            e2b_api_key=os.getenv("E2B_API_KEY", ""),
            model=ModelConfig(
                name=os.getenv("MODEL_NAME", "claude-sonnet-4-20250514"),
                max_turns=int(os.getenv("MAX_TURNS", "50")),
                max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
            ),
            sandbox=SandboxConfig(
                timeout=int(os.getenv("SANDBOX_TIMEOUT", "300")),
                memory_mb=int(os.getenv("SANDBOX_MEMORY_MB", "2048")),
            ),
            rag=RAGConfig(
                persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")),
                embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            ),
            log=LogConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                file=Path(os.getenv("LOG_FILE")) if os.getenv("LOG_FILE") else None,
            ),
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
            gaia_dir=Path(os.getenv("GAIA_DIR", "./data/gaia")),
        )
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")
        
        if not self.e2b_api_key:
            errors.append("E2B_API_KEY is required")
        
        return errors


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
