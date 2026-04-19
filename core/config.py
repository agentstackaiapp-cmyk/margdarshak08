"""
Application Configuration Module
FAANG-Level: Centralized configuration with validation
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / '.env')


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # App Info
    APP_NAME: str = "Margdarshak"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Database
    MONGO_URL: str = Field(..., env="MONGO_URL")
    DB_NAME: str = Field(..., env="DB_NAME")
    
    # LLM
    EMERGENT_LLM_KEY: str = Field(..., env="EMERGENT_LLM_KEY")
    LLM_MODEL: str = Field(default="gpt-4o", env="LLM_MODEL")
    LLM_MAX_TOKENS: int = Field(default=2000, env="LLM_MAX_TOKENS")
    LLM_TEMPERATURE: float = Field(default=0.7, env="LLM_TEMPERATURE")
    
    # Auth
    SESSION_EXPIRE_DAYS: int = Field(default=7, env="SESSION_EXPIRE_DAYS")
    EMERGENT_AUTH_URL: str = Field(
        default="https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
        env="EMERGENT_AUTH_URL"
    )
    
    # Security
    CORS_ORIGINS: List[str] = ["*"]
    MAX_REQUEST_SIZE: int = Field(default=1024 * 1024, env="MAX_REQUEST_SIZE")  # 1MB
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Performance
    CACHE_TTL_SECONDS: int = Field(default=300, env="CACHE_TTL_SECONDS")
    DB_MAX_POOL_SIZE: int = Field(default=10, env="DB_MAX_POOL_SIZE")
    DB_MIN_POOL_SIZE: int = Field(default=1, env="DB_MIN_POOL_SIZE")
    
    @validator('MONGO_URL')
    def validate_mongo_url(cls, v):
        if not v or not v.startswith('mongodb'):
            raise ValueError('MONGO_URL must be a valid MongoDB connection string')
        return v
    
    @validator('EMERGENT_LLM_KEY')
    def validate_llm_key(cls, v):
        if not v or not v.startswith('sk-emergent-'):
            raise ValueError('EMERGENT_LLM_KEY must be a valid Emergent key')
        return v
    
    class Config:
        env_file = '.env'
        case_sensitive = True


# Singleton instance
settings = Settings()
