"""
Application configuration for Flute Helper.

This module loads configuration from environment variables and provides
a singleton Settings instance used throughout the application.

Environment Variables:
    OPENAI_API_KEY: API key for OpenAI services (required for AI features)

Usage:
    from app.config import settings
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class Settings:
    """
    Application settings loaded from environment variables.

    Attributes:
        OPENAI_API_KEY: API key for OpenAI API access
        OPENAI_MODEL: Model to use for vision and reasoning tasks
    """

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-5.2"  # Vision-capable model with reasoning


# Singleton instance - import this in other modules
settings = Settings()
