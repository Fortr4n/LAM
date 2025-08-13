"""
Configuration management for the LAM project.
Handles API keys, model settings, and environment configuration.
"""

import os
from typing import Optional, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration management for the LAM project."""
    
    # API Configuration
    DOORDASH_API_KEY: Optional[str] = os.getenv('DOORDASH_API_KEY')
    UBER_API_KEY: Optional[str] = os.getenv('UBER_API_KEY')
    EXPEDIA_API_KEY: Optional[str] = os.getenv('EXPEDIA_API_KEY')
    PLAID_CLIENT_ID: Optional[str] = os.getenv('PLAID_CLIENT_ID')
    PLAID_SECRET: Optional[str] = os.getenv('PLAID_SECRET')
    PLAID_ENVIRONMENT: str = os.getenv('PLAID_ENVIRONMENT', 'sandbox')
    
    # Model Configuration
    TEXT_MODEL_NAME: str = "bigscience/bloomz"
    SPEECH_MODEL_NAME: str = "openai/whisper-large-v3"
    EMOTION_MODEL_NAME: str = "arpanghoshal/EmoRoBERTa"
    
    # Audio Configuration
    DEFAULT_AUDIO_DURATION: int = 5
    DEFAULT_SAMPLE_RATE: int = 16000
    DEFAULT_CHANNELS: int = 1
    
    # Neural Network Configuration
    BINDSNET_INPUT_SIZE: int = 100
    BINDSNET_SIMULATION_TIME: int = 100
    
    # Task Management Configuration
    DEFAULT_TASK_PRIORITY: int = 2
    URGENCY_THRESHOLD_HOUR: int = 3600  # 1 hour in seconds
    URGENCY_THRESHOLD_DAY: int = 86400   # 1 day in seconds
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Validate that all required API keys are present."""
        validation = {
            'doordash': bool(cls.DOORDASH_API_KEY),
            'uber': bool(cls.UBER_API_KEY),
            'expedia': bool(cls.EXPEDIA_API_KEY),
            'plaid': bool(cls.PLAID_CLIENT_ID and cls.PLAID_SECRET)
        }
        return validation
    
    @classmethod
    def get_missing_api_keys(cls) -> list:
        """Get list of missing API keys."""
        validation = cls.validate_api_keys()
        return [service for service, valid in validation.items() if not valid]
    
    @classmethod
    def print_config_status(cls):
        """Print current configuration status."""
        print("=== LAM Configuration Status ===")
        print(f"Text Model: {cls.TEXT_MODEL_NAME}")
        print(f"Speech Model: {cls.SPEECH_MODEL_NAME}")
        print(f"Emotion Model: {cls.EMOTION_MODEL_NAME}")
        print(f"Audio Duration: {cls.DEFAULT_AUDIO_DURATION}s")
        print(f"Sample Rate: {cls.DEFAULT_SAMPLE_RATE}Hz")
        print()
        
        print("API Key Status:")
        validation = cls.validate_api_keys()
        for service, valid in validation.items():
            status = "✓" if valid else "✗"
            print(f"  {service.title()}: {status}")
        
        missing = cls.get_missing_api_keys()
        if missing:
            print(f"\nMissing API keys: {', '.join(missing)}")
            print("Please set the required environment variables.")
        else:
            print("\nAll API keys are configured!")
