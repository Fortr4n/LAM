"""
Base Plugin Class for LAM

All plugins must inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging


class BasePlugin(ABC):
    """Base class for all LAM plugins."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = True
        self.logger = logging.getLogger(f"plugin.{name}")
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this plugin provides."""
        pass
    
    @abstractmethod
    def execute(self, command: str, **kwargs) -> Dict[str, Any]:
        """Execute a plugin command. Return result dictionary."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "capabilities": self.get_capabilities()
        }
    
    def enable(self) -> None:
        """Enable the plugin."""
        self.enabled = True
        self.logger.info(f"Plugin {self.name} enabled")
    
    def disable(self) -> None:
        """Disable the plugin."""
        self.enabled = False
        self.logger.info(f"Plugin {self.name} disabled")
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self.logger.info(f"Plugin {self.name} cleaned up")
    
    def health_check(self) -> Dict[str, Any]:
        """Check plugin health status."""
        return {
            "name": self.name,
            "status": "healthy" if self.enabled else "disabled",
            "version": self.version
        }
