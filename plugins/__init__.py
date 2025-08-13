"""
LAM Plugin System

This package provides a plugin architecture for extending LAM functionality.
Plugins can add new API integrations, AI capabilities, and features.
"""

from .base import BasePlugin
from .manager import PluginManager

__all__ = ['BasePlugin', 'PluginManager']
