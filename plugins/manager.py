"""
Plugin Manager for LAM

Manages the loading, initialization, and execution of plugins.
"""

import os
import sys
import importlib
import logging
from typing import Dict, Any, List, Optional, Type
from pathlib import Path

from .base import BasePlugin


class PluginManager:
    """Manages LAM plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.logger = logging.getLogger("plugin_manager")
        self.plugins_dir = Path(__file__).parent / "installed"
        
        # Create plugins directory if it doesn't exist
        self.plugins_dir.mkdir(exist_ok=True)
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in the plugins directory."""
        discovered = []
        
        if not self.plugins_dir.exists():
            return discovered
        
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            
            plugin_name = plugin_file.stem
            discovered.append(plugin_name)
            self.logger.info(f"Discovered plugin: {plugin_name}")
        
        return discovered
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin by name."""
        try:
            # Import the plugin module
            plugin_path = self.plugins_dir / f"{plugin_name}.py"
            if not plugin_path.exists():
                self.logger.error(f"Plugin file not found: {plugin_path}")
                return False
            
            # Add plugins directory to path temporarily
            sys.path.insert(0, str(self.plugins_dir))
            
            try:
                module = importlib.import_module(plugin_name)
            finally:
                # Remove from path
                sys.path.pop(0)
            
            # Find plugin class (should inherit from BasePlugin)
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePlugin) and 
                    attr != BasePlugin):
                    plugin_class = attr
                    break
            
            if not plugin_class:
                self.logger.error(f"No plugin class found in {plugin_name}")
                return False
            
            # Create plugin instance
            plugin_instance = plugin_class()
            
            # Initialize plugin
            if not plugin_instance.initialize():
                self.logger.error(f"Failed to initialize plugin: {plugin_name}")
                return False
            
            # Register plugin
            self.plugins[plugin_name] = plugin_instance
            self.logger.info(f"Plugin loaded successfully: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    def load_all_plugins(self) -> int:
        """Load all discovered plugins."""
        discovered = self.discover_plugins()
        loaded_count = 0
        
        for plugin_name in discovered:
            if self.load_plugin(plugin_name):
                loaded_count += 1
        
        self.logger.info(f"Loaded {loaded_count}/{len(discovered)} plugins")
        return loaded_count
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins with their information."""
        return [plugin.get_info() for plugin in self.plugins.values()]
    
    def execute_plugin_command(self, plugin_name: str, command: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute a command on a specific plugin."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            self.logger.error(f"Plugin not found: {plugin_name}")
            return None
        
        if not plugin.enabled:
            self.logger.warning(f"Plugin {plugin_name} is disabled")
            return None
        
        try:
            result = plugin.execute(command, **kwargs)
            self.logger.info(f"Executed command '{command}' on plugin {plugin_name}")
            return result
        except Exception as e:
            self.logger.error(f"Error executing command on plugin {plugin_name}: {e}")
            return {"error": str(e)}
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return False
        
        plugin.enable()
        return True
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return False
        
        plugin.disable()
        return True
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return False
        
        try:
            plugin.cleanup()
            del self.plugins[plugin_name]
            self.logger.info(f"Plugin unloaded: {plugin_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def get_plugin_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all plugins."""
        capabilities = {}
        for name, plugin in self.plugins.items():
            if plugin.enabled:
                capabilities[name] = plugin.get_capabilities()
        return capabilities
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all plugins."""
        health_status = {
            "total_plugins": len(self.plugins),
            "enabled_plugins": len([p for p in self.plugins.values() if p.enabled]),
            "plugins": {}
        }
        
        for name, plugin in self.plugins.items():
            health_status["plugins"][name] = plugin.health_check()
        
        return health_status
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin."""
        if self.unload_plugin(plugin_name):
            return self.load_plugin(plugin_name)
        return False
    
    def reload_all_plugins(self) -> int:
        """Reload all plugins."""
        plugin_names = list(self.plugins.keys())
        reloaded_count = 0
        
        for plugin_name in plugin_names:
            if self.reload_plugin(plugin_name):
                reloaded_count += 1
        
        self.logger.info(f"Reloaded {reloaded_count}/{len(plugin_names)} plugins")
        return reloaded_count
