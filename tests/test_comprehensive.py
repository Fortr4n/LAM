#!/usr/bin/env python3
"""
Comprehensive tests for the LAM project.

Tests all major components including:
- Core LAM functionality
- API integrations
- Plugin system
- Database operations
- Web dashboard
"""

import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LAM import LAM
from agent import Agent
from config import Config
from database import DatabaseService, TaskRepository, UserRepository
from plugins.manager import PluginManager
from plugins.base import BasePlugin


class TestLAMCore(unittest.TestCase):
    """Test core LAM functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.lam = LAM("Test Agent")
    
    def test_lam_initialization(self):
        """Test LAM initialization."""
        self.assertEqual(self.lam.name, "Test Agent")
        self.assertIsInstance(self.lam.tasks, list)
        self.assertIsInstance(self.lam.context, dict)
    
    def test_task_management(self):
        """Test task management functionality."""
        # Test adding task
        success = self.lam.add_task("Test task", priority=1)
        self.assertTrue(success)
        self.assertEqual(len(self.lam.tasks), 1)
        
        # Test showing tasks
        self.lam.show_tasks()  # Should not raise exception
        
        # Test removing task
        success = self.lam.remove_task(1)
        self.assertTrue(success)
        self.assertEqual(len(self.lam.tasks), 0)
    
    def test_context_management(self):
        """Test context management."""
        self.lam.update_context(location="Test Location", time="Test Time")
        self.assertEqual(self.lam.context["location"], "Test Location")
        self.assertEqual(self.lam.context["time"], "Test Time")
    
    def test_ai_capabilities(self):
        """Test AI capabilities."""
        # Test response generation (may fail if models not available)
        response = self.lam.generate_response("Hello")
        self.assertIsInstance(response, str)
        
        # Test emotion recognition
        emotion = self.lam.recognize_emotion("I am happy!")
        self.assertIsInstance(emotion, str)
    
    def test_neural_network(self):
        """Test neural network functionality."""
        # Test BindsNET model creation
        self.assertIsNotNone(self.lam.network)
        
        # Test neural processing
        spike_count = self.lam.process_task_with_bindsnet("Test task")
        self.assertIsInstance(spike_count, int)
    
    def test_predictive_analytics(self):
        """Test predictive analytics."""
        historical_data = [(10, 50), (20, 100), (15, 75)]
        self.lam.predictive_analytics(historical_data)  # Should not raise exception


class TestAgent(unittest.TestCase):
    """Test Agent class functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.agent = Agent("Test Agent")
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertIsInstance(self.agent, Agent)
        self.assertIsInstance(self.agent, LAM)
        self.assertEqual(self.agent.name, "Test Agent")
    
    def test_api_integrations(self):
        """Test API integrations."""
        # Test that API objects are created
        self.assertIsNotNone(self.agent.doordash_api)
        self.assertIsNotNone(self.agent.uber_api)
        self.assertIsNotNone(self.agent.expedia_api)
        self.assertIsNotNone(self.agent.plaid_api)
    
    def test_food_ordering(self):
        """Test food ordering functionality."""
        # This will use mock data when no API key is available
        self.agent.order_food("Test Location")  # Should not raise exception
    
    def test_ride_request(self):
        """Test ride request functionality."""
        self.agent.request_ride(0, 0, 1, 1)  # Should not raise exception


class TestConfiguration(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_attributes(self):
        """Test configuration attributes."""
        self.assertIsNotNone(Config.TEXT_MODEL_NAME)
        self.assertIsNotNone(Config.SPEECH_MODEL_NAME)
        self.assertIsNotNone(Config.EMOTION_MODEL_NAME)
        self.assertIsInstance(Config.DEFAULT_AUDIO_DURATION, int)
        self.assertIsInstance(Config.DEFAULT_SAMPLE_RATE, int)
    
    def test_api_key_validation(self):
        """Test API key validation."""
        validation = Config.validate_api_keys()
        self.assertIsInstance(validation, dict)
        self.assertIn('doordash', validation)
        self.assertIn('uber', validation)
        self.assertIn('expedia', validation)
        self.assertIn('plaid', validation)
    
    def test_missing_api_keys(self):
        """Test missing API keys detection."""
        missing = Config.get_missing_api_keys()
        self.assertIsInstance(missing, list)
    
    def test_config_status_printing(self):
        """Test configuration status printing."""
        # Should not raise exception
        Config.print_config_status()


class TestDatabase(unittest.TestCase):
    """Test database functionality."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db_service = DatabaseService(self.db_path)
    
    def tearDown(self):
        """Clean up test database."""
        self.db_service.close()
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test database initialization."""
        self.assertIsNotNone(self.db_service.db_manager)
        self.assertIsNotNone(self.db_service.tasks)
        self.assertIsNotNone(self.db_service.users)
        self.assertIsNotNone(self.db_service.logs)
    
    def test_task_operations(self):
        """Test task database operations."""
        # Create task
        task_data = {
            'description': 'Test task',
            'priority': 1,
            'user_id': 'test_user'
        }
        task_id = self.db_service.tasks.create_task(task_data)
        self.assertIsNotNone(task_id)
        
        # Get task
        task = self.db_service.tasks.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task['description'], 'Test task')
        
        # Update task
        update_data = {'status': 'completed'}
        success = self.db_service.tasks.update_task(task_id, update_data)
        self.assertTrue(success)
        
        # Delete task
        success = self.db_service.tasks.delete_task(task_id)
        self.assertTrue(success)
    
    def test_user_operations(self):
        """Test user database operations."""
        user_data = {
            'id': 'test_user',
            'username': 'testuser',
            'email': 'test@example.com'
        }
        
        # Create user
        success = self.db_service.users.create_user(user_data)
        self.assertTrue(success)
        
        # Get user
        user = self.db_service.users.get_user('test_user')
        self.assertIsNotNone(user)
        self.assertEqual(user['username'], 'testuser')
    
    def test_logging_operations(self):
        """Test logging operations."""
        # Log event
        self.db_service.logs.log_event(
            level='INFO',
            message='Test log message',
            module='test'
        )
        
        # Get logs
        logs = self.db_service.logs.get_logs(limit=10)
        self.assertIsInstance(logs, list)
        self.assertGreater(len(logs), 0)
    
    def test_database_health_check(self):
        """Test database health check."""
        health = self.db_service.health_check()
        self.assertIn('status', health)
        self.assertIn('message', health)


class TestPluginSystem(unittest.TestCase):
    """Test plugin system functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.plugin_manager = PluginManager()
    
    def test_plugin_manager_initialization(self):
        """Test plugin manager initialization."""
        self.assertIsNotNone(self.plugin_manager)
        self.assertIsInstance(self.plugin_manager.plugins, dict)
    
    def test_plugin_discovery(self):
        """Test plugin discovery."""
        plugins = self.plugin_manager.discover_plugins()
        self.assertIsInstance(plugins, list)
    
    def test_plugin_loading(self):
        """Test plugin loading."""
        # This will depend on available plugins
        loaded_count = self.plugin_manager.load_all_plugins()
        self.assertIsInstance(loaded_count, int)
        self.assertGreaterEqual(loaded_count, 0)
    
    def test_plugin_management(self):
        """Test plugin management operations."""
        # List plugins
        plugins = self.plugin_manager.list_plugins()
        self.assertIsInstance(plugins, list)
        
        # Get plugin capabilities
        capabilities = self.plugin_manager.get_plugin_capabilities()
        self.assertIsInstance(capabilities, dict)
        
        # Health check
        health = self.plugin_manager.health_check()
        self.assertIn('total_plugins', health)
        self.assertIn('enabled_plugins', health)


class TestBasePlugin(unittest.TestCase):
    """Test base plugin functionality."""
    
    def setUp(self):
        """Set up test environment."""
        class TestPlugin(BasePlugin):
            def initialize(self):
                return True
            
            def get_capabilities(self):
                return ['test_capability']
            
            def execute(self, command, **kwargs):
                return {'result': 'test_result'}
        
        self.plugin = TestPlugin("test_plugin")
    
    def test_plugin_initialization(self):
        """Test plugin initialization."""
        self.assertEqual(self.plugin.name, "test_plugin")
        self.assertEqual(self.plugin.version, "1.0.0")
        self.assertTrue(self.plugin.enabled)
    
    def test_plugin_info(self):
        """Test plugin information."""
        info = self.plugin.get_info()
        self.assertIn('name', info)
        self.assertIn('version', info)
        self.assertIn('enabled', info)
        self.assertIn('capabilities', info)
    
    def test_plugin_enable_disable(self):
        """Test plugin enable/disable."""
        self.plugin.disable()
        self.assertFalse(self.plugin.enabled)
        
        self.plugin.enable()
        self.assertTrue(self.plugin.enabled)
    
    def test_plugin_execution(self):
        """Test plugin command execution."""
        result = self.plugin.execute("test_command")
        self.assertEqual(result['result'], 'test_result')
    
    def test_plugin_health_check(self):
        """Test plugin health check."""
        health = self.plugin.health_check()
        self.assertIn('name', health)
        self.assertIn('status', health)
        self.assertIn('version', health)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def setUp(self):
        """Set up test environment."""
        self.lam = LAM("Integration Test Agent")
        self.agent = Agent("Integration Test Agent")
    
    def test_lam_to_agent_integration(self):
        """Test integration between LAM and Agent."""
        # Test that agent inherits from LAM
        self.assertIsInstance(self.agent, LAM)
        
        # Test that both can manage tasks
        self.lam.add_task("LAM task")
        self.agent.add_task("Agent task")
        
        self.assertEqual(len(self.lam.tasks), 1)
        self.assertEqual(len(self.agent.tasks), 1)
    
    def test_config_integration(self):
        """Test configuration integration."""
        # Test that components can access config
        self.assertIsNotNone(Config.TEXT_MODEL_NAME)
        self.assertIsNotNone(Config.DEFAULT_TASK_PRIORITY)
    
    def test_error_handling_integration(self):
        """Test error handling integration."""
        # Test that errors are handled gracefully
        try:
            # This might fail if models aren't available
            self.lam.generate_response("Test")
        except Exception:
            pass  # Expected in test environment
        
        # Test that the system continues to work
        self.lam.add_task("Recovery task")
        self.assertEqual(len(self.lam.tasks), 1)


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 Running comprehensive LAM tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestLAMCore,
        TestAgent,
        TestConfiguration,
        TestDatabase,
        TestPluginSystem,
        TestBasePlugin,
        TestIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n📊 Comprehensive Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All comprehensive tests passed!")
        return True
    else:
        print("\n❌ Some comprehensive tests failed!")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
