#!/usr/bin/env python3
"""
Basic tests for the LAM project.
Tests core functionality without requiring external APIs or models.
"""

import unittest
from unittest.mock import patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_attributes(self):
        """Test that config has expected attributes."""
        self.assertTrue(hasattr(Config, 'DOORDASH_API_KEY'))
        self.assertTrue(hasattr(Config, 'UBER_API_KEY'))
        self.assertTrue(hasattr(Config, 'EXPEDIA_API_KEY'))
        self.assertTrue(hasattr(Config, 'PLAID_CLIENT_ID'))
        self.assertTrue(hasattr(Config, 'PLAID_SECRET'))
        self.assertTrue(hasattr(Config, 'TEXT_MODEL_NAME'))
        self.assertTrue(hasattr(Config, 'SPEECH_MODEL_NAME'))
        self.assertTrue(hasattr(Config, 'EMOTION_MODEL_NAME'))
    
    def test_validate_api_keys(self):
        """Test API key validation."""
        validation = Config.validate_api_keys()
        self.assertIsInstance(validation, dict)
        self.assertIn('doordash', validation)
        self.assertIn('uber', validation)
        self.assertIn('expedia', validation)
        self.assertIn('plaid', validation)
    
    def test_get_missing_api_keys(self):
        """Test missing API keys detection."""
        missing = Config.get_missing_api_keys()
        self.assertIsInstance(missing, list)
    
    def test_print_config_status(self):
        """Test configuration status printing."""
        # This should not raise an exception
        try:
            Config.print_config_status()
        except Exception as e:
            self.fail(f"print_config_status raised {e} unexpectedly!")


class TestBasicImports(unittest.TestCase):
    """Test that basic imports work."""
    
    def test_config_import(self):
        """Test config module import."""
        try:
            from config import Config
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import Config: {e}")
    
    @patch.dict(os.environ, {'DOORDASH_API_KEY': 'test_key'})
    def test_environment_variables(self):
        """Test environment variable loading."""
        # Reload config to pick up new env vars
        import importlib
        import config
        importlib.reload(config)
        
        # Check if the key was loaded
        self.assertEqual(config.Config.DOORDASH_API_KEY, 'test_key')


class TestFileStructure(unittest.TestCase):
    """Test that required files exist."""
    
    def test_required_files_exist(self):
        """Test that all required project files exist."""
        required_files = [
            'main.py',
            'agent.py', 
            'LAM.py',
            'config.py',
            'cli.py',
            'requirements.txt',
            'README.md',
            'env.example'
        ]
        
        for file_name in required_files:
            with self.subTest(file=file_name):
                self.assertTrue(
                    os.path.exists(file_name),
                    f"Required file {file_name} not found"
                )
    
    def test_api_directory_exists(self):
        """Test that API directory exists with required files."""
        api_dir = 'API'
        self.assertTrue(os.path.exists(api_dir), "API directory not found")
        
        required_api_files = [
            'Doordash.py',
            'Uber.py', 
            'Expedia.py',
            'Plaid.py'
        ]
        
        for file_name in required_api_files:
            api_file_path = os.path.join(api_dir, file_name)
            with self.subTest(file=api_file_path):
                self.assertTrue(
                    os.path.exists(api_file_path),
                    f"Required API file {api_file_path} not found"
                )


def run_basic_tests():
    """Run basic tests and return results."""
    print("🧪 Running basic LAM tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicImports))
    suite.addTests(loader.loadTestsFromTestCase(TestFileStructure))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n📊 Test Results:")
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
        print("\n✅ All basic tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
