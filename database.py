"""
Database integration for LAM project.

Provides persistent storage for tasks, user data, and system logs.
Supports both SQLite (default) and PostgreSQL.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, db_path: str = "lam.db", db_type: str = "sqlite"):
        self.db_path = db_path
        self.db_type = db_type
        self.connection = None
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the database and create tables."""
        try:
            if self.db_type == "sqlite":
                self._init_sqlite()
            elif self.db_type == "postgresql":
                self._init_postgresql()
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
            logger.info(f"Database initialized successfully: {self.db_type}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _init_sqlite(self):
        """Initialize SQLite database."""
        self.connection = sqlite3.connect(
            self.db_path, 
            check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        
        # Create tables
        self._create_tables()
    
    def _init_postgresql(self):
        """Initialize PostgreSQL database."""
        # This would require psycopg2 or asyncpg
        # For now, we'll use SQLite as the default
        logger.warning("PostgreSQL not implemented, falling back to SQLite")
        self.db_type = "sqlite"
        self._init_sqlite()
    
    def _create_tables(self):
        """Create necessary database tables."""
        with self.lock:
            cursor = self.connection.cursor()
            
            # Tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    priority INTEGER DEFAULT 2,
                    due_time TIMESTAMP,
                    dependencies TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT DEFAULT 'default'
                )
            """)
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT,
                    preferences TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """)
            
            # System logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    metadata TEXT
                )
            """)
            
            # API calls table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_name TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status_code INTEGER,
                    response_time REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    request_data TEXT,
                    response_data TEXT
                )
            """)
            
            # AI interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_type TEXT NOT NULL,
                    user_input TEXT,
                    ai_response TEXT,
                    emotion_detected TEXT,
                    processing_time REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    metadata TEXT
                )
            """)
            
            self.connection.commit()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        with self.lock:
            try:
                cursor = self.connection.cursor()
                cursor.execute(query, params)
                
                if query.strip().upper().startswith('SELECT'):
                    columns = [description[0] for description in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                else:
                    self.connection.commit()
                    return [{"affected_rows": cursor.rowcount}]
                    
            except Exception as e:
                logger.error(f"Database query error: {e}")
                self.connection.rollback()
                raise
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


class TaskRepository:
    """Repository for task-related database operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create_task(self, task_data: Dict[str, Any]) -> Optional[int]:
        """Create a new task."""
        try:
            query = """
                INSERT INTO tasks (description, priority, due_time, dependencies, user_id)
                VALUES (?, ?, ?, ?, ?)
            """
            params = (
                task_data.get('description'),
                task_data.get('priority', 2),
                task_data.get('due_time'),
                json.dumps(task_data.get('dependencies', [])),
                task_data.get('user_id', 'default')
            )
            
            result = self.db.execute_query(query, params)
            if result and 'affected_rows' in result[0]:
                return result[0]['affected_rows']
            return None
            
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return None
    
    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get a task by ID."""
        try:
            query = "SELECT * FROM tasks WHERE id = ?"
            result = self.db.execute_query(query, (task_id,))
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error getting task: {e}")
            return None
    
    def get_all_tasks(self, user_id: str = 'default') -> List[Dict[str, Any]]:
        """Get all tasks for a user."""
        try:
            query = "SELECT * FROM tasks WHERE user_id = ? ORDER BY priority, due_time"
            return self.db.execute_query(query, (user_id,))
            
        except Exception as e:
            logger.error(f"Error getting tasks: {e}")
            return []
    
    def update_task(self, task_id: int, task_data: Dict[str, Any]) -> bool:
        """Update a task."""
        try:
            query = """
                UPDATE tasks 
                SET description = ?, priority = ?, due_time = ?, 
                    dependencies = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
            params = (
                task_data.get('description'),
                task_data.get('priority'),
                task_data.get('due_time'),
                json.dumps(task_data.get('dependencies', [])),
                task_data.get('status'),
                task_id
            )
            
            result = self.db.execute_query(query, params)
            return result[0]['affected_rows'] > 0 if result else False
            
        except Exception as e:
            logger.error(f"Error updating task: {e}")
            return False
    
    def delete_task(self, task_id: int) -> bool:
        """Delete a task."""
        try:
            query = "DELETE FROM tasks WHERE id = ?"
            result = self.db.execute_query(query, (task_id,))
            return result[0]['affected_rows'] > 0 if result else False
            
        except Exception as e:
            logger.error(f"Error deleting task: {e}")
            return False
    
    def get_tasks_by_status(self, status: str, user_id: str = 'default') -> List[Dict[str, Any]]:
        """Get tasks by status."""
        try:
            query = "SELECT * FROM tasks WHERE status = ? AND user_id = ? ORDER BY priority, due_time"
            return self.db.execute_query(query, (status, user_id))
            
        except Exception as e:
            logger.error(f"Error getting tasks by status: {e}")
            return []


class UserRepository:
    """Repository for user-related database operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create_user(self, user_data: Dict[str, Any]) -> bool:
        """Create a new user."""
        try:
            query = """
                INSERT INTO users (id, username, email, preferences)
                VALUES (?, ?, ?, ?)
            """
            params = (
                user_data.get('id'),
                user_data.get('username'),
                user_data.get('email'),
                json.dumps(user_data.get('preferences', {}))
            )
            
            result = self.db.execute_query(query, params)
            return result[0]['affected_rows'] > 0 if result else False
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user by ID."""
        try:
            query = "SELECT * FROM users WHERE id = ?"
            result = self.db.execute_query(query, (user_id,))
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        try:
            query = "UPDATE users SET preferences = ? WHERE id = ?"
            params = (json.dumps(preferences), user_id)
            
            result = self.db.execute_query(query, params)
            return result[0]['affected_rows'] > 0 if result else False
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False


class LogRepository:
    """Repository for system logging operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def log_event(self, level: str, message: str, module: str = None, 
                  user_id: str = None, metadata: Dict[str, Any] = None):
        """Log a system event."""
        try:
            query = """
                INSERT INTO system_logs (level, message, module, user_id, metadata)
                VALUES (?, ?, ?, ?, ?)
            """
            params = (
                level,
                message,
                module,
                user_id,
                json.dumps(metadata) if metadata else None
            )
            
            self.db.execute_query(query, params)
            
        except Exception as e:
            logger.error(f"Error logging event: {e}")
    
    def get_logs(self, level: str = None, module: str = None, 
                 limit: int = 100) -> List[Dict[str, Any]]:
        """Get system logs with optional filtering."""
        try:
            query = "SELECT * FROM system_logs WHERE 1=1"
            params = []
            
            if level:
                query += " AND level = ?"
                params.append(level)
            
            if module:
                query += " AND module = ?"
                params.append(module)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            return self.db.execute_query(query, tuple(params))
            
        except Exception as e:
            logger.error(f"Error getting logs: {e}")
            return []


class DatabaseService:
    """Main service class for database operations."""
    
    def __init__(self, db_path: str = "lam.db", db_type: str = "sqlite"):
        self.db_manager = DatabaseManager(db_path, db_type)
        self.tasks = TaskRepository(self.db_manager)
        self.users = UserRepository(self.db_manager)
        self.logs = LogRepository(self.db_manager)
    
    def close(self):
        """Close the database service."""
        self.db_manager.close()
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            # Test basic operations
            test_task = {
                'description': 'Health check task',
                'priority': 1,
                'user_id': 'system'
            }
            
            task_id = self.tasks.create_task(test_task)
            if task_id:
                self.tasks.delete_task(task_id)
                return {"status": "healthy", "message": "Database operations working"}
            else:
                return {"status": "unhealthy", "message": "Failed to create test task"}
                
        except Exception as e:
            return {"status": "unhealthy", "message": f"Database error: {e}"}
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            if self.db_manager.db_type == "sqlite":
                import shutil
                shutil.copy2(self.db_manager.db_path, backup_path)
                return True
            else:
                logger.warning("Backup not implemented for this database type")
                return False
                
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False
