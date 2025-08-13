#!/usr/bin/env python3
"""
Command Line Interface for the LAM (Large Action Model) project.
Provides interactive access to all LAM capabilities.
"""

import cmd
import sys
from datetime import datetime, timedelta
from agent import Agent
from config import Config


class LAMCLI(cmd.Cmd):
    """Interactive command-line interface for the LAM project."""
    
    intro = """
🤖 Welcome to the Large Action Model (LAM) CLI!
Type 'help' or '?' to see available commands.
Type 'demo' to run a full demonstration.
Type 'quit' or 'exit' to exit.
"""
    prompt = 'LAM> '
    
    def __init__(self):
        super().__init__()
        self.agent = None
        self.initialize_agent()
    
    def initialize_agent(self):
        """Initialize the AI agent."""
        try:
            print("🚀 Initializing LAM agent...")
            Config.print_config_status()
            self.agent = Agent("CLI Assistant")
            print("✅ Agent initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            self.agent = None
    
    def do_demo(self, arg):
        """Run a full demonstration of LAM capabilities."""
        if not self.agent:
            print("❌ Agent not initialized. Cannot run demo.")
            return
        
        print("\n🎬 Running LAM demonstration...")
        try:
            # Task management demo
            print("\n📋 === Task Management ===")
            self.agent.add_task("Demo task 1", priority=1)
            self.agent.add_task("Demo task 2", priority=2)
            self.agent.show_tasks()
            
            # AI capabilities demo
            print("\n🧠 === AI Capabilities ===")
            response = self.agent.generate_response("Hello, how are you?")
            print(f"AI Response: {response}")
            
            # Context update demo
            print("\n🔄 === Context Management ===")
            self.agent.update_context(location="CLI", time="Demo")
            print(f"Context: {self.agent.context}")
            
            print("\n✅ Demo completed successfully!")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    def do_add_task(self, arg):
        """Add a new task. Usage: add_task <task_description> [priority] [due_hours]"""
        if not self.agent:
            print("❌ Agent not initialized.")
            return
        
        if not arg:
            print("❌ Please provide a task description.")
            return
        
        try:
            parts = arg.split()
            task_desc = parts[0]
            priority = int(parts[1]) if len(parts) > 1 else 2
            due_hours = int(parts[2]) if len(parts) > 2 else None
            
            due_time = None
            if due_hours:
                due_time = datetime.now() + timedelta(hours=due_hours)
            
            success = self.agent.add_task(task_desc, priority, due_time)
            if success:
                print(f"✅ Task '{task_desc}' added successfully!")
            else:
                print("❌ Failed to add task.")
                
        except ValueError:
            print("❌ Invalid priority or due time. Use: add_task <description> [priority] [due_hours]")
        except Exception as e:
            print(f"❌ Error adding task: {e}")
    
    def do_show_tasks(self, arg):
        """Show all current tasks."""
        if not self.agent:
            print("❌ Agent not initialized.")
            return
        
        try:
            self.agent.show_tasks()
        except Exception as e:
            print(f"❌ Error showing tasks: {e}")
    
    def do_remove_task(self, arg):
        """Remove a task by number. Usage: remove_task <task_number>"""
        if not self.agent:
            print("❌ Agent not initialized.")
            return
        
        if not arg:
            print("❌ Please provide a task number.")
            return
        
        try:
            task_num = int(arg)
            success = self.agent.remove_task(task_num)
            if success:
                print("✅ Task removed successfully!")
            else:
                print("❌ Failed to remove task.")
        except ValueError:
            print("❌ Invalid task number.")
        except Exception as e:
            print(f"❌ Error removing task: {e}")
    
    def do_ai_chat(self, arg):
        """Chat with the AI. Usage: ai_chat <your_message>"""
        if not self.agent:
            print("❌ Agent not initialized.")
            return
        
        if not arg:
            print("❌ Please provide a message.")
            return
        
        try:
            response = self.agent.generate_response(arg)
            print(f"🤖 AI: {response}")
        except Exception as e:
            print(f"❌ Error generating response: {e}")
    
    def do_emotion(self, arg):
        """Analyze emotion in text. Usage: emotion <text>"""
        if not self.agent:
            print("❌ Agent not initialized.")
            return
        
        if not arg:
            print("❌ Please provide text to analyze.")
            return
        
        try:
            emotion = self.agent.recognize_emotion(arg)
            print(f"😊 Detected emotion: {emotion}")
        except Exception as e:
            print(f"❌ Error analyzing emotion: {e}")
    
    def do_record(self, arg):
        """Record and transcribe audio. Usage: record [duration_seconds]"""
        if not self.agent:
            print("❌ Agent not initialized.")
            return
        
        try:
            duration = int(arg) if arg else 5
            print(f"🎤 Recording for {duration} seconds...")
            transcription = self.agent.record_and_transcribe_audio(duration)
            print(f"📝 Transcription: {transcription}")
        except ValueError:
            print("❌ Invalid duration. Please provide a number of seconds.")
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
    
    def do_context(self, arg):
        """Show or update context. Usage: context [location] [time] [preferences]"""
        if not self.agent:
            print("❌ Agent not initialized.")
            return
        
        if not arg:
            print(f"📍 Current context: {self.agent.context}")
            return
        
        try:
            parts = arg.split()
            location = parts[0] if len(parts) > 0 else None
            time_info = parts[1] if len(parts) > 1 else None
            preferences = parts[2] if len(parts) > 2 else None
            
            self.agent.update_context(location, time_info, preferences)
            print(f"✅ Context updated: {self.agent.context}")
        except Exception as e:
            print(f"❌ Error updating context: {e}")
    
    def do_status(self, arg):
        """Show system status and configuration."""
        Config.print_config_status()
        if self.agent:
            print(f"\n🤖 Agent: {self.agent.name}")
            print(f"📋 Tasks: {len(self.agent.tasks)}")
            print(f"📍 Context: {self.agent.context}")
        else:
            print("\n❌ Agent not initialized.")
    
    def do_quit(self, arg):
        """Exit the CLI."""
        print("\n👋 Goodbye! Thanks for using LAM!")
        return True
    
    def do_exit(self, arg):
        """Exit the CLI."""
        return self.do_quit(arg)
    
    def do_EOF(self, arg):
        """Handle Ctrl+D exit."""
        return self.do_quit(arg)
    
    def default(self, line):
        """Handle unknown commands."""
        print(f"❓ Unknown command: {line}")
        print("Type 'help' to see available commands.")
    
    def emptyline(self):
        """Do nothing on empty line."""
        pass


def main():
    """Main entry point for the CLI."""
    try:
        cli = LAMCLI()
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using LAM!")
    except Exception as e:
        print(f"❌ CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
