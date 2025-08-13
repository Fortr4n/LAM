#!/usr/bin/env python3
"""
Main entry point for the LAM (Large Action Model) project.
Demonstrates all capabilities of the AI assistant.
"""

import sys
from datetime import datetime, timedelta
from agent import Agent
from config import Config


def main():
    """Main function to demonstrate LAM capabilities."""
    print("🚀 Initializing Large Action Model (LAM)...")
    
    # Check configuration status
    Config.print_config_status()
    
    try:
        # Create an agent instance
        print("\n🤖 Creating AI agent...")
        my_agent = Agent("Assistant Bot")
        
        # Demonstrate task management
        print("\n📋 === Task Management Demo ===")
        demo_task_management(my_agent)
        
        # Demonstrate AI capabilities
        print("\n🧠 === AI Capabilities Demo ===")
        demo_ai_capabilities(my_agent)
        
        # Demonstrate API integrations
        print("\n🔌 === API Integration Demo ===")
        demo_api_integrations(my_agent)
        
        # Demonstrate advanced features
        print("\n⚡ === Advanced Features Demo ===")
        demo_advanced_features(my_agent)
        
        print("\n✅ LAM demonstration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during LAM demonstration: {e}")
        sys.exit(1)


def demo_task_management(agent: Agent):
    """Demonstrate task management capabilities."""
    try:
        # Add tasks with different priorities and due times
        agent.add_task(
            "Schedule meeting with team", 
            priority=2, 
            due_time=datetime.now() + timedelta(seconds=10)
        )
        agent.add_task("Send project update email", priority=1)
        agent.add_task("Prepare presentation for Friday", priority=3)
        
        # Show all tasks
        agent.show_tasks()
        
        # Set a reminder for the second task
        agent.set_reminder(2, datetime.now() + timedelta(seconds=20))
        
        # Remove the first task
        agent.remove_task(1)
        
        # Show updated tasks
        agent.show_tasks()
        
        # Analyze task priorities
        agent.analyze_task_priority_and_urgency()
        
    except Exception as e:
        print(f"⚠️ Task management demo failed: {e}")


def demo_ai_capabilities(agent: Agent):
    """Demonstrate AI capabilities."""
    try:
        # Generate AI response
        prompt = "What should I do today?"
        response = agent.generate_response(prompt)
        print(f"🤖 AI Response to '{prompt}': {response}")
        
        # Record and transcribe audio
        print("\n🎤 Recording audio for transcription...")
        transcription = agent.record_and_transcribe_audio(duration=3)
        print(f"📝 Transcription: {transcription}")
        
        # Recognize emotion in text
        text = "I am so happy with the progress we are making!"
        emotion = agent.recognize_emotion(text)
        print(f"😊 Recognized emotion in '{text}': {emotion}")
        
        # Record, process emotion, and respond
        print("\n🎤 Recording audio for emotion analysis...")
        agent.record_process_emotion_and_respond(duration=3)
        
    except Exception as e:
        print(f"⚠️ AI capabilities demo failed: {e}")


def demo_api_integrations(agent: Agent):
    """Demonstrate API integration capabilities."""
    try:
        # Check if API keys are available
        missing_keys = Config.get_missing_api_keys()
        if missing_keys:
            missing_keys_str = ', '.join(missing_keys)
            print(f"⚠️ Skipping API demo - missing keys: {missing_keys_str}")
            return
        
        # Food delivery
        print("🍕 Ordering food...")
        agent.order_food("New York")
        
        # Ride service
        print("🚗 Requesting ride...")
        agent.request_ride(40.7128, -74.0060, 40.730610, -73.935242)
        
        # Hotel booking
        print("🏨 Searching for hotels...")
        agent.search_and_book_hotel("Las Vegas", "2024-06-01", "2024-06-07")
        
        # Financial services
        print("💰 Accessing bank accounts...")
        agent.get_bank_accounts("user123", "access-sandbox-123456")
        
    except Exception as e:
        print(f"⚠️ API integration demo failed: {e}")


def demo_advanced_features(agent: Agent):
    """Demonstrate advanced features."""
    try:
        # Predictive analytics
        print("📊 Running predictive analytics...")
        historical_data = [(10, 50), (20, 100), (15, 75)]
        agent.predictive_analytics(historical_data)
        
        # Update context
        print("🔄 Updating agent context...")
        agent.update_context(
            location="Office", 
            time="Morning", 
            user_preferences={"theme": "dark"}
        )
        print(f"📍 Updated context: {agent.context}")
        
        # Neural network processing
        print("🧠 Processing task with neural network...")
        task = "Complete project documentation"
        spike_count = agent.process_task_with_bindsnet(task)
        print(f"⚡ Neural spikes generated: {spike_count}")
        
    except Exception as e:
        print(f"⚠️ Advanced features demo failed: {e}")


if __name__ == "__main__":
    main()
