# main.py
from agent import Agent
from datetime import datetime, timedelta

if __name__ == "__main__":
    # Create an agent instance
    my_agent = Agent("Assistant Bot")

    # Add tasks to the agent
    my_agent.add_task("Schedule meeting with team", priority=2, due_time=datetime.now() + timedelta(seconds=10))
    my_agent.add_task("Send project update email", priority=1)
    my_agent.add_task("Prepare presentation for Friday", priority=3)

    # Show all tasks
    my_agent.show_tasks()

    # Set a reminder for the second task
    my_agent.set_reminder(2, datetime.now() + timedelta(seconds=20))

    # Remove the first task
    my_agent.remove_task(1)

    # Show updated tasks
    my_agent.show_tasks()

    # Generate a response from the language model
    prompt = "What should I do today?"
    response = my_agent.generate_response(prompt)
    print(f"AI Response: {response}")

    # Record and transcribe audio
    transcription = my_agent.record_and_transcribe_audio(duration=5)
    print(f"Transcription: {transcription}")

    # Recognize emotion in a text
    text = "I am so happy with the progress we are making!"
    emotion = my_agent.recognize_emotion(text)
    print(f"Recognized emotion: {emotion}")

    # Record, process emotion, and respond
    my_agent.record_process_emotion_and_respond(duration=5)

    # Analyze task priority and urgency
    my_agent.analyze_task_priority_and_urgency()

    # Perform predictive analytics
    historical_data = [(10, 50), (20, 100), (15, 75)]  # Example data
    my_agent.predictive_analytics(historical_data)

    # Update context
    my_agent.update_context(location="Office", time="Morning", user_preferences={"theme": "dark"})
    print(f"Updated context: {my_agent.context}")

    # Use APIs
    my_agent.order_food("New York")
    my_agent.request_ride(40.7128, -74.0060, 40.730610, -73.935242)
    my_agent.search_and_book_hotel("Las Vegas", "2024-06-01", "2024-06-07")
    my_agent.get_bank_accounts("user123", "access-sandbox-123456")
