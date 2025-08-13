#!/usr/bin/env python3
"""
Web Dashboard for the LAM (Large Action Model) project.
Provides a graphical web interface for all LAM capabilities.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import Agent
from config import Config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lam-secret-key-2024'
socketio = SocketIO(app)

# Global agent instance
lam_agent = None

def initialize_agent():
    """Initialize the LAM agent."""
    global lam_agent
    try:
        lam_agent = Agent("Web Dashboard Assistant")
        return True
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return False

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard with all features."""
    if not lam_agent:
        if not initialize_agent():
            flash("Failed to initialize LAM agent", "error")
            return redirect(url_for('index'))
    
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """Get system status."""
    if not lam_agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    try:
        status = {
            "agent_name": lam_agent.name,
            "task_count": len(lam_agent.tasks),
            "context": lam_agent.context,
            "api_keys": Config.validate_api_keys(),
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/tasks', methods=['GET', 'POST', 'DELETE'])
def api_tasks():
    """Handle task operations."""
    if not lam_agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    try:
        if request.method == 'GET':
            # Get all tasks
            tasks = []
            for i, task in enumerate(lam_agent.tasks, 1):
                try:
                    # This would need to be handled properly in a real implementation
                    task_info = {
                        "id": i,
                        "priority": task["priority"],
                        "due_time": task["due_time"].isoformat() if task["due_time"] else None,
                        "dependencies": task["dependencies"]
                    }
                    tasks.append(task_info)
                except Exception:
                    task_info = {
                        "id": i,
                        "priority": task["priority"],
                        "due_time": None,
                        "dependencies": task["dependencies"]
                    }
                    tasks.append(task_info)
            
            return jsonify({"tasks": tasks})
        
        elif request.method == 'POST':
            # Add new task
            data = request.get_json()
            task_desc = data.get('description', '')
            priority = int(data.get('priority', 2))
            due_hours = data.get('due_hours')
            
            due_time = None
            if due_hours:
                due_time = datetime.now() + timedelta(hours=int(due_hours))
            
            success = lam_agent.add_task(task_desc, priority, due_time)
            if success:
                return jsonify({"message": "Task added successfully"}), 201
            else:
                return jsonify({"error": "Failed to add task"}), 500
        
        elif request.method == 'DELETE':
            # Remove task
            task_id = int(request.args.get('id', 0))
            if task_id > 0:
                success = lam_agent.remove_task(task_id)
                if success:
                    return jsonify({"message": "Task removed successfully"})
                else:
                    return jsonify({"error": "Failed to remove task"}), 500
            else:
                return jsonify({"error": "Invalid task ID"}), 400
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/chat', methods=['POST'])
def api_ai_chat():
    """Handle AI chat requests."""
    if not lam_agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        response = lam_agent.generate_response(message)
        return jsonify({
            "message": message,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/emotion', methods=['POST'])
def api_emotion():
    """Handle emotion analysis requests."""
    if not lam_agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        emotion = lam_agent.recognize_emotion(text)
        return jsonify({
            "text": text,
            "emotion": emotion,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/food/search', methods=['POST'])
def api_food_search():
    """Search for food delivery options."""
    if not lam_agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    try:
        data = request.get_json()
        location = data.get('location', '')
        cuisine = data.get('cuisine', '')
        max_delivery_time = data.get('max_delivery_time')
        
        if not location:
            return jsonify({"error": "Location is required"}), 400
        
        restaurants = lam_agent.doordash_api.get_restaurants(
            location, cuisine, max_delivery_time
        )
        
        return jsonify({
            "restaurants": restaurants,
            "search_params": {
                "location": location,
                "cuisine": cuisine,
                "max_delivery_time": max_delivery_time
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/food/order', methods=['POST'])
def api_food_order():
    """Place a food order."""
    if not lam_agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    try:
        data = request.get_json()
        order_details = {
            "restaurant_id": data.get('restaurant_id'),
            "items": data.get('items', []),
            "delivery_address": data.get('delivery_address'),
            "total_amount": data.get('total_amount'),
            "restaurant_name": data.get('restaurant_name')
        }
        
        result = lam_agent.doordash_api.place_order(order_details)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify({
            "order": result,
            "message": "Order placed successfully"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/context', methods=['GET', 'PUT'])
def api_context():
    """Handle context updates."""
    if not lam_agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    try:
        if request.method == 'GET':
            return jsonify(lam_agent.context)
        
        elif request.method == 'PUT':
            data = request.get_json()
            location = data.get('location')
            time_info = data.get('time')
            preferences = data.get('user_preferences')
            
            lam_agent.update_context(location, time_info, preferences)
            return jsonify({
                "message": "Context updated successfully",
                "context": lam_agent.context
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/predict', methods=['POST'])
def api_predict():
    """Run predictive analytics."""
    if not lam_agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    try:
        data = request.get_json()
        historical_data = data.get('historical_data', [])
        
        if not historical_data:
            return jsonify({"error": "Historical data is required"}), 400
        
        lam_agent.predictive_analytics(historical_data)
        
        return jsonify({
            "message": "Predictive analytics completed",
            "data_points": len(historical_data)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/neural/process', methods=['POST'])
def api_neural_process():
    """Process task with neural network."""
    if not lam_agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    try:
        data = request.get_json()
        task = data.get('task', '')
        
        if not task:
            return jsonify({"error": "Task is required"}), 400
        
        spike_count = lam_agent.process_task_with_bindsnet(task)
        
        return jsonify({
            "task": task,
            "spike_count": spike_count,
            "message": "Neural processing completed"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('status', {'message': 'Connected to LAM Dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

@socketio.on('request_status')
def handle_status_request():
    """Handle status update requests."""
    if lam_agent:
        status = {
            "agent_name": lam_agent.name,
            "task_count": len(lam_agent.tasks),
            "timestamp": datetime.now().isoformat()
        }
        emit('status_update', status)
    else:
        emit('status_update', {"error": "Agent not available"})

if __name__ == '__main__':
    # Initialize agent on startup
    if initialize_agent():
        print("✅ LAM agent initialized successfully")
    else:
        print("❌ Failed to initialize LAM agent")
    
    # Run the web dashboard
    print("🚀 Starting LAM Web Dashboard...")
    print("📱 Open your browser and go to: http://localhost:5000")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
