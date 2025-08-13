# lam.py
import sched
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoModelForSequenceClassification
)
import soundfile as sf
import sounddevice as sd
import pyttsx3
import torch
import torch_directml
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from phe import paillier
from sklearn.linear_model import LinearRegression
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize DirectML device
try:
    dml = torch_directml.device()
    logger.info("DirectML device initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize DirectML: {e}")
    dml = torch.device('cpu')
    logger.info("Falling back to CPU")

# Initialize Text-to-Speech (TTS) engine
try:
    tts_engine = pyttsx3.init()
    logger.info("TTS engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize TTS engine: {e}")
    tts_engine = None

# Load models with error handling
def load_model_safely(model_name: str, model_type: str):
    """Safely load a model with fallback options."""
    try:
        if model_type == "tokenizer":
            return AutoTokenizer.from_pretrained(model_name)
        elif model_type == "text":
            return AutoModelForCausalLM.from_pretrained(
                model_name
            ).to(dml)
        elif model_type == "speech":
            return AutoProcessor.from_pretrained(model_name)
        elif model_type == "emotion":
            return AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(dml)
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return None

# Load models
text_tokenizer = load_model_safely(Config.TEXT_MODEL_NAME, "tokenizer")
text_model = load_model_safely(Config.TEXT_MODEL_NAME, "text")
speech_processor = load_model_safely(Config.SPEECH_MODEL_NAME, "speech")
speech_model = load_model_safely(Config.SPEECH_MODEL_NAME, "speech")
emotion_tokenizer = load_model_safely(Config.EMOTION_MODEL_NAME, "tokenizer")
emotion_model = load_model_safely(Config.EMOTION_MODEL_NAME, "emotion")

# Initialize homomorphic encryption
try:
    public_key, private_key = paillier.generate_paillier_keypair()
    logger.info("Homomorphic encryption keys generated successfully")
except Exception as e:
    logger.error(f"Failed to generate encryption keys: {e}")
    public_key, private_key = None, None

# Machine learning model for predictive analytics
predictive_model = LinearRegression()


class IzhikevichNodes(torch.nn.Module):
    """Izhikevich neuron model implementation."""
    
    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.v = -65.0 * torch.ones(n, device=dml)
        self.u = self.v * 0.2
        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 8.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v
        u = self.u
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        fired = v >= 30.0
        self.v[fired] = c
        self.u[fired] += d
        self.v += 0.04 * v**2 + 5 * v + 140 - u + x
        self.u += a * (b * v - u)

        return fired.float()


class LAM:
    """Large Action Model - Core AI assistant with task management and AI capabilities."""
    
    def __init__(self, name: str):
        self.name = name
        self.tasks: List[Dict[str, Any]] = []
        self.task_dependencies: Dict[str, List[str]] = {}
        self.network = self.create_bindsnet_model()
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.context = {"location": None, "time": None, "user_preferences": {}}

    def create_bindsnet_model(self) -> Network:
        """Create a BindsNET neural network."""
        try:
            network = Network()

            # Define input and Izhikevich neuron layers
            input_layer = Input(n=Config.BINDSNET_INPUT_SIZE)
            izhikevich_layer = IzhikevichNodes(n=Config.BINDSNET_INPUT_SIZE)

            # Connect input to Izhikevich neurons
            connection = Connection(
                source=input_layer, 
                target=izhikevich_layer, 
                w=0.05 * torch.randn(
                    input_layer.n, 
                    izhikevich_layer.n, 
                    device=dml
                )
            )

            # Add layers and connections to the network
            network.add_layer(input_layer, name="Input")
            network.add_layer(izhikevich_layer, name="Izhikevich")
            network.add_connection(connection, source="Input", target="Izhikevich")

            # Add a monitor to record spikes from the Izhikevich layer
            monitor = Monitor(
                obj=izhikevich_layer, 
                state_vars=["s"], 
                time=Config.BINDSNET_SIMULATION_TIME
            )
            network.add_monitor(monitor, name="Izhikevich")

            return network
        except Exception as e:
            logger.error(f"Failed to create BindsNET model: {e}")
            return None

    def simulate_bindsnet_model(self, input_value: float) -> Optional[torch.Tensor]:
        """Simulate the BindsNET neural network."""
        try:
            if self.network is None:
                logger.warning("BindsNET network not available")
                return None
                
            # Reset the network
            self.network.reset_state_variables()

            # Create input spikes (binary)
            spikes = torch.bernoulli(
                input_value * torch.ones(Config.BINDSNET_INPUT_SIZE, device=dml)
            ).byte()

            # Run the network
            self.network.run(
                inputs={"Input": spikes}, 
                time=Config.BINDSNET_SIMULATION_TIME
            )

            # Get spike recordings
            izhikevich_spikes = self.network.monitors["Izhikevich"].get("s")

            return izhikevich_spikes
        except Exception as e:
            logger.error(f"Error in simulate_bindsnet_model: {e}")
            return None

    def add_task(self, task: str, priority: int = Config.DEFAULT_TASK_PRIORITY, 
                 due_time: Optional[datetime] = None, dependencies: List[str] = []) -> bool:
        """Add a new task with encryption and scheduling."""
        try:
            if not public_key or not private_key:
                logger.error("Encryption keys not available")
                return False
                
            encrypted_task = paillier.EncryptedNumber(
                public_key, 
                public_key.encrypt(task)
            )
            
            task_data = {
                "task": encrypted_task, 
                "priority": priority, 
                "due_time": due_time, 
                "dependencies": dependencies
            }
            
            self.tasks.append(task_data)
            self.tasks.sort(
                key=lambda x: (
                    x["priority"], 
                    x["due_time"] if x["due_time"] else datetime.max
                )
            )
            
            for dep in dependencies:
                self.task_dependencies[dep] = self.task_dependencies.get(dep, []) + [task]
            
            print(f"Task '{task}' added with priority {priority}.")
            
            if due_time:
                delay = (due_time - datetime.now()).total_seconds()
                if delay > 0:
                    self.scheduler.enter(delay, 1, self.notify, argument=(task,))
                    self.scheduler.run(blocking=False)
            
            return True
        except Exception as e:
            logger.error(f"Error in add_task: {e}")
            return False

    def show_tasks(self) -> None:
        """Display all tasks with decryption."""
        try:
            if not private_key:
                logger.error("Private key not available for decryption")
                return
                
            print(f"Tasks for {self.name}:")
            for idx, task in enumerate(self.tasks, start=1):
                try:
                    decrypted_task = private_key.decrypt(task["task"])
                    due_time_str = (
                        task["due_time"].strftime('%Y-%m-%d %H:%M:%S') 
                        if task["due_time"] else "No due time"
                    )
                    print(f"{idx}. {decrypted_task} (Priority: {task['priority']}, Due: {due_time_str})")
                except Exception as e:
                    logger.error(f"Failed to decrypt task {idx}: {e}")
                    print(f"{idx}. [Encrypted Task] (Priority: {task['priority']})")
        except Exception as e:
            logger.error(f"Error in show_tasks: {e}")

    def remove_task(self, task_number: int) -> bool:
        """Remove a task by number."""
        try:
            if 0 < task_number <= len(self.tasks):
                removed_task = self.tasks.pop(task_number - 1)
                if private_key:
                    decrypted_task = private_key.decrypt(removed_task["task"])
                    print(f"Task '{decrypted_task}' removed.")
                else:
                    print("Task removed.")
                return True
            else:
                print("Invalid task number.")
                return False
        except Exception as e:
            logger.error(f"Error in remove_task: {e}")
            return False

    def set_reminder(self, task_number: int, reminder_time: datetime) -> bool:
        """Set a reminder for a specific task."""
        try:
            if 0 < task_number <= len(self.tasks):
                task = self.tasks[task_number - 1]["task"]
                if private_key:
                    decrypted_task = private_key.decrypt(task)
                else:
                    decrypted_task = "[Encrypted Task]"
                    
                self.tasks[task_number - 1]["due_time"] = reminder_time
                delay = (reminder_time - datetime.now()).total_seconds()
                
                if delay > 0:
                    self.scheduler.enter(delay, 1, self.notify, argument=(decrypted_task,))
                    self.scheduler.run(blocking=False)
                
                print(f"Reminder set for task '{decrypted_task}' at {reminder_time}.")
                return True
            else:
                print("Invalid task number.")
                return False
        except Exception as e:
            logger.error(f"Error in set_reminder: {e}")
            return False

    def notify(self, task: str) -> None:
        """Send notification for a task."""
        print(f"Reminder: Task '{task}' is due!")
        if tts_engine:
            try:
                self.speak(f"Reminder: Task '{task}' is due!")
            except Exception as e:
                logger.error(f"TTS notification failed: {e}")

    def generate_response(self, prompt: str) -> str:
        """Generate AI response using the language model."""
        try:
            if not text_tokenizer or not text_model:
                return "Language model not available."
                
            inputs = text_tokenizer(prompt, return_tensors="pt").to(dml)
            outputs = text_model.generate(inputs.input_ids, max_length=50)
            response = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return "Sorry, I couldn't generate a response at the moment."

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        try:
            if not speech_processor or not speech_model:
                return "Speech recognition model not available."
                
            speech_array, _ = sf.read(audio_path)
            inputs = speech_processor(
                speech_array, 
                sampling_rate=Config.DEFAULT_SAMPLE_RATE, 
                return_tensors="pt"
            ).to(dml)
            generated_ids = speech_model.generate(inputs.input_ids)
            transcription = speech_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            return transcription
        except Exception as e:
            logger.error(f"Error in transcribe_audio: {e}")
            return "Sorry, I couldn't transcribe the audio at the moment."

    def record_and_transcribe_audio(self, duration: int = Config.DEFAULT_AUDIO_DURATION, 
                                   fs: int = Config.DEFAULT_SAMPLE_RATE) -> str:
        """Record audio and transcribe it."""
        try:
            print("Recording...")
            audio = sd.rec(
                int(duration * fs), 
                samplerate=fs, 
                channels=Config.DEFAULT_CHANNELS, 
                dtype='float32'
            )
            sd.wait()
            print("Recording complete.")
            
            audio_path = "temp_audio.wav"
            sf.write(audio_path, audio, fs)
            return self.transcribe_audio(audio_path)
        except Exception as e:
            logger.error(f"Error in record_and_transcribe_audio: {e}")
            return "Sorry, I couldn't transcribe the audio at the moment."

    def recognize_emotion(self, text: str) -> str:
        """Recognize emotion in text."""
        try:
            if not emotion_tokenizer or not emotion_model:
                return "Emotion recognition model not available."
                
            inputs = emotion_tokenizer(text, return_tensors="pt").to(dml)
            outputs = emotion_model(**inputs)
            scores = outputs.logits[0].detach().cpu().numpy()
            emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            emotion = emotions[np.argmax(scores)]
            return emotion
        except Exception as e:
            logger.error(f"Error in recognize_emotion: {e}")
            return "Emotion recognition failed."

    def process_task_with_bindsnet(self, task: str) -> int:
        """Process a task using the neural network."""
        try:
            input_value = torch.tensor([len(task)]).float().to(dml) / 10
            spikes = self.simulate_bindsnet_model(input_value)
            if spikes is not None:
                spike_count = spikes.sum().item()
                return spike_count
            else:
                return 0
        except Exception as e:
            logger.error(f"Error in process_task_with_bindsnet: {e}")
            return 0

    def record_process_emotion_and_respond(self, duration: int = Config.DEFAULT_AUDIO_DURATION, 
                                         fs: int = Config.DEFAULT_SAMPLE_RATE) -> None:
        """Record audio, process emotion, and generate response."""
        try:
            transcription = self.record_and_transcribe_audio(duration, fs)
            print(f"Transcribed text: {transcription}")
            
            emotion = self.recognize_emotion(transcription)
            print(f"Recognized emotion: {emotion}")
            
            response = self.generate_response(
                f"The detected emotion is {emotion}. How can I help you with that?"
            )
            
            if tts_engine:
                self.speak(response)
            else:
                print(f"AI Response: {response}")
        except Exception as e:
            logger.error(f"Error in record_process_emotion_and_respond: {e}")
            if tts_engine:
                try:
                    self.speak("Sorry, I couldn't process the emotion at the moment.")
                except Exception as tts_error:
                    logger.error(f"TTS error: {tts_error}")

    def analyze_task_priority_and_urgency(self) -> None:
        """Analyze and adjust task priorities based on urgency."""
        try:
            for task in self.tasks:
                if private_key:
                    decrypted_task = private_key.decrypt(task["task"])
                else:
                    continue
                    
                priority = task["priority"]
                due_time = task["due_time"]
                
                if due_time:
                    urgency = (due_time - datetime.now()).total_seconds()
                    
                    if urgency < Config.URGENCY_THRESHOLD_HOUR:
                        task["priority"] = max(priority, 1)
                    elif urgency < Config.URGENCY_THRESHOLD_DAY:
                        task["priority"] = max(priority, 2)
                    else:
                        task["priority"] = max(priority, 3)
            
            self.tasks.sort(
                key=lambda x: (
                    x["priority"], 
                    x["due_time"] if x["due_time"] else datetime.max
                )
            )
        except Exception as e:
            logger.error(f"Error in analyze_task_priority_and_urgency: {e}")

    def predictive_analytics(self, historical_data: List[Tuple[int, int]]) -> None:
        """Perform predictive analytics on task completion times."""
        try:
            if not historical_data:
                logger.warning("No historical data provided for prediction")
                return
                
            X = np.array([data[0] for data in historical_data]).reshape(-1, 1)
            y = np.array([data[1] for data in historical_data])
            
            predictive_model.fit(X, y)
            
            if self.tasks and private_key:
                last_task = private_key.decrypt(self.tasks[-1]["task"])
                predicted_time = predictive_model.predict(
                    np.array([[len(last_task)]])
                ).item()
                print(f"Predicted time for the new task: {predicted_time:.2f} seconds")
            else:
                print("No tasks available for prediction")
        except Exception as e:
            logger.error(f"Error in predictive_analytics: {e}")

    def update_context(self, location: Optional[str] = None, 
                      time: Optional[str] = None, 
                      user_preferences: Optional[Dict[str, Any]] = None) -> None:
        """Update the agent's context information."""
        if location:
            self.context["location"] = location
        if time:
            self.context["time"] = time
        if user_preferences:
            self.context["user_preferences"] = user_preferences

    def speak(self, text: str) -> None:
        """Convert text to speech."""
        if tts_engine:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS failed: {e}")
                print(f"TTS Error: {text}")
        else:
            print(f"TTS not available: {text}")
