"""
Weather Plugin for LAM

Provides weather information and forecasts through various weather APIs.
"""

import requests
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta

from ..base import BasePlugin


class WeatherPlugin(BasePlugin):
    """Weather information plugin for LAM."""
    
    def __init__(self):
        super().__init__("weather", "1.0.0")
        self.api_key = None
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        # Mock weather data for demonstration
        self.mock_weather = {
            "New York": {
                "current": {
                    "temp": 72,
                    "feels_like": 75,
                    "humidity": 65,
                    "description": "Partly cloudy",
                    "icon": "02d"
                },
                "forecast": [
                    {"date": "2024-01-15", "temp": 70, "description": "Sunny"},
                    {"date": "2024-01-16", "temp": 68, "description": "Cloudy"},
                    {"date": "2024-01-17", "temp": 72, "description": "Partly cloudy"}
                ]
            },
            "London": {
                "current": {
                    "temp": 55,
                    "feels_like": 52,
                    "humidity": 80,
                    "description": "Light rain",
                    "icon": "10d"
                },
                "forecast": [
                    {"date": "2024-01-15", "temp": 54, "description": "Rainy"},
                    {"date": "2024-01-16", "temp": 56, "description": "Cloudy"},
                    {"date": "2024-01-17", "temp": 58, "description": "Partly cloudy"}
                ]
            },
            "Tokyo": {
                "current": {
                    "temp": 45,
                    "feels_like": 42,
                    "humidity": 70,
                    "description": "Clear sky",
                    "icon": "01d"
                },
                "forecast": [
                    {"date": "2024-01-15", "temp": 44, "description": "Clear"},
                    {"date": "2024-01-16", "temp": 46, "description": "Sunny"},
                    {"date": "2024-01-17", "temp": 48, "description": "Partly cloudy"}
                ]
            }
        }
    
    def initialize(self) -> bool:
        """Initialize the weather plugin."""
        try:
            # In a real implementation, you would load API keys from config
            # self.api_key = Config.WEATHER_API_KEY
            
            self.logger.info("Weather plugin initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize weather plugin: {e}")
            return False
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this plugin provides."""
        return [
            "get_current_weather",
            "get_weather_forecast",
            "get_weather_alerts",
            "convert_temperature",
            "get_air_quality"
        ]
    
    def execute(self, command: str, **kwargs) -> Dict[str, Any]:
        """Execute a weather plugin command."""
        if not self.enabled:
            return {"error": "Plugin is disabled"}
        
        try:
            if command == "get_current_weather":
                return self.get_current_weather(**kwargs)
            elif command == "get_weather_forecast":
                return self.get_weather_forecast(**kwargs)
            elif command == "get_weather_alerts":
                return self.get_weather_alerts(**kwargs)
            elif command == "convert_temperature":
                return self.convert_temperature(**kwargs)
            elif command == "get_air_quality":
                return self.get_air_quality(**kwargs)
            else:
                return {"error": f"Unknown command: {command}"}
                
        except Exception as e:
            self.logger.error(f"Error executing weather command: {e}")
            return {"error": str(e)}
    
    def get_current_weather(self, city: str = "New York") -> Dict[str, Any]:
        """Get current weather for a city."""
        if self.api_key:
            # Real API call
            try:
                url = f"{self.base_url}/weather"
                params = {
                    "q": city,
                    "appid": self.api_key,
                    "units": "imperial"
                }
                response = requests.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                self.logger.error(f"Error fetching weather data: {e}")
                return {"error": f"Failed to fetch weather data: {e}"}
        else:
            # Mock response
            if city in self.mock_weather:
                weather_data = self.mock_weather[city]["current"]
                return {
                    "city": city,
                    "temperature": weather_data["temp"],
                    "feels_like": weather_data["feels_like"],
                    "humidity": weather_data["humidity"],
                    "description": weather_data["description"],
                    "icon": weather_data["icon"],
                    "timestamp": datetime.now().isoformat(),
                    "source": "mock_data"
                }
            else:
                return {"error": f"Weather data not available for {city}"}
    
    def get_weather_forecast(self, city: str = "New York", days: int = 3) -> Dict[str, Any]:
        """Get weather forecast for a city."""
        if self.api_key:
            # Real API call
            try:
                url = f"{self.base_url}/forecast"
                params = {
                    "q": city,
                    "appid": self.api_key,
                    "units": "imperial"
                }
                response = requests.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                self.logger.error(f"Error fetching forecast data: {e}")
                return {"error": f"Failed to fetch forecast data: {e}"}
        else:
            # Mock response
            if city in self.mock_weather:
                forecast_data = self.mock_weather[city]["forecast"][:days]
                return {
                    "city": city,
                    "forecast": forecast_data,
                    "days": len(forecast_data),
                    "timestamp": datetime.now().isoformat(),
                    "source": "mock_data"
                }
            else:
                return {"error": f"Forecast data not available for {city}"}
    
    def get_weather_alerts(self, city: str = "New York") -> Dict[str, Any]:
        """Get weather alerts for a city."""
        # Mock alerts
        alerts = []
        if city == "New York":
            alerts = [
                {
                    "type": "Severe Thunderstorm",
                    "description": "Severe thunderstorm warning in effect",
                    "severity": "moderate",
                    "expires": (datetime.now() + timedelta(hours=2)).isoformat()
                }
            ]
        elif city == "London":
            alerts = [
                {
                    "type": "Flood Warning",
                    "description": "Flood warning for low-lying areas",
                    "severity": "high",
                    "expires": (datetime.now() + timedelta(hours=6)).isoformat()
                }
            ]
        
        return {
            "city": city,
            "alerts": alerts,
            "alert_count": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
    
    def convert_temperature(self, temp: float, from_unit: str = "fahrenheit", to_unit: str = "celsius") -> Dict[str, Any]:
        """Convert temperature between units."""
        try:
            if from_unit == "fahrenheit" and to_unit == "celsius":
                converted = (temp - 32) * 5/9
            elif from_unit == "celsius" and to_unit == "fahrenheit":
                converted = temp * 9/5 + 32
            elif from_unit == "celsius" and to_unit == "kelvin":
                converted = temp + 273.15
            elif from_unit == "kelvin" and to_unit == "celsius":
                converted = temp - 273.15
            else:
                return {"error": f"Unsupported conversion: {from_unit} to {to_unit}"}
            
            return {
                "original": {"value": temp, "unit": from_unit},
                "converted": {"value": round(converted, 2), "unit": to_unit},
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Temperature conversion failed: {e}"}
    
    def get_air_quality(self, city: str = "New York") -> Dict[str, Any]:
        """Get air quality information for a city."""
        # Mock air quality data
        aqi_data = {
            "New York": {"aqi": 45, "category": "Good", "pm25": 12, "pm10": 25},
            "London": {"aqi": 65, "category": "Moderate", "pm25": 18, "pm10": 35},
            "Tokyo": {"aqi": 35, "category": "Good", "pm25": 8, "pm10": 20}
        }
        
        if city in aqi_data:
            return {
                "city": city,
                "air_quality": aqi_data[city],
                "timestamp": datetime.now().isoformat(),
                "source": "mock_data"
            }
        else:
            return {"error": f"Air quality data not available for {city}"}
