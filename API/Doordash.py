import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random


class DoorDashAPI:
    """Enhanced DoorDash API integration with realistic functionality."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.doordash.com"
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        # Mock data for demonstration when API key is not available
        self.mock_restaurants = [
            {
                "id": "rest_001",
                "name": "Pizza Palace",
                "cuisine": "Italian",
                "rating": 4.5,
                "delivery_time": "25-35 min",
                "delivery_fee": "$2.99",
                "minimum_order": "$15.00",
                "address": "123 Main St, Downtown",
                "phone": "+1-555-0123",
                "hours": "11:00 AM - 11:00 PM",
                "popular_items": ["Margherita Pizza", "Pepperoni Pizza", "Garlic Bread"]
            },
            {
                "id": "rest_002", 
                "name": "Burger Barn",
                "cuisine": "American",
                "rating": 4.2,
                "delivery_time": "20-30 min",
                "delivery_fee": "$1.99",
                "minimum_order": "$12.00",
                "address": "456 Oak Ave, Midtown",
                "phone": "+1-555-0456",
                "hours": "10:00 AM - 10:00 PM",
                "popular_items": ["Classic Burger", "Cheese Fries", "Milkshake"]
            },
            {
                "id": "rest_003",
                "name": "Sushi Express",
                "cuisine": "Japanese",
                "rating": 4.7,
                "delivery_time": "30-45 min",
                "delivery_fee": "$3.99",
                "minimum_order": "$20.00",
                "address": "789 Pine St, Uptown",
                "phone": "+1-555-0789",
                "hours": "12:00 PM - 10:00 PM",
                "popular_items": ["California Roll", "Salmon Nigiri", "Miso Soup"]
            }
        ]
        
        self.mock_menu_items = {
            "rest_001": [
                {"id": "item_001", "name": "Margherita Pizza", "price": "$18.99", "description": "Fresh mozzarella, tomato sauce, basil"},
                {"id": "item_002", "name": "Pepperoni Pizza", "price": "$20.99", "description": "Pepperoni, mozzarella, tomato sauce"},
                {"id": "item_003", "name": "Garlic Bread", "price": "$6.99", "description": "Crispy bread with garlic butter and herbs"}
            ],
            "rest_002": [
                {"id": "item_004", "name": "Classic Burger", "price": "$12.99", "description": "Beef patty, lettuce, tomato, onion"},
                {"id": "item_005", "name": "Cheese Fries", "price": "$8.99", "description": "Crispy fries topped with melted cheese"},
                {"id": "item_006", "name": "Milkshake", "price": "$5.99", "description": "Vanilla, chocolate, or strawberry"}
            ],
            "rest_003": [
                {"id": "item_007", "name": "California Roll", "price": "$14.99", "description": "Crab, avocado, cucumber"},
                {"id": "item_008", "name": "Salmon Nigiri", "price": "$16.99", "description": "Fresh salmon over seasoned rice"},
                {"id": "item_009", "name": "Miso Soup", "price": "$4.99", "description": "Traditional Japanese soup with tofu"}
            ]
        }

    def get_restaurants(self, location: str, cuisine: Optional[str] = None, 
                       max_delivery_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get restaurants in a specific location with optional filters."""
        try:
            if self.api_key:
                # Real API call
                params = {"location": location}
                if cuisine:
                    params["cuisine"] = cuisine
                if max_delivery_time:
                    params["max_delivery_time"] = max_delivery_time
                
                response = self.session.get(f"{self.base_url}/v2/restaurant", params=params)
                response.raise_for_status()
                return response.json()
            else:
                # Mock response
                restaurants = self.mock_restaurants.copy()
                
                # Apply filters
                if cuisine:
                    restaurants = [r for r in restaurants if cuisine.lower() in r["cuisine"].lower()]
                
                if max_delivery_time:
                    # Parse delivery time and filter
                    filtered_restaurants = []
                    for restaurant in restaurants:
                        time_range = restaurant["delivery_time"]
                        max_time = int(time_range.split("-")[1].split()[0])
                        if max_time <= max_delivery_time:
                            filtered_restaurants.append(restaurant)
                    restaurants = filtered_restaurants
                
                return restaurants
                
        except requests.RequestException as e:
            print(f"Error fetching restaurants: {e}")
            return self.mock_restaurants if not self.api_key else []

    def get_restaurant_details(self, restaurant_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific restaurant."""
        try:
            if self.api_key:
                response = self.session.get(f"{self.base_url}/v2/restaurant/{restaurant_id}")
                response.raise_for_status()
                return response.json()
            else:
                # Mock response
                restaurant = next((r for r in self.mock_restaurants if r["id"] == restaurant_id), None)
                if restaurant:
                    restaurant["menu"] = self.mock_menu_items.get(restaurant_id, [])
                return restaurant
                
        except requests.RequestException as e:
            print(f"Error fetching restaurant details: {e}")
            return None

    def get_menu(self, restaurant_id: str) -> List[Dict[str, Any]]:
        """Get menu items for a specific restaurant."""
        try:
            if self.api_key:
                response = self.session.get(f"{self.base_url}/v2/restaurant/{restaurant_id}/menu")
                response.raise_for_status()
                return response.json()
            else:
                # Mock response
                return self.mock_menu_items.get(restaurant_id, [])
                
        except requests.RequestException as e:
            print(f"Error fetching menu: {e}")
            return []

    def place_order(self, order_details: Dict[str, Any]) -> Dict[str, Any]:
        """Place a food order."""
        try:
            if self.api_key:
                response = self.session.post(f"{self.base_url}/v2/order", json=order_details)
                response.raise_for_status()
                return response.json()
            else:
                # Mock order response
                order_id = f"order_{random.randint(100000, 999999)}"
                estimated_delivery = datetime.now() + timedelta(minutes=random.randint(25, 45))
                
                return {
                    "order_id": order_id,
                    "status": "confirmed",
                    "estimated_delivery": estimated_delivery.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_amount": order_details.get("total_amount", "$0.00"),
                    "delivery_address": order_details.get("delivery_address", "Unknown"),
                    "restaurant_name": order_details.get("restaurant_name", "Unknown"),
                    "items": order_details.get("items", []),
                    "message": "Mock order placed successfully! This is a demonstration."
                }
                
        except requests.RequestException as e:
            print(f"Error placing order: {e}")
            return {"error": f"Failed to place order: {e}"}

    def track_order(self, order_id: str) -> Dict[str, Any]:
        """Track the status of an order."""
        try:
            if self.api_key:
                response = self.session.get(f"{self.base_url}/v2/order/{order_id}/status")
                response.raise_for_status()
                return response.json()
            else:
                # Mock tracking response
                statuses = ["confirmed", "preparing", "ready_for_pickup", "out_for_delivery", "delivered"]
                current_status = random.choice(statuses)
                
                return {
                    "order_id": order_id,
                    "status": current_status,
                    "estimated_delivery": (datetime.now() + timedelta(minutes=random.randint(5, 30))).strftime("%Y-%m-%d %H:%M:%S"),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": "Mock order tracking - this is a demonstration."
                }
                
        except requests.RequestException as e:
            print(f"Error tracking order: {e}")
            return {"error": f"Failed to track order: {e}"}

    def get_delivery_estimate(self, restaurant_id: str, delivery_address: str) -> Dict[str, Any]:
        """Get delivery time and fee estimate."""
        try:
            if self.api_key:
                params = {
                    "restaurant_id": restaurant_id,
                    "delivery_address": delivery_address
                }
                response = self.session.get(f"{self.base_url}/v2/delivery/estimate", params=params)
                response.raise_for_status()
                return response.json()
            else:
                # Mock estimate
                delivery_time = random.randint(20, 45)
                delivery_fee = random.choice([0.99, 1.99, 2.99, 3.99])
                
                return {
                    "estimated_delivery_time": f"{delivery_time} minutes",
                    "delivery_fee": f"${delivery_fee:.2f}",
                    "minimum_order": "$12.00",
                    "message": "Mock delivery estimate - this is a demonstration."
                }
                
        except requests.RequestException as e:
            print(f"Error getting delivery estimate: {e}")
            return {"error": f"Failed to get delivery estimate: {e}"}

    def search_restaurants(self, query: str, location: str) -> List[Dict[str, Any]]:
        """Search for restaurants by name or cuisine."""
        try:
            if self.api_key:
                params = {"q": query, "location": location}
                response = self.session.get(f"{self.base_url}/v2/restaurant/search", params=params)
                response.raise_for_status()
                return response.json()
            else:
                # Mock search
                query_lower = query.lower()
                results = []
                
                for restaurant in self.mock_restaurants:
                    if (query_lower in restaurant["name"].lower() or 
                        query_lower in restaurant["cuisine"].lower() or
                        any(query_lower in item.lower() for item in restaurant["popular_items"])):
                        results.append(restaurant)
                
                return results
                
        except requests.RequestException as e:
            print(f"Error searching restaurants: {e}")
            return []
