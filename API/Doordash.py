import requests

class DoorDashAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.doordash.com"

    def get_restaurants(self, location):
        url = f"{self.base_url}/v2/restaurant"
        params = {"location": location}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers, params=params)
        return response.json()

    def place_order(self, order_details):
        url = f"{self.base_url}/v2/order"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=order_details)
        return response.json()
