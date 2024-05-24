import requests

class UberAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.uber.com/v1.2"

    def get_ride_estimates(self, start_lat, start_lng, end_lat, end_lng):
        url = f"{self.base_url}/estimates/price"
        params = {"start_latitude": start_lat, "start_longitude": start_lng, "end_latitude": end_lat, "end_longitude": end_lng}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers, params=params)
        return response.json()

    def request_ride(self, ride_details):
        url = f"{self.base_url}/requests"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=ride_details)
        return response.json()
