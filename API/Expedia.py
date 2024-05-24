import requests

class ExpediaAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.expedia.com"

    def search_hotels(self, destination, checkin_date, checkout_date, rooms=1, adults=1):
        url = f"{self.base_url}/api/hotels/search"
        params = {
            "destination": destination,
            "checkin_date": checkin_date,
            "checkout_date": checkout_date,
            "rooms": rooms,
            "adults": adults,
            "apiKey": self.api_key
        }
        response = requests.get(url, params=params)
        return response.json()

    def book_hotel(self, booking_details):
        url = f"{self.base_url}/api/hotels/book"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=booking_details)
        return response.json()
