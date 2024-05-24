# agent.py
import os
from lam import LAM
from apis.doordash_api import DoorDashAPI
from apis.uber_api import UberAPI
from apis.expedia_api import ExpediaAPI
from apis.plaid_api import PlaidAPI

class Agent(LAM):
    def __init__(self, name):
        super().__init__(name)
        self.doordash_api = DoorDashAPI(api_key=os.getenv('DOORDASH_API_KEY'))
        self.uber_api = UberAPI(api_key=os.getenv('UBER_API_KEY'))
        self.expedia_api = ExpediaAPI(api_key=os.getenv('EXPEDIA_API_KEY'))
        self.plaid_api = PlaidAPI(client_id=os.getenv('PLAID_CLIENT_ID'), secret=os.getenv('PLAID_SECRET'))

    def order_food(self, location):
        try:
            restaurants = self.doordash_api.get_restaurants(location)
            print("Available restaurants:", restaurants)
            # Add logic to place order...
        except Exception as e:
            logger.error(f"Error in order_food: {e}")

    def request_ride(self, start_lat, start_lng, end_lat, end_lng):
        try:
            estimates = self.uber_api.get_ride_estimates(start_lat, start_lng, end_lat, end_lng)
            print("Ride estimates:", estimates)
            # Add logic to request ride...
        except Exception as e:
            logger.error(f"Error in request_ride: {e}")

    def search_and_book_hotel(self, destination, checkin_date, checkout_date):
        try:
            hotels = self.expedia_api.search_hotels(destination, checkin_date, checkout_date)
            print("Available hotels:", hotels)
            # Add logic to book hotel...
        except Exception as e:
            logger.error(f"Error in search_and_book_hotel: {e}")

    def get_bank_accounts(self, user_id, access_token):
        try:
            link_token = self.plaid_api.create_link_token({"id": user_id})
            print("Link token:", link_token)
            accounts = self.plaid_api.get_accounts(access_token)
            print("Bank accounts:", accounts)
            # Add logic to manage accounts...
        except Exception as e:
            logger.error(f"Error in get_bank_accounts: {e}")
