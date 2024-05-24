import requests

class PlaidAPI:
    def __init__(self, client_id, secret, environment='sandbox'):
        self.client_id = client_id
        self.secret = secret
        self.base_url = f"https://{environment}.plaid.com"

    def create_link_token(self, user):
        url = f"{self.base_url}/link/token/create"
        headers = {"Content-Type": "application/json"}
        payload = {
            "client_id": self.client_id,
            "secret": self.secret,
            "user": {"client_user_id": user["id"]},
            "client_name": "Your App Name",
            "products": ["auth"],
            "country_codes": ["US"],
            "language": "en"
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()

    def get_accounts(self, access_token):
        url = f"{self.base_url}/accounts/get"
        headers = {"Content-Type": "application/json"}
        payload = {
            "client_id": self.client_id,
            "secret": self.secret,
            "access_token": access_token
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()
