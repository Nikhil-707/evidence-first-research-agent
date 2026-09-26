import os

import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise SystemExit("GROQ_API_KEY was not found in .env")

response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=15,
)

print("Status Code:", response.status_code)
if response.ok:
    models = [model["id"] for model in response.json().get("data", [])]
    print("Active models:", models)
else:
    print("Error Output:", response.text)