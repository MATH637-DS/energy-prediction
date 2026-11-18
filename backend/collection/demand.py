from dotenv import load_dotenv
import os
import subprocess as sp
import httpx as hp
import json


def load_credentials():
    load_dotenv()
    # Retrieve variables
    api_key = os.getenv("API_KEY")
    db_url = os.environ.get("DATABASE_URL")
    return api_key,db_url

def retrieve_one():
    return

api_key,db_url = load_credentials()

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)