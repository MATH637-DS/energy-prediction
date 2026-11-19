import json
import os
import httpx as hp
from dotenv import load_dotenv

from energy_prediction.config import DATA_DIR, ROOT

# endpoint path -> output file name (predictions is POST-only, so not here)
ENDPOINTS = {
    "/health": "health",
    "/openapi.json": "openapi",
    "/v1/status": "status",
    "/v1/buildings": "buildings",
    "/v1/observations": "observations",
    "/v1/weather/current": "current",
    "/v1/weather/forecast": "forecast",
    "/v1/submissions": "submissions",
    "/v1/scoreboard": "scoreboard",
}


def load_credentials() -> tuple[str | None, str | None]:
    load_dotenv(ROOT / ".env")
    return os.getenv("API_KEY"), os.getenv("API_BASE_URL")


def main() -> None:
    api_key, base_url = load_credentials()
    if not api_key or not base_url:
        raise SystemExit("API_KEY and API_BASE_URL must be set in .env")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Bearer {api_key}"}
    with hp.Client(base_url=base_url, headers=headers, timeout=10) as client:
        for endpoint, resource in ENDPOINTS.items():
            try:
                response = client.get(endpoint)
                print(f"{response.status_code} {endpoint}")
                response.raise_for_status()
                with open(DATA_DIR / f"{resource}.json", "w", encoding="utf-8") as f:
                    json.dump(response.json(), f, indent=4)
            except (hp.HTTPError, ValueError) as e:
                print(f"Error in Retrieval of {resource}:", e)


if __name__ == "__main__":
    main()
