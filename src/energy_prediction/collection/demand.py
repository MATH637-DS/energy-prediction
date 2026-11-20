
import json
import os
import time
import httpx as hp
from dotenv import load_dotenv

from energy_prediction.config import DATA_DIR, ROOT


# Endpoint path -> output file name
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


def get_all_observations(client: hp.Client) -> dict:
    observations = []
    cursor = 0

    while True:
        response = client.get(
            "/v1/observations",
            params={"after": cursor, "limit": 96}
        )

        # Handle API rate limits
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 60))
            print(f"Rate limit reached. Waiting {wait} seconds.")
            time.sleep(wait)
            continue

        response.raise_for_status()
        page = response.json()

        observations.extend(page["data"])

        print(f"Retrieved {len(observations)} observations")

        # Stop when all available observations have been retrieved
        if not page["has_more"]:
            break

        next_cursor = page["next_cursor"]

        if next_cursor is None or next_cursor <= cursor:
            raise ValueError("Pagination cursor did not advance")

        cursor = next_cursor

        # Respect the API request limit
        time.sleep(2.1)

    return {
        "data": observations,
        "next_cursor": page["next_cursor"],
        "has_more": False
    }


def main() -> None:
    api_key, base_url = load_credentials()

    if not api_key or not base_url:
        raise SystemExit(
            "API_KEY and API_BASE_URL must be set in .env"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Bearer {api_key}"}

    with hp.Client(
        base_url=base_url,
        headers=headers,
        timeout=10
    ) as client:

        for endpoint, resource in ENDPOINTS.items():
            try:
                if endpoint == "/v1/observations":
                    data = get_all_observations(client)
                else:
                    response = client.get(endpoint)
                    print(f"{response.status_code} {endpoint}")
                    response.raise_for_status()
                    data = response.json()

                with open(
                    DATA_DIR / f"{resource}.json",
                    "w",
                    encoding="utf-8"
                ) as f:
                    json.dump(data, f, indent=4)

                print(f"Saved {resource}.json")

            except (hp.HTTPError, ValueError) as e:
                print(f"Error in Retrieval of {resource}:", e)


if __name__ == "__main__":
    main()
