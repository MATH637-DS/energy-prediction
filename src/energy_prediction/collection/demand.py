import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
import httpx as hp
from dotenv import load_dotenv
from energy_prediction.config import DATA_DIR, ROOT

OBS_FILE = DATA_DIR / "observations.jsonl"
FORECAST_FILE = DATA_DIR / "weather_forecasts.jsonl"
STATE_FILE = DATA_DIR / "collection_state.json"
PAGE_LIMIT = 96
REQUEST_SPACING = 2.1
MAX_RETRIES = 5


def load_credentials():
    """
    Returns api key and url as strings
    """
    load_dotenv(ROOT / ".env")
    key = os.getenv("API_KEY")
    url = os.getenv("API_BASE_URL")
    return key, url


def load_state() -> dict[str, Any]:
    """
    FOR ASYNCH RUN
    returns cursor, final timestamp, and last forecast as of value
    of the final observation of the last batch (for live run)
    """
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"cursor": 0, "first_timestamp": None, "last_forecast_as_of": None}


def save_state(state):
    """
    FOR ASYNCH RUN
    saves the cursor, final timestamp and last forcast as of value of the
    final observation in the batch 
    """
    STATE_FILE.write_text(json.dumps(state))


def append_jsonl(path, obs):
    """
    Appends a batch of observations to the
    each row in rows is one observation
    """
    with open(path, "a", encoding="utf-8") as f:
        for ob in obs:
            f.write(json.dumps(ob) + "\n")


def get(client, path, params = None):
    """webfetch with rate-limit pacing and retries on network errors."""
    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_SPACING)  # avoids over requesting
        try:
            r = client.get(path, params=params)
        except hp.TransportError as e:
            print(f"network error on {path}: {e}; retrying")
            time.sleep(2**attempt)
            continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 60))
            print("429 rate limited")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(2**attempt)
            continue
        return r
    raise RuntimeError(f"giving up on {path} after {MAX_RETRIES} attempts")


def collect_observations(client, state):

    total = 0
    while True:
        request = get(
            client, "/v1/observations", {"after": state["cursor"], "limit": PAGE_LIMIT}
        )
        request.raise_for_status()
        page = request.json()
        rows = page["data"]
        if rows:
            append_jsonl(OBS_FILE, rows)
            state["first_timestamp"] = state["first_timestamp"] or rows[0]["timestamp"]
            total += len(rows)
        state["cursor"] = page["next_cursor"]
        save_state(state)  # save after each page so an interruption loses nothing
        if not page["has_more"]:
            break
    print(f"observations: +{total} rows, cursor={state['cursor']}")


def parse_ts(s):
    """
    datetime parser
    """
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def collect_forecasts(client, state):
    """Fetch the forecast available at each whole UTC hour since the first observation."""
    if state["last_forecast_as_of"]:
        as_of = parse_ts(state["last_forecast_as_of"]) + timedelta(hours=1)
    elif state["first_timestamp"]:
        as_of = parse_ts(state["first_timestamp"]).replace(
            minute=0, second=0, microsecond=0
        )
    else:
        return
    now = datetime.now(UTC)
    total = 0
    while as_of <= now:
        stamp = as_of.strftime("%Y-%m-%dT%H:%M:%SZ")
        r = get(client, "/v1/weather/forecast", {"as_of": stamp})
        if r.status_code == 200:
            append_jsonl(FORECAST_FILE, [r.json()])
            total += 1
        elif r.status_code != 404:  # 404 = no forecast collected yet at that time
            r.raise_for_status()
        state["last_forecast_as_of"] = stamp
        save_state(state)
        as_of += timedelta(hours=1)
    print(f"weather forecasts: +{total} snapshots")


def main():
    api_key, base_url = load_credentials()
    if not api_key or not base_url:
        raise SystemExit("API_KEY and API_BASE_URL must be set in .env")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    headers = {"Authorization": f"Bearer {api_key}"}
    with hp.Client(base_url=base_url, headers=headers, timeout=30) as client:
        request = get(client, "/v1/buildings")
        request.raise_for_status()
        (DATA_DIR / "buildings.json").write_text(json.dumps(request.json(), indent=2))
        collect_observations(client, state)
        collect_forecasts(client, state)


if __name__ == "__main__":
    main()
