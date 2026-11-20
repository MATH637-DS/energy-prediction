
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

BUILDINGS = [
    "classroom",
    "residence",
    "laboratory",
    "recreation",
    "library",
]

INTERVAL = pd.Timedelta(minutes=15)
HORIZON = pd.Timedelta(hours=6)
SUBMISSION_OFFSET = pd.Timedelta(minutes=5)

DEMAND_COLUMNS = [f"demand_kw.{b}" for b in BUILDINGS]

INPUT_COLUMNS = DEMAND_COLUMNS + [
    "weather.temperature_c",
    "is_stale",
]

INPUT_NAMES = BUILDINGS + [
    "temperature_c",
    "is_stale",
]

TARGET_COLUMNS = [f"y_{b}" for b in BUILDINGS]


def load_observations(path):
    # Load and flatten the historical JSON data
    with open(path, encoding="utf-8") as f:
        df = pd.json_normalize(json.load(f)["data"])

    # Convert timestamps to UTC datetime objects
    for column in [
        "timestamp",
        "published_at",
        "weather.observed_at",
        "weather.received_at",
    ]:
        df[column] = pd.to_datetime(df[column], utc=True)

    # Validate observation timestamps
    if df["timestamp"].isna().any():
        raise ValueError("Missing observation timestamp")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate observation timestamps")

    if (
        df["timestamp"] != df["timestamp"].dt.floor("15min")
    ).any():
        raise ValueError(
            "Observation timestamps must be on the 15-minute grid"
        )

    # Validate observation quality
    if not df["quality"].isin(["live", "stale"]).all():
        raise ValueError("Unknown or missing observation quality")

    # Binary encoding: live = 0, stale = 1
    df["is_stale"] = df["quality"].eq("stale").astype(int)

    # Sort observations chronologically
    return df.set_index("timestamp").sort_index()


def build_training_data(df, lookback_hours):
    if lookback_hours < 1:
        raise ValueError("Lookback must be at least one hour")

    lookback = pd.Timedelta(hours=lookback_hours)
    n_steps = lookback_hours * 4

    # Create input feature names for each historical reading
    feature_columns = [
        f"x_{name}_lag_{lag}m"
        for lag in range(n_steps * 15, 0, -15)
        for name in INPUT_NAMES
    ]

    # Generate every possible forecast origin
    origins = pd.date_range(
        start=df.index.min() + lookback,
        end=df.index.max() - HORIZON,
        freq=INTERVAL,
    )

    records = []
    skipped_history = 0
    skipped_target = 0

    for origin in origins:
        cutoff = origin - SUBMISSION_OFFSET
        target_time = origin + HORIZON

        # Construct the preceding historical window
        history_times = pd.date_range(
            end=origin - INTERVAL,
            periods=n_steps,
            freq=INTERVAL,
        )

        history = df.reindex(history_times)

        # Allow stale historical observations.
        # Reject missing or unavailable observations.
        valid_history = (
            history["published_at"].le(cutoff).all()
            and history["weather.observed_at"].le(cutoff).all()
            and history["weather.received_at"].le(cutoff).all()
        )

        if not valid_history:
            skipped_history += 1
            continue

        # Extract historical demand, temperature, and stale flags
        x = history[INPUT_COLUMNS].to_numpy(dtype=float)

        # Reject missing or nonfinite input values
        if not np.isfinite(x).all():
            skipped_history += 1
            continue

        # Locate the exact instantaneous six-hour target
        if target_time not in df.index:
            skipped_target += 1
            continue

        target = df.loc[target_time]

        # Exclude unscorable targets
        if (
            target["quality"] != "live"
            or target["scorable"] != True
            or pd.isna(target["published_at"])
        ):
            skipped_target += 1
            continue

        # Extract instantaneous demand for all five buildings
        y = target[DEMAND_COLUMNS].to_numpy(dtype=float)

        # Reject missing or nonfinite target values
        if not np.isfinite(y).all():
            skipped_target += 1
            continue

        # Preserve timestamps for chronological evaluation
        record = {
            "origin_time": origin,
            "submission_time": cutoff,
            "target_time": target_time,
            "target_published_at": target["published_at"],
            "is_hourly_origin": origin.minute == 0,
        }

        # Flatten historical readings into input features
        record.update(zip(feature_columns, x.flatten()))

        # Add the five instantaneous target values
        record.update(zip(TARGET_COLUMNS, y))

        records.append(record)

    # Display preprocessing statistics
    print("Candidate windows:", len(origins))
    print("Skipped due to input data:", skipped_history)
    print("Skipped due to target data:", skipped_target)
    print("Valid training packets:", len(records))
    print("Input features per packet:", len(feature_columns))
    print("Targets per packet:", len(TARGET_COLUMNS))

    if not records:
        raise ValueError(
            "No valid training packets for this lookback"
        )

    return pd.DataFrame(records)


def main():
    # Allow the lookback length to be specified in the terminal
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    # Load the historical observations
    df = load_observations(
        DATA_DIR / "observations.json"
    )

    # Generate sliding-window training packets
    training = build_training_data(
        df,
        args.lookback_hours
    )

    # Create the training data output directory
    output_dir = DATA_DIR / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the processed training dataset
    output_path = output_dir / (
        f"lookback_{args.lookback_hours}h_horizon_6h.csv"
    )

    training.to_csv(output_path, index=False)

    # Count packets containing stale historical observations
    stale_columns = [
        column for column in training.columns
        if column.startswith("x_is_stale_")
    ]

    stale_packets = (
        training[stale_columns].eq(1).any(axis=1).sum()
    )

    print("Packets containing stale inputs:", stale_packets)
    print("Hourly-origin packets:", training["is_hourly_origin"].sum())
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
