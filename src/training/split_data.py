
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "data" / "training"

TIME_COLUMNS = [
    "origin_time",
    "submission_time",
    "target_time",
    "target_published_at",
]


def load_training_data(lookback_hours=24):
    path = TRAINING_DIR / (
        f"lookback_{lookback_hours}h_horizon_6h.csv"
    )

    df = pd.read_csv(path)

    # Convert metadata timestamps to UTC
    for column in TIME_COLUMNS:
        df[column] = pd.to_datetime(
            df[column], utc=True
        )

    # Sort chronologically
    return df.sort_values(
        "origin_time"
    ).reset_index(drop=True)


def split_data(df, train_ratio=0.8):
    if not 0 < train_ratio < 1:
        raise ValueError("Invalid training ratio")

    df = df.sort_values("origin_time").copy()

    # Identify hourly forecast origins
    hourly = df[
        df["origin_time"].dt.minute.eq(0)
    ]

    # Determine the chronological split boundary
    split_index = int(len(hourly) * train_ratio)

    if not 0 < split_index < len(hourly):
        raise ValueError("Insufficient data for splitting")

    test_start = hourly.iloc[
        split_index
    ]["submission_time"]

    # Training: all eligible 15-minute windows
    # whose targets are available before testing
    train = df[
        (df["submission_time"] < test_start)
        & (df["target_published_at"] <= test_start)
    ].copy()

    # Testing: hourly forecast origins only
    test = hourly[
        hourly["submission_time"] >= test_start
    ].copy()

    if train.empty or test.empty:
        raise ValueError("Training or testing set is empty")

    return train, test


def get_xy(df):
    # Separate input features and target variables
    X = df.filter(regex="^x_")
    y = df.filter(regex="^y_")

    return X, y


def save_splits(train, test, lookback_hours, overwrite=False):
    # Create an output directory for this lookback
    output_dir = TRAINING_DIR / (
        f"lookback_{lookback_hours}h_horizon_6h"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    # Prevent accidental replacement of existing splits
    if not overwrite and (
        train_path.exists() or test_path.exists()
    ):
        raise FileExistsError(
            "Saved splits already exist. "
            "Use --overwrite to replace them."
        )

    # Save the complete datasets, including metadata
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print("\nSaved training dataset:", train_path)
    print("Saved testing dataset:", test_path)


def load_saved_splits(lookback_hours=24):
    # Load previously saved training and testing datasets
    output_dir = TRAINING_DIR / (
        f"lookback_{lookback_hours}h_horizon_6h"
    )

    train = pd.read_csv(output_dir / "train.csv")
    test = pd.read_csv(output_dir / "test.csv")

    # Restore metadata timestamps
    for df in [train, test]:
        for column in TIME_COLUMNS:
            df[column] = pd.to_datetime(
                df[column], utc=True
            )

    return train, test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    # Load the processed dataset
    df = load_training_data(args.lookback_hours)

    # Split chronologically
    train, test = split_data(
        df,
        args.train_ratio,
    )

    print("Total available packets:", len(df))

    # Display statistics for both datasets
    for name, dataset in [
        ("Training", train),
        ("Testing", test),
    ]:
        X, y = get_xy(dataset)

        print(f"\n{name}")
        print("Samples:", len(dataset))
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print(
            "First origin:",
            dataset["origin_time"].min()
        )
        print(
            "Last origin:",
            dataset["origin_time"].max()
        )
        print(
            "Latest target publication:",
            dataset["target_published_at"].max()
        )

    print(
        "\nUnused packets:",
        len(df) - len(train) - len(test)
    )

    # Save training and testing datasets
    save_splits(
        train,
        test,
        args.lookback_hours,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
