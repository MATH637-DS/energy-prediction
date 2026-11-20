import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from energy_prediction.config import DATA_DIR, ROOT
from src.models.baseline.neural_network_6h import SimpleNN

BUILDINGS = [
    "classroom",
    "residence",
    "laboratory",
    "recreation",
    "library",
]
INPUT_COLUMNS = [*BUILDINGS, "temperature_c"]
AVAILABILITY_COLUMNS = [
    "published_at",
    "weather_observed_at",
    "weather_received_at",
]
MODEL_ARTIFACTS = [
    ("linear_regression_6h_lookback_24h.joblib", "Linear regression · 6-hour"),
    ("linear_regression_24h_lookback_24h.joblib", "Linear regression · 24-hour"),
    ("neural_network_6h_lookback_24h.pt", "Neural network · 6-hour"),
    ("neural_network_24h_lookback_24h.pt", "Neural network · 24-hour"),
]
MAX_IMPUTED_HISTORY_POINTS = 1
OBSERVATIONS_FILE = DATA_DIR / "observations.jsonl"
ARTIFACTS_DIR = ROOT / "artifacts" / "baseline"
DASHBOARD_FILE = ROOT / "docs" / "data" / "forecasts.json"


def timestamp_text(value: pd.Timestamp | datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def load_observations() -> pd.DataFrame:
    rows = []
    with OBSERVATIONS_FILE.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            observation = json.loads(line)
            weather = observation["weather"]
            rows.append(
                {
                    "timestamp": observation["timestamp"],
                    "published_at": observation["published_at"],
                    "weather_observed_at": weather["observed_at"],
                    "weather_received_at": weather["received_at"],
                    "temperature_c": weather["temperature_c"],
                    "quality": observation["quality"],
                    **observation["demand_kw"],
                }
            )

    if not rows:
        raise ValueError(f"No observations found in {OBSERVATIONS_FILE}")

    frame = pd.DataFrame(rows)
    for column in ["timestamp", *AVAILABILITY_COLUMNS]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
        raise ValueError("Observations must have unique, valid timestamps")

    return frame.set_index("timestamp").sort_index()


def latest_feature_window(
    observations: pd.DataFrame, lookback_hours: int, now: datetime
) -> tuple[pd.DataFrame, pd.Timestamp, int]:
    latest_origin = min(observations.index.max(), pd.Timestamp(now)).floor("h")
    origins = pd.date_range(
        start=observations.index.min().ceil("h"),
        end=latest_origin,
        freq="h",
    )

    for origin in reversed(origins):
        cutoff = origin - pd.Timedelta(minutes=5)
        history_times = pd.date_range(
            end=origin - pd.Timedelta(minutes=15),
            periods=lookback_hours * 4,
            freq="15min",
        )
        history = observations.reindex(history_times).copy()
        has_observation = history["published_at"].notna()
        if not has_observation.any():
            continue

        availability_is_valid = all(
            history.loc[has_observation, column].notna().all()
            and history.loc[has_observation, column].le(cutoff).all()
            for column in AVAILABILITY_COLUMNS
        )
        if not availability_is_valid:
            continue

        quality = history.loc[has_observation, "quality"]
        if not quality.isin(["live", "stale"]).all():
            continue

        missing_input = history[INPUT_COLUMNS].isna().any(axis=1)
        imputed_points = int(missing_input.sum())
        if imputed_points > MAX_IMPUTED_HISTORY_POINTS:
            continue

        if imputed_points:
            history[INPUT_COLUMNS] = history[INPUT_COLUMNS].interpolate(
                method="time", limit_area="inside"
            )
        if history[INPUT_COLUMNS].isna().any().any():
            continue

        history["is_imputed"] = missing_input
        history["quality"] = history["quality"].fillna("stale")
        return history, origin, imputed_points

    raise ValueError(
        "No complete 24-hour observation window is available for model inference"
    )


def feature_row(
    history: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    values: dict[str, float] = {}
    steps = len(history)
    for index, (_, row) in enumerate(history.iterrows()):
        lag_minutes = (steps - index) * 15
        for building in BUILDINGS:
            values[f"x_{building}_lag_{lag_minutes}m"] = float(row[building])
        values[f"x_temperature_c_lag_{lag_minutes}m"] = float(row["temperature_c"])
        values[f"x_is_stale_lag_{lag_minutes}m"] = float(
            row["quality"] == "stale" or row["is_imputed"]
        )

    missing_features = set(feature_columns) - values.keys()
    if missing_features:
        raise ValueError(f"Unrecognized model features: {sorted(missing_features)}")
    ordered_values = [values[column] for column in feature_columns]
    if not np.isfinite(ordered_values).all():
        raise ValueError("Model input contains non-finite values")
    return pd.DataFrame([ordered_values], columns=feature_columns)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix == ".joblib":
        return joblib.load(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def predict(checkpoint: dict[str, Any], features: pd.DataFrame, is_torch: bool):
    if list(features.columns) != checkpoint["feature_columns"]:
        raise ValueError("Model feature columns do not match its saved metadata")

    if not is_torch:
        scaler = checkpoint["scaler"]
        return checkpoint["model"].predict(scaler.transform(features))[0]

    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    scaled = (features.to_numpy(dtype=np.float32) - mean) / scale
    model = SimpleNN(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        output_size=checkpoint["output_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(scaled)).numpy()[0]


def model_evaluation(checkpoint: dict[str, Any]) -> dict[str, Any]:
    per_building = []
    for metric in checkpoint["building_metrics"]:
        per_building.append(
            {
                "building_id": metric["Building"],
                "mae_kw": float(metric["MAE (kW)"]),
                "rmse_kw": float(metric["RMSE (kW)"]),
                "r2": float(metric["R2"]),
            }
        )
    return {
        "test_mae_kw": float(checkpoint["test_mae"]),
        "test_rmse_kw": float(checkpoint["test_rmse"]),
        "test_r2": float(checkpoint["test_r2"]),
        "per_building": per_building,
    }


def export_dashboard() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    observations = load_observations()
    historic, origin, imputed_points = latest_feature_window(
        observations, lookback_hours=24, now=now
    )
    models = []

    for filename, name in MODEL_ARTIFACTS:
        path = ARTIFACTS_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Trained model artifact not found: {path}")
        checkpoint = load_checkpoint(path)
        feature_columns = list(checkpoint["feature_columns"])
        inputs = feature_row(historic, feature_columns)
        horizon_hours = int(checkpoint["horizon_hours"])
        target_columns = list(checkpoint["target_columns"])
        if len(target_columns) != len(BUILDINGS):
            raise ValueError(f"Unexpected target columns in {filename}")

        prediction = predict(checkpoint, inputs, path.suffix == ".pt")
        if not np.isfinite(prediction).all():
            raise ValueError(f"{filename} produced non-finite predictions")

        target_time = origin + timedelta(hours=horizon_hours)
        forecasts = []
        if target_time.to_pydatetime() > now:
            for target_column, demand in zip(target_columns, prediction, strict=True):
                building_id = target_column.removeprefix("y_")
                if building_id not in BUILDINGS:
                    raise ValueError(f"Unexpected model target: {target_column}")
                forecasts.append(
                    {
                        "target_time": timestamp_text(target_time),
                        "building_id": building_id,
                        "horizon_hours": horizon_hours,
                        "prediction_kw": float(demand),
                    }
                )

        models.append(
            {
                "id": path.stem.replace("_lookback_24h", ""),
                "name": name,
                "horizon_hours": horizon_hours,
                "lookback_hours": int(checkpoint["lookback_hours"]),
                "generated_at": timestamp_text(now),
                "origin_time": timestamp_text(origin),
                "input_latest_observation": timestamp_text(historic.index[-1]),
                "imputed_history_points": imputed_points,
                "training_evaluation": model_evaluation(checkpoint),
                "forecasts": forecasts,
            }
        )

    historical_demand = []
    for timestamp, observation in observations.iterrows():
        for building in BUILDINGS:
            demand = observation[building]
            if pd.notna(demand) and np.isfinite(float(demand)):
                historical_demand.append(
                    {
                        "timestamp": timestamp_text(timestamp),
                        "building_id": building,
                        "demand_kw": float(demand),
                        "quality": observation["quality"],
                    }
                )

    latest = observations.iloc[-1]
    dashboard = {
        "updated_at": timestamp_text(now),
        "historical_demand": historical_demand,
        "models": models,
        "accepted_forecasts": [],
        "performance": None,
        "system_status": {
            "latest_collected_reading": {
                "timestamp": timestamp_text(observations.index[-1]),
                "demand_kw": {
                    building: float(latest[building]) for building in BUILDINGS
                },
            },
            "latest_accepted_submission": None,
            "next_submission_deadline": None,
            "errors": [],
        },
    }
    return dashboard


def main() -> None:
    dashboard = export_dashboard()
    DASHBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = DASHBOARD_FILE.with_suffix(".json.tmp")
    temporary_file.write_text(
        json.dumps(dashboard, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_file.replace(DASHBOARD_FILE)
    point_count = sum(len(model["forecasts"]) for model in dashboard["models"])
    print(
        f"Published {len(dashboard['models'])} models and {point_count} forecast "
        f"points to {DASHBOARD_FILE}"
    )


if __name__ == "__main__":
    main()
