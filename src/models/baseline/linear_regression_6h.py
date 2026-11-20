
# To run:
# uv run python -m src.models.baseline.linear_regression_6h --lookback-hours 24

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.training.split_data import (
    load_saved_splits,
    get_xy,
)


# ==========================================
# MODEL PARAMETERS
# ==========================================

HORIZON_HOURS = 6
OUTPUT_SIZE = 5

ROOT = Path(__file__).resolve().parents[3]


# ==========================================
# MAIN
# ==========================================

def main():

    # --------------------------------------
    # 1. COMMAND-LINE ARGUMENTS
    # --------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Number of historical hours used for prediction"
    )

    args = parser.parse_args()

    lookback_hours = args.lookback_hours

    if lookback_hours < 1:
        raise ValueError("Lookback must be at least one hour")


    # --------------------------------------
    # 2. LOAD SAVED TRAINING AND TESTING DATA
    # --------------------------------------

    train, test = load_saved_splits(
        lookback_hours
    )

    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)

    # Automatically determine input dimensions
    input_size = X_train.shape[1]

    print("Lookback hours:", lookback_hours)
    print("Prediction horizon:", HORIZON_HOURS)

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Check expected dataset dimensions
    if (
        input_size != lookback_hours * 4 * 7
        or y_train.shape[1] != OUTPUT_SIZE
        or list(X_train.columns) != list(X_test.columns)
        or list(y_train.columns) != list(y_test.columns)
    ):
        raise ValueError("Unexpected dataset structure")


    # --------------------------------------
    # 3. STANDARDIZE INPUT FEATURES
    # --------------------------------------

    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # --------------------------------------
    # 4. INITIALIZE THE LINEAR REGRESSION
    # --------------------------------------

    model = LinearRegression(
        fit_intercept=True
    )


    # --------------------------------------
    # 5. TRAIN THE LINEAR REGRESSION
    # --------------------------------------

    print("\nTraining linear regression...")

    model.fit(
        X_train_scaled,
        y_train
    )

    print("Linear regression training complete.")

    print(
        "\nCoefficient shape:",
        model.coef_.shape
    )

    print(
        "Intercept shape:",
        model.intercept_.shape
    )

    print(
        "Total model parameters:",
        model.coef_.size + model.intercept_.size
    )

    print(
        "Feature matrix rank:",
        model.rank_
    )


    # --------------------------------------
    # 6. GENERATE PREDICTIONS
    # --------------------------------------

    train_predictions = model.predict(
        X_train_scaled
    )

    test_predictions = model.predict(
        X_test_scaled
    )


    # --------------------------------------
    # 7. EVALUATE TRAINING PERFORMANCE
    # --------------------------------------

    train_errors = (
        y_train.to_numpy() - train_predictions
    )

    train_mae = np.abs(
        train_errors
    ).mean()

    train_rmse = np.sqrt(
        (train_errors ** 2).mean()
    )

    train_r2 = model.score(
        X_train_scaled,
        y_train
    )

    print("\nTraining performance:")

    print(f"MAE:  {train_mae:.4f} kW")
    print(f"RMSE: {train_rmse:.4f} kW")
    print(f"R2:   {train_r2:.4f}")


    # --------------------------------------
    # 8. EVALUATE TESTING PERFORMANCE
    # --------------------------------------

    test_errors = (
        y_test.to_numpy() - test_predictions
    )

    test_mae = np.abs(
        test_errors
    ).mean()

    test_rmse = np.sqrt(
        (test_errors ** 2).mean()
    )

    test_r2 = model.score(
        X_test_scaled,
        y_test
    )

    print("\nTesting performance:")

    print(f"MAE:  {test_mae:.4f} kW")
    print(f"RMSE: {test_rmse:.4f} kW")
    print(f"R2:   {test_r2:.4f}")


    # --------------------------------------
    # 9. PERFORMANCE BY BUILDING
    # --------------------------------------

    building_mae = np.abs(
        test_errors
    ).mean(axis=0)

    building_rmse = np.sqrt(
        (test_errors ** 2).mean(axis=0)
    )

    building_r2 = r2_score(
        y_test,
        test_predictions,
        multioutput="raw_values"
    )

    results = pd.DataFrame({

        "Building": [
            column.removeprefix("y_")
            for column in y_test.columns
        ],

        "MAE (kW)": building_mae,
        "RMSE (kW)": building_rmse,
        "R2": building_r2,

    })

    print("\nTesting performance by building:")

    print(results.to_string(index=False))


    # --------------------------------------
    # 10. DISPLAY SAMPLE PREDICTIONS
    # --------------------------------------

    predictions_df = pd.DataFrame(
        test_predictions,
        columns=y_test.columns
    )

    print("\nFirst five predictions:")
    print(predictions_df.head().to_string(index=False))

    print("\nFirst five actual values:")
    print(y_test.head().to_string(index=False))


    # --------------------------------------
    # 11. SAVE TRAINED MODEL
    # --------------------------------------

    output_dir = ROOT / "artifacts" / "baseline"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = output_dir / (
        f"linear_regression_6h_lookback_{lookback_hours}h.joblib"
    )

    joblib.dump({

        # Trained model and scaler
        "model": model,
        "scaler": scaler,

        # Regression parameters
        "coefficients": model.coef_,
        "intercepts": model.intercept_,

        # Model dimensions
        "input_size": input_size,
        "output_size": OUTPUT_SIZE,

        # Forecast configuration
        "lookback_hours": lookback_hours,
        "horizon_hours": HORIZON_HOURS,

        # Feature and target ordering
        "feature_columns": list(X_train.columns),
        "target_columns": list(y_train.columns),

        # Training performance
        "train_mae": float(train_mae),
        "train_rmse": float(train_rmse),
        "train_r2": float(train_r2),

        # Testing performance
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),

        # Individual building performance
        "building_metrics": results.to_dict(
            orient="records"
        ),

    }, model_path)

    print("\nModel saved:", model_path)


if __name__ == "__main__":
    main()
