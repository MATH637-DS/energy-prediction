
# To run:
# uv run python -m src.models.baseline.neural_network_24h --lookback-hours 24

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.training.split_data import (
    load_saved_splits,
    get_xy,
)


# ==========================================
# MODEL PARAMETERS
# ==========================================

HORIZON_HOURS = 24
HIDDEN_SIZE = 32
OUTPUT_SIZE = 5
LEARNING_RATE = 0.01
NUM_EPOCHS = 1000
RANDOM_SEED = 0

ROOT = Path(__file__).resolve().parents[3]


# ==========================================
# NEURAL NETWORK ARCHITECTURE
# ==========================================

class SimpleNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_size,
            hidden_size
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


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
        lookback_hours,
        horizon_hours=HORIZON_HOURS
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

    # Verify that targets match the prediction horizon
    for dataset in [train, test]:

        actual_horizon = (
            dataset["target_time"]
            - dataset["origin_time"]
        )

        if not actual_horizon.eq(
            pd.Timedelta(hours=HORIZON_HOURS)
        ).all():
            raise ValueError("Incorrect prediction horizon")


    # --------------------------------------
    # 3. STANDARDIZE INPUT FEATURES
    # --------------------------------------

    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # --------------------------------------
    # 4. CONVERT DATA TO PYTORCH TENSORS
    # --------------------------------------

    X_train_tensor = torch.FloatTensor(
        X_train_scaled
    )

    X_test_tensor = torch.FloatTensor(
        X_test_scaled
    )

    y_train_tensor = torch.FloatTensor(
        y_train.to_numpy()
    )


    # --------------------------------------
    # 5. INITIALIZE THE NEURAL NETWORK
    # --------------------------------------

    torch.manual_seed(RANDOM_SEED)

    model = SimpleNN(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nModel architecture:")
    print(model)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "\nTrainable parameters:",
        trainable_parameters
    )


    # --------------------------------------
    # 6. TRAIN THE NEURAL NETWORK
    # --------------------------------------

    print("\nTraining neural network...")

    for epoch in range(NUM_EPOCHS):

        model.train()

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)

        # Calculate mean squared error
        loss = criterion(
            outputs,
            y_train_tensor
        )

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        if (epoch + 1) % 100 == 0:

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
                f"Loss: {loss.item():.4f}"
            )


    # --------------------------------------
    # 7. GENERATE PREDICTIONS
    # --------------------------------------

    model.eval()

    with torch.no_grad():

        train_predictions = model(
            X_train_tensor
        ).numpy()

        test_predictions = model(
            X_test_tensor
        ).numpy()


    # --------------------------------------
    # 8. EVALUATE TRAINING PERFORMANCE
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

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    print("\nTraining performance:")

    print(f"MAE:  {train_mae:.4f} kW")
    print(f"RMSE: {train_rmse:.4f} kW")
    print(f"R2:   {train_r2:.4f}")


    # --------------------------------------
    # 9. EVALUATE TESTING PERFORMANCE
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

    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    print("\nTesting performance:")

    print(f"MAE:  {test_mae:.4f} kW")
    print(f"RMSE: {test_rmse:.4f} kW")
    print(f"R2:   {test_r2:.4f}")


    # --------------------------------------
    # 10. PERFORMANCE BY BUILDING
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
    # 11. DISPLAY SAMPLE PREDICTIONS
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
    # 12. SAVE TRAINED MODEL
    # --------------------------------------

    output_dir = ROOT / "artifacts" / "baseline"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = output_dir / (
        f"neural_network_{HORIZON_HOURS}h"
        f"_lookback_{lookback_hours}h.pt"
    )

    torch.save({

        # Trained model parameters
        "model_state_dict": model.state_dict(),

        # Model dimensions
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "output_size": OUTPUT_SIZE,

        # Training hyperparameters
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "random_seed": RANDOM_SEED,

        # Forecast configuration
        "lookback_hours": lookback_hours,
        "horizon_hours": HORIZON_HOURS,

        # Feature and target ordering
        "feature_columns": list(X_train.columns),
        "target_columns": list(y_train.columns),

        # Standardization parameters
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),

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
