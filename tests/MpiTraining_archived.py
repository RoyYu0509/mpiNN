# test_mpi_pipeline.py
import os
import numpy as np
import pandas as pd
from mpi4py import MPI

# Import your classes
from mpinn.mpiDataDistribution import MPIDD
from mpinn.mlp import MLP, ActivationFunction
from mpinn.mpiMLP import mpiMLP  # make sure your mpiMLP class is in mpiMLP.py

from sklearn.model_selection import train_test_split

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

def count_and_dropna(df, required_cols):
    before = len(df)
    df = df.dropna(subset=required_cols)
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"[Rank {RANK}] Dropped {dropped} rows due to NA values in {required_cols}")
    return df, dropped

def preprocess_nytaxi(raw_path="nytaxi2022.csv", out_train="nytaxi_train.csv", out_test="nytaxi_test.csv", test_size=0.3, random_state=42):
    chunks = []
    for chunk in pd.read_csv(
        raw_path,
        dtype={
            "VendorID": int,
            "tpep_pickup_datetime": str,
            "tpep_dropoff_datetime": str,
            "passenger_count": "Int32",
            "trip_distance": float,
            "RatecodeID": float,
            "PULocationID": float,
            "DOLocationID": float,
            "payment_type": float,
            "extra": float,
        },
        low_memory=False,
        chunksize=1_000_000
    ):
        # drop rows with missing values and count
        required_cols = [
            "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "passenger_count", "trip_distance", "RatecodeID",
            "PULocationID", "DOLocationID", "payment_type", "extra", "total_amount"
        ]

        chunk, dropped = count_and_dropna(chunk, required_cols)

        # compute trip duration
        chunk["pickup"] = pd.to_datetime(chunk["tpep_pickup_datetime"], format="%m/%d/%Y %I:%M:%S %p")
        chunk["dropoff"] = pd.to_datetime(chunk["tpep_dropoff_datetime"], format="%m/%d/%Y %I:%M:%S %p")
        chunk["trip_duration"] = (
            (chunk["dropoff"] - chunk["pickup"]).dt.total_seconds() / 60.0
        )

        chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)

        # Features to use (raw, no normalization)
        feature_cols = [
            "trip_duration", "passenger_count", "trip_distance",
            "RatecodeID", "PULocationID", "DOLocationID",
            "payment_type", "extra"
        ]

        df_out = df[feature_cols + ["total_amount"]]

        # Train/test split
        train_df, test_df = train_test_split(df_out, test_size=test_size, random_state=random_state)

        # Save
        train_df.to_csv(out_train, index=False)
        test_df.to_csv(out_test, index=False)

        return out_train, out_test, feature_cols




def main():
    act_name = "tanh"
    if RANK == 0:
        print("zero")
        train_path, test_path, feature_cols = preprocess_nytaxi(
            raw_path="nytaxi2022.csv",
            out_train="nytaxi_train.csv",
            out_test="nytaxi_test.csv"
        )
    else:
        train_path, test_path, feature_cols = None, None, None

    # Broadcast to all ranks
    train_path = COMM.bcast(train_path, root=0)
    test_path = COMM.bcast(test_path, root=0)
    feature_cols = COMM.bcast(feature_cols, root=0)

    # Build model
    input_size = len(feature_cols)
    hidden_size = 64
    model = MLP(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=1,
        activation=act_name,
        learning_rate=1e-3,
        random_seed=42
    )

    # Train with MPI
    trainer = mpiMLP(model)
    train_losses, val_losses = trainer.mpiSGD(
        file_path=train_path,
        readin_chunksize=5000,
        valid_portion=0.15,
        lr=1e-3,
        epochs=100,
        batch_portion=0.05,
        patience=20,
        lr_decay=0.6,
        report_per=1,
        lr_resch_stepsize=10,
        save_fig=f"results/training_history_{act_name}.png"
    )

    # Test evaluation
    test_df = pd.read_csv(test_path)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["total_amount"].to_numpy(dtype=float).reshape(-1, 1)
    
    df = pd.DataFrame({
    "epoch": np.arange(1, len(train_losses) + 1),  # add epoch index if needed
    "train_loss": train_losses,
    "val_loss": val_losses
    })

    df.to_csv(f"loss_record_{act_name}.csv", index=False)

    test_rmse = trainer.compute_RMSE(X_test, y_test)

    if RANK == 0:
        print("\n========== Results ==========")
        print(f"Train RMSE (last): {np.sqrt(train_losses[-1]):.4f}")
        print(f"Val RMSE   (last): {np.sqrt(val_losses[-1]):.4f}")
        print(f"Test RMSE:        {test_rmse:.4f}")
        print("================================\n")
    

if __name__ == "__main__":
    main()