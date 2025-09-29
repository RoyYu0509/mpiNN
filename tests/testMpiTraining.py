# test_mpi_pipeline.py
import os
import numpy as np
import pandas as pd
from mpi4py import MPI

# Import your classes
from mpinn.mpiDataDistribution import MPIDD
from mpinn.mlp import MLP, ActivationFunction
from mpinn.mpiMLP import mpiMLP  # make sure your mpiMLP class is in mpiMLP.py

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

def make_toy_csvs(train_rows=12000, test_rows=3000, path_train="train.csv", path_test="test.csv", seed=123):
    """Create toy train/test CSVs (rank 0 only). Columns: f1,f2,f3,total_amount"""
    if RANK == 0:
        rng = np.random.default_rng(seed)
        # Features
        f1_tr = rng.integers(0, 1000, size=train_rows)
        f2_tr = rng.integers(0, 1000, size=train_rows)
        f3_tr = rng.integers(0, 3,    size=train_rows)
        # Linear-ish target with noise
        y_tr = 3.0 * f1_tr - 2.0 * f2_tr + 0.5 * f3_tr + rng.normal(0, 10.0, size=train_rows)

        train_df = pd.DataFrame({
            "f1": f1_tr.astype(int),
            "f2": f2_tr.astype(int),
            "f3": f3_tr.astype(int),
            "total_amount": y_tr.astype(float),
        })
        train_df.to_csv(path_train, index=False)
        print(f"[Rank 0] Wrote {path_train} with shape {train_df.shape}", flush=True)

        f1_te = rng.integers(0, 1000, size=test_rows)
        f2_te = rng.integers(0, 1000, size=test_rows)
        f3_te = rng.integers(0, 3,    size=test_rows)
        y_te = 3.0 * f1_te - 2.0 * f2_te + 0.5 * f3_te + rng.normal(0, 10.0, size=test_rows)
        test_df = pd.DataFrame({
            "f1": f1_te.astype(int),
            "f2": f2_te.astype(int),
            "f3": f3_te.astype(int),
            "total_amount": y_te.astype(float),
        })
        test_df.to_csv(path_test, index=False)
        print(f"[Rank 0] Wrote {path_test} with shape {test_df.shape}", flush=True)
    COMM.Barrier()

def load_test_arrays(path_test="test.csv"):
    """Every rank loads the same test set (cheap). MSE computed with reduce stays correct."""
    te = pd.read_csv(path_test)
    X_te = te[["f1", "f2", "f3"]].to_numpy()
    y_te = te["total_amount"].to_numpy().reshape(-1, 1)
    return X_te, y_te

def main():
    # 1) Make toy CSVs (rank 0 writes, all wait)
    make_toy_csvs(train_rows=12000, test_rows=3000)

    # 2) Build model
    input_size = 3
    hidden_size = 32
    model = MLP(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=1,
        activation="relu",
        learning_rate=1e-2,   # not used directly in your mpiMLP update; lr is passed below
        random_seed=42,
    )

    # 3) Train with mpiMLP using MPIDD under the hood
    trainer = mpiMLP(model)
    # Choose a chunksize that yields multiple rounds to exercise the distributor
    readin_chunksize = 3000  # with 12k rows total -> 4 rounds

    # Less aggressive lr decay than default (0.5), to keep training stable
    train_losses, val_losses = trainer.mpiSGD(
        file_path="train.csv",
        readin_chunksize=readin_chunksize,
        valid_portion=0.1,
        lr=1e-6,
        epochs=60000,
        batch_portion=0.2,
        patience=100,
        lr_decay=0.5,
        save_fig = "training_history.png"
    )

    # 4) Evaluate on test set (global MSE computed via reductions inside mpi_compute_MSE_root)
    X_test, y_test = load_test_arrays("train.csv")
    test_mse = trainer.mpi_compute_MSE_root(X_test, y_test)
    test_rmse = trainer.mpi_compute_MSE_root(X_test, y_test)

    # 5) Print final result (rank 0 has the scalar)
    if RANK == 0:
        last_train = train_losses[-1] if len(train_losses) else float("nan")
        last_val = val_losses[-1] if len(val_losses) else float("nan")
        print("\n========== Results ==========", flush=True)
        print(f"Ranks: {SIZE}", flush=True)
        print(f"Last train MSE: {last_train:.6f}", flush=True)
        print(f"Last val   MSE: {last_val:.6f}", flush=True)
        print(f"Final TEST MSE: {test_mse:.6f}", flush=True)
        print(f"Final TEST RMSE: {test_rmse:.6f}", flush=True)
        print("================================\n", flush=True)

if __name__ == "__main__":
    main()
