# test_mpi_pipeline.py
import os
import numpy as np
import pandas as pd
from mpi4py import MPI
import gc


# Import your classes
from mpinn.mpiDataDistribution import MPIDD
from mpinn.mlp import MLP, ActivationFunction
from mpinn.mpiMLP import mpiMLP  # make sure your mpiMLP class is in mpiMLP.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings("ignore")

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

# ---------------- New Preprocessing Functions ---------------- #
def load_and_clean_data(file_path, chunk_size=100000, outlier_config=None):
    """
    Load and clean data from CSV file in chunks
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Size of chunks to process
        outlier_config: Configuration for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    print(f"[Rank {RANK}] Loading data from {file_path} in chunks of {chunk_size}...")
    
    selected_columns = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "passenger_count", "trip_distance", "RatecodeID",
        "PULocationID", "DOLocationID", "payment_type", "extra", "total_amount"
    ]
    
    chunks = []
    total_original_rows = 0
    total_cleaned_rows = 0
    
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
        if (RANK == 0) & (i % 100 == 0):
            print(f"[Rank {RANK}] Processing chunk {i+1}...")
        
        original_rows = len(chunk)
        total_original_rows += original_rows
        
        chunk = chunk[selected_columns]
        chunk = clean_data_chunk(chunk, outlier_config)
        
        cleaned_rows = len(chunk)
        total_cleaned_rows += cleaned_rows
        
        chunks.append(chunk)
        
        # if RANK == 0:
        #     removed_rows = original_rows - cleaned_rows
        #     removal_percentage = (removed_rows / original_rows) * 100 if original_rows > 0 else 0
        #     print(f"[Rank {RANK}] Chunk {i+1}: Removed {removed_rows} rows ({removal_percentage:.2f}%)")
    
    df = pd.concat(chunks, ignore_index=True)
    
    total_removed = total_original_rows - total_cleaned_rows
    total_removal_percentage = (total_removed / total_original_rows) * 100 if total_original_rows > 0 else 0
    
    print(f"[Rank {RANK}] Final dataset shape: {df.shape}")
    print(f"[Rank {RANK}] Total outliers removed: {total_removed} ({total_removal_percentage:.2f}%)")
    
    return df

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Remove outliers using Interquartile Range (IQR) method
    
    Args:
        df: DataFrame to process
        columns: List of column names to check for outliers
        multiplier: IQR multiplier for outlier threshold (default: 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Remove outliers
            initial_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            removed_count = initial_count - len(df_clean)
            
            # if removed_count > 0:
            #     print(f"[Rank {RANK}] Removed {removed_count} outliers from column '{col}' using IQR method")
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method
    
    Args:
        df: DataFrame to process  
        columns: List of column names to check for outliers
        threshold: Z-score threshold (default: 3)
    
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            
            initial_count = len(df_clean)
            df_clean = df_clean[z_scores <= threshold]
            removed_count = initial_count - len(df_clean)
            
            if removed_count > 0:
                print(f"[Rank {RANK}] Removed {removed_count} outliers from column '{col}' using Z-score method")
    
    return df_clean


def clean_data_chunk(chunk, outlier_config=None):
    """
    Clean a data chunk by removing NaN values, invalid entries, and outliers
    
    Args:
        chunk: DataFrame chunk to clean
        outlier_config: Dictionary with outlier removal configuration
    
    Returns:
        Cleaned DataFrame chunk
    """
    # Set default outlier configuration
    if outlier_config is None:
        outlier_config = {
            "method": "iqr",
            "iqr_multiplier": 1.5,
            "zscore_threshold": 3,
        }
    
    chunk_clean = chunk.dropna()
    chunk_clean = chunk_clean[chunk_clean["trip_distance"] > 0]
    chunk_clean = chunk_clean[chunk_clean["passenger_count"] > 0]
    
    
    # Apply statistical outlier removal based on method
    numerical_cols = ["trip_distance", "total_amount", "extra"]
    
    if outlier_config.get("method") == "iqr":
        multiplier = outlier_config.get("iqr_multiplier", 1.5)
        chunk_clean = remove_outliers_iqr(chunk_clean, numerical_cols, multiplier=multiplier)
    elif outlier_config.get("method") == "zscore":
        threshold = outlier_config.get("zscore_threshold", 3)
        chunk_clean = remove_outliers_zscore(chunk_clean, numerical_cols, threshold=threshold)
    elif outlier_config.get("method") == "both":
        # Apply both methods sequentially
        multiplier = outlier_config.get("iqr_multiplier", 1.5)
        threshold = outlier_config.get("zscore_threshold", 3)
        chunk_clean = remove_outliers_iqr(chunk_clean, numerical_cols, multiplier=multiplier)
        chunk_clean = remove_outliers_zscore(chunk_clean, numerical_cols, threshold=threshold)
    
    return chunk_clean

def normalize_features(df):
    print(f"[Rank {RANK}] Normalizing features...")
    
    numerical_cols = [
        "passenger_count", "trip_distance", "RatecodeID",
        "PULocationID", "DOLocationID", "payment_type", "extra"
    ]
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler

def preprocess_pipeline(raw_path="nytaxi2022.csv", 
                        out_train="nytaxi_train.csv", 
                        out_test="nytaxi_test.csv",
                        test_size=0.3,
                        random_state=42,
                        outlier_removal_method="iqr",
                        iqr_multiplier=1.5,
                        zscore_threshold=3,
                        apply_domain_outliers=True):
    """
    Complete preprocessing pipeline with configurable outlier removal
    
    Args:
        raw_path: Path to raw CSV file
        out_train: Output path for training data
        out_test: Output path for test data
        test_size: Fraction of data for testing
        random_state: Random seed
        outlier_removal_method: Method for outlier removal ("iqr", "zscore", "both", "none")
        iqr_multiplier: Multiplier for IQR method (default: 1.5)
        zscore_threshold: Threshold for Z-score method (default: 3)
        apply_domain_outliers: Whether to apply domain-specific outlier removal
    
    Returns:
        Tuple of (train_path, test_path, feature_cols)
    """
    # Create outlier configuration
    outlier_config = {
        "method": outlier_removal_method,
        "iqr_multiplier": iqr_multiplier,
        "zscore_threshold": zscore_threshold,
        "apply_domain_outliers": apply_domain_outliers
    }
    
    # Load, clean, normalize
    df = load_and_clean_data(raw_path, outlier_config=outlier_config)
    df, scaler = normalize_features(df)
    
    feature_cols = [
        "passenger_count", "trip_distance", "RatecodeID",
        "PULocationID", "DOLocationID", "payment_type", "extra"
    ]
    
    df_out = df[feature_cols + ["total_amount"]]
    
    # Train/test split
    train_df, test_df = train_test_split(df_out, test_size=test_size, random_state=random_state)
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    
    # Save scaler for inference
    with open("preprocessing_objects.pkl", "wb") as f:
        pickle.dump({"scaler": scaler}, f)
    
    return out_train, out_test, feature_cols
# ------------------------------------------------------------- #

def experiment(act_name, batch_portion):
    act_name = act_name
    batch_portion = batch_portion

    if RANK == 0:
        print("[Rank 0] Starting preprocessing pipeline...")
        train_path, test_path, feature_cols = preprocess_pipeline(
            raw_path="nytaxi2022.csv",
            out_train="nytaxi_train.csv",
            out_test="nytaxi_test.csv",
            outlier_removal_method="iqr",
            iqr_multiplier=2.0,
            apply_domain_outliers=True
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

    save_fig = f"results/bat_size{batch_portion}/{act_name}/training_history_{act_name}.png"

    if RANK == 0:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)

    # wait for timing the training process
    COMM.Barrier()
    t0 = MPI.Wtime()

    train_losses, val_losses = trainer.mpiSGD(
        file_path=train_path,
        readin_chunksize=5000,
        valid_portion=0.15,
        lr=4e-3,
        epochs=300,
        batch_portion=batch_portion,
        patience=20,
        lr_decay=0.9,
        report_per=1,
        lr_resch_stepsize=50,
        save_fig=save_fig
    )

    COMM.Barrier()
    t1 = MPI.Wtime()
    local_time = t1 - t0
    train_time = COMM.reduce(local_time, op=MPI.MAX, root=0)

    # Test evaluation
    test_df = pd.read_csv(test_path)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["total_amount"].to_numpy(dtype=float).reshape(-1, 1)
    
    df = pd.DataFrame({
    "epoch": np.arange(1, len(train_losses) + 1),  # add epoch index if needed
    "train_loss": train_losses,
    "val_loss": val_losses
    })

    csv_path = f"results/bat_size{batch_portion}/{act_name}/loss_record_{act_name}.csv"
    if RANK == 0:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    COMM.Barrier()
    df.to_csv(csv_path, index=False)


    test_rmse = trainer.compute_RMSE(X_test, y_test)

    if RANK == 0:
        print("\n========== Results ==========")
        print(f"Train MSE (last): {train_losses[-1]:.4f}")
        print(f"Val MSE   (last): {val_losses[-1]:.4f}")
        print(f"Test RMSE:        {test_rmse:.4f}")
        print("================================\n")


    # log experiment results
    if RANK == 0:
        train_rmse = np.sqrt(train_losses[-1])
        validation_rmse = np.sqrt(val_losses[-1])
        

    # Clean up phase
    # --- CLEANUP (end of experiment iteration) ---
    try:
        import matplotlib.pyplot as plt
        plt.close('all')           # close any figures created when saving plots
    except Exception:
        pass

    # Drop large references
    del trainer, model, train_losses, val_losses, test_df, X_test, y_test, df

    gc.collect()                   # force Python GC

    # calcualte the time cost
    COMM.Barrier()
    
    if RANK == 0:
        return train_time, train_rmse, validation_rmse, test_rmse
    else:
        return None, None, None, None


if __name__ == "__main__":
    experiment("relu", 0.1)
