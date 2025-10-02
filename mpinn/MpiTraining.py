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
# GPT: Remove the outliers in terms of percentage
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


def remove_outliers_quantile(df, columns, total_drop_frac=0.05, combine_logic="all"):
    """
    Keep only the central (1 - total_drop_frac) mass for each column.
    E.g., total_drop_frac=0.01 -> drop ~0.5% low + 0.5% high per column.
    combine_logic:
      - "all": keep row only if it's inside bounds for ALL columns (strict; recommended)
      - "any": keep row if it's inside bounds for AT LEAST one column (looser)
    """
    if total_drop_frac <= 0:
        return df

    tail = total_drop_frac / 2.0
    inside_masks = []
    for col in columns:
        if col not in df.columns:
            continue
        lo = df[col].quantile(tail)
        hi = df[col].quantile(1 - tail)
        inside_masks.append((df[col] >= lo) & (df[col] <= hi))

    if not inside_masks:
        return df

    if combine_logic == "any":
        keep_mask = np.logical_or.reduce(inside_masks)
    else:  # "all"
        keep_mask = np.logical_and.reduce(inside_masks)

    return df[keep_mask]


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

def load_and_clean_data_basic(file_path, chunk_size=100000):
    """
    Load data with basic cleaning only (no outlier removal)
    Used for proper train/test split before outlier removal
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Size of chunks to process
    
    Returns:
        DataFrame with basic cleaning applied
    """
    print(f"[Rank {RANK}] Loading data from {file_path} in chunks of {chunk_size}...")
    
    selected_columns = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "passenger_count", "trip_distance", "RatecodeID",
        "PULocationID", "DOLocationID", "payment_type", "extra", "total_amount"
    ]
    
    chunks = []
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
        if (RANK == 0) & (i % 100 == 0):
            print(f"[Rank {RANK}] Processing chunk {i+1}...")
        
        chunk = chunk[selected_columns]
        # Only basic cleaning - remove NaN and basic invalid values
        chunk_clean = chunk.dropna()
        chunk_clean = chunk_clean[chunk_clean["trip_distance"] > 0]
        chunk_clean = chunk_clean[chunk_clean["passenger_count"] > 0]
        
        chunks.append(chunk_clean)
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"[Rank {RANK}] Dataset after basic cleaning: {df.shape}")
    return df

def remove_outliers_from_dataframe(df, outlier_config):
    """
    Remove outliers from a DataFrame using the specified configuration
    
    Args:
        df: DataFrame to clean
        outlier_config: Dictionary with outlier removal configuration
    
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    numerical_cols = ["trip_distance", "total_amount", "extra", "passenger_count"]
    
    method = outlier_config.get("method", "iqr")
    
    if method == "iqr":
        multiplier = outlier_config.get("iqr_multiplier", 1.5)
        df_clean = remove_outliers_iqr(df_clean, numerical_cols, multiplier=multiplier)
    elif method == "zscore":
        threshold = outlier_config.get("zscore_threshold", 3)
        df_clean = remove_outliers_zscore(df_clean, numerical_cols, threshold=threshold)
    elif method == "both":
        # Apply both methods sequentially
        multiplier = outlier_config.get("iqr_multiplier", 1.5)
        threshold = outlier_config.get("zscore_threshold", 3)
        df_clean = remove_outliers_iqr(df_clean, numerical_cols, multiplier=multiplier)
        df_clean = remove_outliers_zscore(df_clean, numerical_cols, threshold=threshold)
    elif method == "quantile":
        qfrac = outlier_config.get("quantile_drop_frac", 0.01)   # ~1% total removed
        logic = outlier_config.get("combine_logic", "all")
        df_clean = remove_outliers_quantile(df_clean, numerical_cols,
                                            total_drop_frac=qfrac,
                                            combine_logic=logic)
    elif method == "none":
        # No statistical outlier removal, return as is
        pass
    
    return df_clean

def preprocess_dataframe(df, scaler, outlier_config, feature_cols, fit_scaler=False):
    """
    Apply preprocessing to a DataFrame including outlier removal and normalization
    
    Args:
        df: DataFrame to preprocess
        scaler: StandardScaler object (fitted if fit_scaler=False)
        outlier_config: Dictionary with outlier removal configuration
        feature_cols: List of feature column names
        fit_scaler: Whether to fit the scaler on this data (True for train, False for test)
    
    Returns:
        Tuple of (preprocessed_df, scaler)
    """
    print(f"[Rank {RANK}] Preprocessing dataframe with shape: {df.shape}")
    
    # Step 1: Remove outliers
    df_clean = remove_outliers_from_dataframe(df, outlier_config)
    outliers_removed = len(df) - len(df_clean)
    print(f"[Rank {RANK}] Removed {outliers_removed} outliers ({(outliers_removed/len(df)*100):.2f}%)")
    
    # Step 2: Normalize features
    df_preprocessed = df_clean.copy()
    
    if fit_scaler:
        # Fit scaler on training data
        print(f"[Rank {RANK}] Fitting scaler on training data...")
        scaler = StandardScaler()
        df_preprocessed[feature_cols] = scaler.fit_transform(df_clean[feature_cols])
    else:
        # Use existing fitted scaler for test data
        print(f"[Rank {RANK}] Applying fitted scaler to data...")
        df_preprocessed[feature_cols] = scaler.transform(df_clean[feature_cols])
    
    print(f"[Rank {RANK}] Final preprocessed shape: {df_preprocessed.shape}")
    return df_preprocessed, scaler

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
    Complete preprocessing pipeline: Split first, then preprocess both train and test sets
    
    Pipeline Steps:
    1. Load raw data with basic cleaning only
    2. Split into train/test sets
    3. Remove outliers from both train and test sets
    4. Normalize features (fit scaler on train, apply to both)
    5. Save processed datasets
    
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
    print(f"[Rank {RANK}] Starting preprocessing pipeline...")
    print()
    # Step 1: Load raw data with basic cleaning only (no outlier removal yet)
    print(f"[Rank {RANK}] Step 1: Loading raw data with basic cleaning...")
    print()
    df_raw = load_and_clean_data_basic(raw_path)
    
    feature_cols = [
        "passenger_count", "trip_distance", "RatecodeID", 
        "PULocationID", "DOLocationID", "payment_type", "extra"
    ]
    
    df_features = df_raw[feature_cols + ["total_amount"]]
    
    # Step 2: Split into train/test BEFORE preprocessing
    print()
    print(f"[Rank {RANK}] Step 2: Splitting data into train/test sets...")
    print()
    train_df_raw, test_df_raw = train_test_split(df_features, test_size=test_size, random_state=random_state)
    
    print(f"[Rank {RANK}] Raw training set shape: {train_df_raw.shape}")
    print(f"[Rank {RANK}] Raw test set shape: {test_df_raw.shape}")
    print()
    
    # Create outlier configuration
    outlier_config = {
        "method": outlier_removal_method,   # set to "quantile"
        "iqr_multiplier": iqr_multiplier,
        "zscore_threshold": zscore_threshold,
        "apply_domain_outliers": apply_domain_outliers,
        "quantile_drop_frac": 0.04,   
        "combine_logic": "all",       
    }

    
    # Step 3: Preprocess training set (fit scaler)
    print(f"[Rank {RANK}] Step 3: Preprocessing training set...")
    print()
    train_df_processed, scaler = preprocess_dataframe(
        train_df_raw, None, outlier_config, feature_cols, fit_scaler=True
    )
    
    # Step 4: Preprocess test set (use fitted scaler)
    print()
    print(f"[Rank {RANK}] Step 4: Preprocessing test set...")
    print()
    test_df_processed, _ = preprocess_dataframe(
        test_df_raw, scaler, outlier_config, feature_cols, fit_scaler=False
    )
    
    # Step 5: Save processed datasets
    print()
    print(f"[Rank {RANK}] Step 5: Saving processed datasets...")
    print()
    train_df_processed.to_csv(out_train, index=False)
    test_df_processed.to_csv(out_test, index=False)
    
    # Save preprocessing objects for inference
    with open("preprocessing_objects.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "outlier_config": outlier_config}, f)
    
    print(f"[Rank {RANK}] Training data saved to: {out_train}")
    print(f"[Rank {RANK}] Test data saved to: {out_test}")

    
    # Print final summary
    if RANK == 0:
        print()
        print(f"\n[Rank {RANK}] === PREPROCESSING SUMMARY ===")
        print(f"Original dataset: {df_raw.shape}")
        print(f"Raw train: {train_df_raw.shape} -> Processed train: {train_df_processed.shape}")
        print(f"Raw test: {test_df_raw.shape} -> Processed test: {test_df_processed.shape}")
        print()
        train_reduction = (len(train_df_raw) - len(train_df_processed)) / len(train_df_raw) * 100
        test_reduction = (len(test_df_raw) - len(test_df_processed)) / len(test_df_raw) * 100
        
        print(f"Train outliers removed: {train_reduction:.2f}%")
        print(f"Test outliers removed: {test_reduction:.2f}%")
        print(f"==============================\n")
        print()
    return out_train, out_test, feature_cols
# ------------------------------------------------------------- #

def experiment(act_name, bat_size, proc_num):
    act_name = act_name
    bat_size = bat_size
    proc_num = proc_num

    if RANK == 0:
        # print("[Rank 0] Starting preprocessing pipeline...")
        train_path, test_path, feature_cols = preprocess_pipeline(
            raw_path="nytaxi2022.csv",
            out_train="nytaxi_train.csv",
            out_test="nytaxi_test.csv",
            outlier_removal_method="quantile",   # <── change this
            iqr_multiplier=2.0,                  # ignored now
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

    save_fig = f"results/{proc_num}_proc/bat_size{bat_size}/{act_name}/training_history_{act_name}.png"

    if RANK == 0:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)

    # Rounding the batch size
    proc_num_int = int(proc_num)
    reminder = bat_size%proc_num_int
    bat_size = bat_size+reminder

    # SGD Process
    train_losses, val_losses, batch_size, train_time = trainer.mpiSGD(
        train_file_path=train_path,
        readin_chunksize=5000,
        valid_portion=0.15,
        lr=1e-3,
        epochs=800,
        batch_size=int(bat_size),
        patience=10,
        lr_decay=0.95,
        report_per=20,
        lr_resch_stepsize=50,
        save_fig=save_fig
    )
    print()
    print(f"Finished SDG in {train_time} seconds")
    print()

    # # Test evaluation
    # test_df = pd.read_csv(test_path)
    # X_test = test_df[feature_cols].to_numpy(dtype=float)
    # y_test = test_df["total_amount"].to_numpy(dtype=float).reshape(-1, 1)
    
    df = pd.DataFrame({
    "epoch": np.arange(1, len(train_losses) + 1),  # add epoch index if needed
    "train_loss": train_losses,
    "val_loss": val_losses
    })

    csv_path = f"results/{proc_num}_proc/bat_size{bat_size}/{act_name}/loss_record_{act_name}.csv"
    if RANK == 0:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    COMM.Barrier()
    df.to_csv(csv_path, index=False)


    # Load in testing dataset on RANK 0 and distribute
    testDistributor = MPIDD(test_path, readin_chunksize=5000)
    testDistributor.mpi_load_in_and_distribute(target_col="total_amount", shuffle=True)
    X_test, y_test = testDistributor.get_X_y(target_name="total_amount")

    print()
    print(f"Computing Test RMSE of {X_test.shape[0]} rows data on Rank {trainer.RANK}")
    print()
    # Computing Test RMSE
    COMM.Barrier()
    t0_test = MPI.Wtime()
    test_rmse = trainer.compute_RMSE(X_test, y_test)
    COMM.Barrier()
    t1_test = MPI.Wtime()
    local_time = t1_test - t0_test
    test_time = COMM.reduce(local_time, op=MPI.MAX, root=0)
    print()
    print(f"Finished Computing RMSE in {test_time} seconds")
    print()


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
    del trainer, model, train_losses, val_losses, X_test, y_test, df

    gc.collect()                   # force Python GC

    # calcualte the time cost
    COMM.Barrier()
    
    if RANK == 0:
        return train_time, test_time, train_rmse, validation_rmse, test_rmse, batch_size
    else:
        return None, None, None, None, None, None


if __name__ == "__main__":
    experiment("relu", 0.1)
