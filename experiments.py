# from MpiTraining import experiment
from mpinn.MpiTraining import experiment
# from MpiTraining_archived import experiment_arc
import os, ast, argparse
from datetime import datetime
import pandas as pd
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("act_func_list", type=str)
        parser.add_argument("batch_portion_list", type=str)
        parser.add_argument("p_num", type=str)
        return parser.parse_args()
    
    args = parse_args()
    bat_size_list = ast.literal_eval(args.batch_portion_list)
    act_func_list = ast.literal_eval(args.act_func_list)
    process_number = args.p_num
    rows = []

    for batch_size in bat_size_list:
        for act_func in act_func_list:
            batch_size = int(batch_size)
            # Run experiment
            sgd_iter_time, test_time, train_rmse, validation_rmse, test_rmse, bat_siz = experiment(act_name=act_func, bat_size=batch_size, proc_num=process_number)
            # Store metrics
            if RANK == 0:  # only root collects to avoid duplicates
                rows.append({
                    "process_num": process_number,
                    "act_func": act_func,
                    "SGD_100_iter_time": sgd_iter_time*100,
                    "testing_time": test_time,
                    "train_rmse": train_rmse,
                    "val_rmse": validation_rmse,
                    "test_rmse": test_rmse,
                    "batch_size": bat_siz
                })
    COMM.Barrier()
    # GPT: Update the table if it already exists
    if RANK == 0 and rows:
        os.makedirs("table", exist_ok=True)
        out_path = "table/summary_metrics.csv"

        # Create DataFrame from new rows
        new_df = pd.DataFrame(rows, columns=[
            "process_num", "act_func",
            "SGD_100_iter_time", "testing_time", "train_rmse", 
            "val_rmse", "test_rmse", "batch_size"
        ])

        # If file exists, read existing CSV and append
        if os.path.exists(out_path):
            old_df = pd.read_csv(out_path)
            df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            df = new_df

        # Save updated DataFrame
        df.to_csv(out_path, index=False)
             
            