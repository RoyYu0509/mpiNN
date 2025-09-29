from MpiTraining import experiment
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
    bat_por_list = ast.literal_eval(args.batch_portion_list)
    act_func_list = ast.literal_eval(args.act_func_list)
    process_number = args.p_num
    rows = []

    for portion in bat_por_list:
        for act_func in act_func_list:
            # Run experiment
            train_time, train_rmse, validation_rmse, test_rmse = experiment(act_name=act_func, batch_portion=portion)
            # Store metrics
            if RANK == 0:  # only root collects to avoid duplicates
                rows.append({
                    "act_func": act_func,
                    "batch_portion": portion,
                    "training_time": train_time,
                    "train_rmse": train_rmse,
                    "val_rmse": validation_rmse,
                    "test_rmse": test_rmse,
                })

    COMM.Barrier()
    if RANK == 0 and rows:
        df = pd.DataFrame(rows, columns=[
            "act_func", "batch_portion", "training_time",
            "train_rmse", "val_rmse", "test_rmse"
        ])
        os.makedirs(f"tables", exist_ok=True)

        out_path = f"tables/summary_{process_number}process.csv"
        df.to_csv(out_path, index=False)
             
            