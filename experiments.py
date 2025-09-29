from MpiTraining import experiment
import argparse, ast

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("act_func", type=str)
        parser.add_argument("batch_portion_list", type=str)
        return parser.parse_args()
    
    args = parse_args()
    bat_por_list = ast.literal_eval(args.batch_portion_list)
    for portion in bat_por_list:
        experiment(act_name=args.act_func, batch_portion=portion)
        