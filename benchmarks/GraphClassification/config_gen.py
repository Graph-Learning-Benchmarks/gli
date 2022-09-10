"""
Random search.

References:
https://github.com/pyg-team/pytorch_geometric/blob/master/graphgym/configs_gen.py
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/
graphgym/utils/io.py
"""

import argparse
import yaml
import time
from utils import load_config_file, makedirs_rm_exist
from random import randint

train_cfg_list = ["self_loop", "to_dense", "lr", "weight_decay", "num_trials",
                  "max_epoch", "early_stopping"]


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cfg", type=str,
                        default="configs/model_default.yaml",
                        help="The model configuration file path.")
    parser.add_argument("--train-cfg", type=str,
                        default="configs/train_default.yaml",
                        help="The training configuration file path.")
    parser.add_argument("--grid", type=str,
                        help="configuration file for grid search.",
                        default="grid/grid_example.yaml")
    parser.add_argument("--sample-num", dest="sample_num",
                        help="Number of random samples in the space.",
                        default=10, type=int)
    parser.add_argument("--trial-num", type=int, default=5,
                        help="Number of trials for same configuration.")
    parser.add_argument("--model", type=str, default="GCN",
                        help="model to be used. GCN, GIN, ChebNet for now.")
    return parser.parse_args()


def grid_gen(args, gen_cfg, model_cfg, train_cfg):
    """Generate random search configuration files."""
    dir_name = "./grid/" + args.model + time.strftime("_%Y%m%d_%H%M%S")
    makedirs_rm_exist(dir_name)
    for i in range(args.sample_num):
        for key in gen_cfg:
            key_len = len(gen_cfg[key])
            if key in train_cfg_list:
                train_cfg[key] = gen_cfg[key][randint(0, key_len-1)]
            else:
                # otherwise, the key is for model
                model_cfg[key] = gen_cfg[key][randint(0, key_len-1)]
        for j in range(args.trial_num):
            index_str = str(i) + "_" + str(j)
            # the i-th configuration, j-th trial
            train_cfg_name = args.model + "_train_" + index_str + ".yaml"
            model_cfg_name = args.model + "_model_" + index_str + ".yaml"
            train_cfg["seed"] = randint(1, 10000)
            with open(dir_name + "/" + train_cfg_name,
                      "w", encoding="utf-8") as f:
                yaml.dump(train_cfg, f, default_flow_style=False)
            with open(dir_name + "/" + model_cfg_name,
                      "w", encoding="utf-8") as f:
                yaml.dump(model_cfg, f, default_flow_style=False)


if __name__ == "__main__":
    Args = parse_args()
    Gen_cfg = load_config_file(Args.grid)
    # load default configuration for training and model
    Model_cfg = load_config_file(Args.model_cfg)
    Train_cfg = load_config_file(Args.train_cfg)
    grid_gen(Args, Gen_cfg, Model_cfg, Train_cfg)
