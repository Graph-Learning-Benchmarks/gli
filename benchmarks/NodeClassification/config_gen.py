"""
Grid search.

References:
https://github.com/pyg-team/pytorch_geometric/blob/master/graphgym/configs_gen.py
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/
graphgym/utils/io.py
"""

import argparse
import yaml
import random
from utils import load_config_file, makedirs_rm_exist
from random import randint

train_cfg_list = ["self_loop", "to_dense", "lr", "weight_decay", "num_trials",
                  "max_epoch", "early_stopping"]

random.seed(123)


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
                        help="configuration file for grid search",
                        default="grid/grid_example.yaml")
    parser.add_argument("--sample_num", dest="sample_num",
                        help="Number of random samples in the space",
                        default=10, type=int)
    return parser.parse_args()


def gen_grid(args, gen_cfg, model_cfg, train_cfg):
    """Generate random search configuration files."""
    for i in range(args.sample_num):
        train_cfg_name = "train"
        model_cfg_name = "model"
        for key in gen_cfg:
            key_len = len(gen_cfg[key])
            if key in train_cfg_list:
                train_cfg[key] = gen_cfg[key][randint(0, key_len-1)]
                train_cfg_name += f"_{key}={train_cfg[key]}"
            else:
                # otherwise, the key is for model
                model_cfg[key] = gen_cfg[key][randint(0, key_len-1)]
                model_cfg_name += f"_{key}={model_cfg[key]}"
        makedirs_rm_exist(f"grid/{i}")
        with open(f"grid/{i}/{train_cfg_name}.yaml",
                  "w", encoding="utf-8") as f:
            yaml.dump(train_cfg, f, default_flow_style=False)
        with open(f"grid/{i}/{model_cfg_name}.yaml",
                  "w", encoding="utf-8") as f:
            yaml.dump(model_cfg, f, default_flow_style=False)


if __name__ == "__main__":
    Args = parse_args()
    Gen_cfg = load_config_file(Args.grid)
    # load default configuration for training and model
    Model_cfg = load_config_file(Args.model_cfg)
    Train_cfg = load_config_file(Args.train_cfg)
    gen_grid(Args, Gen_cfg, Model_cfg, Train_cfg)
