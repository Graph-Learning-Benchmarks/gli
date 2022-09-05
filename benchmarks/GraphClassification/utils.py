"""
Utility functions in gli.

References:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin
"""

import torch
import torch.nn.functional as F
import random
import numpy as np
import yaml
from models.gin import GIN
from models.gcn import GCN
import argparse


def generate_model(args, in_size, out_size, **model_cfg):
    """Generate required model."""
    # create models
    if args.model == "GIN":
        model = GIN(in_size,
                    model_cfg["hidden_dim"],
                    out_size)
    elif args.model == "GCN":
        model = GCN(in_size,
                    model_cfg["hidden_dim"],
                    out_size,
                    model_cfg["num_layers"],
                    F.relu,
                    model_cfg["dropout"])

    try:
        model
    except UnboundLocalError as exc:
        raise NameError(f"model {args.model} is not supported yet.") from exc
    else:
        return model


def load_config_file(path):
    """Load yaml files."""
    with open(path, "r", encoding="utf-8") as stream:
        try:
            parsed_yaml = yaml.full_load(stream)
            print(parsed_yaml)
            return parsed_yaml
        except yaml.YAMLError as exc:
            print(exc)


def set_seed(seed):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="train for graph\
                                                  classification")
    parser.add_argument("--model-cfg", type=str,
                        default="configs/model_default.yaml",
                        help="The model configuration file path.")
    parser.add_argument("--train-cfg", type=str,
                        default="configs/train_default.yaml",
                        help="The training configuration file path.")
    parser.add_argument("--model", type=str, default="GIN",
                        help="model to be used. GCN, GAT, MoNet,\
                              GraphSAGE, MLP, LINKX, MixHop for now")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv",
                        help="dataset to be trained")
    parser.add_argument("--task", type=str,
                        default="GraphClassification",
                        help="task name. NodeClassification,\
                        GraphClassification, LinkPrediction,\
                        TimeDependentLinkPrediction,\
                        KGRelationPrediction, NodeRegression.\
                        KGEntityPrediction, GraphRegression,\
                        for now")
    parser.add_argument("--task-id", type=int, default=1,
                        help="task id, starting from 1")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="whether to print verbosely")
    return parser.parse_args()
