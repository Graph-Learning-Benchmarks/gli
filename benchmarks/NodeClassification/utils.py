"""
Utility functions.

References:
https://github.com/pyg-team/pytorch_geometric/blob/
575611f4f5e2209c7923dba977a1eebc207bd2e2/torch_geometric/
graphgym/cmd_args.py
"""
import argparse
import errno
import os
import os.path as osp
import torch.nn.functional as F
from models.gcn import GCN
from models.gat import GAT
from models.monet import MoNet
from models.graph_sage import GraphSAGE
from models.mlp import MLP


Models_need_to_be_densed = ["MoNet", "GAT"]


def generate_model(cfg, g, in_feats, n_classes):
    """Generate required model."""
    # create model
    if cfg.model.name == "GCN":
        model = GCN(g,
                    in_feats,
                    cfg.model.num_hidden,
                    n_classes,
                    cfg.model.num_layers,
                    F.relu,
                    cfg.model.dropout)
    elif cfg.model.name == "GAT":
        heads = ([cfg.model.num_heads] * (cfg.model.num_layers-1))\
                + [cfg.model.num_out_heads]
        model = GAT(g,
                    cfg.model.num_layers,
                    in_feats,
                    cfg.model.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    cfg.model.dropout,
                    cfg.model.dropout,
                    cfg.model.Negative_slope,
                    cfg.model.residual)
    elif cfg.model.name == "MoNet":
        model = MoNet(g,
                      in_feats,
                      cfg.model.num_hidden,
                      n_classes,
                      cfg.model.num_layers,
                      cfg.model.pseudo_dim,
                      cfg.model.num_kernels,
                      cfg.model.dropout)
    elif cfg.model.name == "GraphSAGE":
        model = GraphSAGE(g,
                          in_feats,
                          cfg.model.num_hidden,
                          n_classes,
                          cfg.model.num_layers,
                          F.relu,
                          cfg.model.dropout,
                          cfg.model.aggregator_type)
    elif cfg.model.name == "MLP":
        model = MLP(in_feats,
                    cfg.model.num_hidden,
                    n_classes,
                    cfg.model.num_layers,
                    F.relu,
                    cfg.model.in_drop)
    return model


def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="train for node\
                                                  classification")
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="The configuration file path.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="The number of repeated jobs.")
    parser.add_argument("--mark_done", action="store_true",
                        help="Mark yaml as done after a job has finished.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See graphgym/config.py for remaining options.")
    parser.add_argument("--model", type=str, default="GCN",
                        help="model to be used. GCN, GAT, MoNet,\
                              GraphSAGE, MLP for now")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="dataset to be trained")
    parser.add_argument("--task", type=str,
                        default="task",
                        help="task name for NodeClassification,")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    return parser.parse_args()
