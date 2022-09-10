"""
Utility functions.

References:
https://github.com/pyg-team/pytorch_geometric/blob/
575611f4f5e2209c7923dba977a1eebc207bd2e2/torch_geometric/
graphgym/cmd_args.py
https://github.com/dmlc/dgl/blob/a107993f106cecb1c375f7a6ae41088d04f29e29/
examples/pytorch/caregnn/utils.py
https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/data_utils.py
"""
import argparse
import yaml
import os
import fnmatch
import json
import random
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from models.gcn import GCN
from models.gat import GAT
from models.monet import MoNet
from models.graph_sage import GraphSAGE
from models.mlp import MLP
from models.gcn_minibatch import GCNminibatch
from models.mixhop import MixHop
from models.linkx import LINKX

Models_need_to_be_densed = ["GCN", "GraphSAGE", "GAT", "MixHop", "LINKX"]
Datasets_need_to_be_undirected = ["pokec", "genius", "penn94", "twitch-gamers"]


def generate_model(args, g, in_feats, n_classes, **model_cfg):
    """Generate required model."""
    # create models
    if args.model == "GCN":
        model = GCN(g=g,
                    in_feats=in_feats,
                    n_hidden=model_cfg["num_hidden"],
                    n_classes=n_classes,
                    n_layers=model_cfg["num_layers"],
                    activation=F.relu,
                    dropout=model_cfg["dropout"])
    elif args.model == "GAT":
        heads = ([model_cfg["num_heads"]] * (model_cfg["num_layers"]))\
                + [model_cfg["num_out_heads"]]
        model = GAT(g,
                    model_cfg["num_layers"],
                    in_feats,
                    model_cfg["num_hidden"],
                    n_classes,
                    heads,
                    F.elu,
                    model_cfg["dropout"],
                    model_cfg["dropout"],
                    model_cfg["negative_slope"],
                    model_cfg["residual"])
    elif args.model == "MoNet":
        model = MoNet(g,
                      in_feats,
                      model_cfg["num_hidden"],
                      n_classes,
                      model_cfg["num_layers"],
                      model_cfg["pseudo_dim"],
                      model_cfg["num_kernels"],
                      model_cfg["dropout"])
    elif args.model == "GraphSAGE":
        model = GraphSAGE(g,
                          in_feats,
                          model_cfg["num_hidden"],
                          n_classes,
                          model_cfg["num_layers"],
                          F.relu,
                          model_cfg["dropout"],
                          model_cfg["aggregator_type"])
    elif args.model == "MLP":
        model = MLP(in_feats,
                    model_cfg["num_hidden"],
                    n_classes,
                    model_cfg["num_layers"],
                    F.relu,
                    model_cfg["dropout"])
    elif args.model == "GCN_minibatch":
        model = GCNminibatch(in_feats,
                             model_cfg["num_hidden"],
                             n_classes,
                             model_cfg["num_layers"],
                             F.relu,
                             model_cfg["dropout"])
    elif args.model == "MixHop":
        model = MixHop(g,
                       in_dim=in_feats,
                       hid_dim=model_cfg["num_hidden"],
                       out_dim=n_classes,
                       p=model_cfg["p"],
                       num_layers=model_cfg["num_layers"],
                       input_dropout=model_cfg["dropout"],
                       layer_dropout=model_cfg["layer_dropout"],
                       batchnorm=model_cfg["batchnorm"])
    elif args.model == "LINKX":
        model = LINKX(g=g,
                      in_channels=in_feats,
                      num_nodes=g.ndata["NodeFeature"].shape[0],
                      hidden_channels=model_cfg["num_hidden"],
                      out_channels=n_classes,
                      num_layers=model_cfg["num_layers"],
                      dropout=model_cfg["dropout"],
                      inner_activation=model_cfg["inner_activation"],
                      inner_dropout=model_cfg["inner_dropout"],
                      init_layers_A=model_cfg["init_layers_A"],
                      init_layers_X=model_cfg["init_layers_X"])
    try:
        model
    except UnboundLocalError as exc:
        raise NameError(f"model {args.model} is not supported yet.") from exc
    else:
        return model


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="train for node\
                                                  classification")
    parser.add_argument("--model-cfg", type=str,
                        default="configs/model_default.yaml",
                        help="The model configuration file path.")
    parser.add_argument("--train-cfg", type=str,
                        default="configs/train_default.yaml",
                        help="The training configuration file path.")
    parser.add_argument("--model", type=str, default="GCN",
                        help="model to be used. GCN, GAT, MoNet,\
                              GraphSAGE, MLP, LINKX, MixHop for now")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="dataset to be trained")
    parser.add_argument("--task", type=str,
                        default="NodeClassification",
                        help="task name. NodeClassification,\
                        GraphClassification, LinkPrediction,\
                        TimeDependentLinkPrediction,\
                        KGRelationPrediction, NodeRegression.\
                        KGEntityPrediction, GraphRegression,\
                        for now")
    parser.add_argument("--task-id", type=int, default=1,
                        help="task id, starting from 1")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CP U.")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="whether to print verbosely")
    return parser.parse_args()


def load_config_file(path):
    """Load yaml files."""
    with open(path, "r", encoding="utf-8") as stream:
        try:
            parsed_yaml = yaml.full_load(stream)
            print(parsed_yaml)
            return parsed_yaml
        except yaml.YAMLError as exc:
            print(exc)


def check_multiple_split(dataset):
    """Check whether the dataset has multiple splits."""
    dataset_directory = os.path.dirname(os.path.dirname(os.getcwd())) \
        + "/datasets/" + dataset
    for file in os.listdir(dataset_directory):
        if fnmatch.fnmatch(file, "task*.json"):
            with open(dataset_directory + "/" + file,  encoding="utf-8") as f:
                task_dict = json.load(f)
                if "num_splits" in task_dict and task_dict["num_splits"] > 1:
                    return 1
                else:
                    return 0


class EarlyStopping:
    """Do early stopping."""

    def __init__(self, ckpt_name, patience=50):
        """Init early stopping."""
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.dir_name = "checkpoints/"
        if ~os.path.isdir(self.dir_name):
            os.makedirs(self.dir_name, exist_ok=True)
        ckpt_name = ckpt_name.replace("/", "_")
        ckpt_name = os.path.splitext(ckpt_name)[0]
        self.ckpt_dir = self.dir_name + ckpt_name + "_checkpoint.pt"

    def step(self, acc, model):
        """Step early stopping."""
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}\
                    out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Save model when validation loss decrease."""
        torch.save(model.state_dict(), self.ckpt_dir)


def makedirs_rm_exist(dir_name):
    """Make a directory, remove any existing data."""
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name, exist_ok=True)


def set_seed(seed):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def eval_rocauc(y_pred, y_true):
    """Evalution function for ROC."""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if len(y_true.shape) > 1:
        if y_true.shape[1] == 1:
            # use the predicted class for single-class classification
            y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
        else:
            y_pred = y_pred.detach().cpu().numpy()

        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_labeled = y_true[:, i] == y_true[:, i]
                score = roc_auc_score(y_true[is_labeled, i],
                                      y_pred[is_labeled, i])

                rocauc_list.append(score)
    else:
        y_pred = y_pred.detach().cpu().numpy()
        is_labeled = ~torch.isnan(torch.tensor(y_true))
        score = roc_auc_score(y_true[is_labeled], y_pred[is_labeled, 1])
        rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            "No positively labeled data available. Cannot compute ROC-AUC.")

    return sum(rocauc_list)/len(rocauc_list)


def check_binary_classification(dataset):
    """Check whether the dataset has multiple splits."""
    dataset_directory = os.path.dirname(os.path.dirname(os.getcwd())) \
        + "/datasets/" + dataset
    for file in os.listdir(dataset_directory):
        if fnmatch.fnmatch(file, "task*.json"):
            with open(dataset_directory + "/" + file,  encoding="utf-8") as f:
                task_dict = json.load(f)
                if "num_classes" in task_dict and\
                   task_dict["num_classes"] == 2:
                    return 1
                else:
                    return 0
