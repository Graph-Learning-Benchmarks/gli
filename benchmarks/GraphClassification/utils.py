"""
Utility functions in gli.

References:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin
https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/data_utils.py
https://scikit-learn.org/stable/modules/generated/sklearn.
metrics.roc_auc_score.html
"""

import os
import torch
import torch.nn.functional as F
import random
import numpy as np
import yaml
import fnmatch
import json
import shutil
from sklearn.metrics import roc_auc_score
from models.gin import GIN
from models.gcn import GCNgraph
from models.cheb_net import ChebNet
from models.dgn import DGN
import argparse


def generate_model(args, in_feats, n_classes, **model_cfg):
    """Generate required model."""
    # create models
    if args.model == "GIN":
        model = GIN(in_feats,
                    model_cfg["hidden_dim"],
                    n_classes)
    elif args.model == "GCN":
        model = GCNgraph(in_feats,
                         model_cfg["hidden_dim"],
                         n_classes,
                         model_cfg["num_layers"],
                         F.relu,
                         model_cfg["dropout"])
    elif args.model == "ChebNet":
        model = ChebNet(in_feats,
                        model_cfg["hidden_dim"],
                        n_classes,
                        model_cfg["num_layers"],
                        model_cfg["k"])
    elif args.model == "DGN":
        model = DGN(in_feats,
                    model_cfg["hidden_dim"],
                    n_classes,
                    model_cfg["num_layers"],
                    model_cfg["aggregators"],
                    model_cfg["scalers"],
                    model_cfg["delta"],
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
    parser.add_argument("--model", type=str, default="GCN",
                        help="model to be used. GIN, ChebNet, GCN,\
                              for now")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv",
                        help="dataset to be trained, ogbg-molhiv\
                             and ogbg-molpcba for now")
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


def eval_acc(y_pred, y_true):
    """
    Evaluate accuracy.

    Return a list of binary number, indicating the
    correctness of prediction.
    """
    correct_list = []
    y_true = y_true.detach().cpu().numpy()
    # y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    if len(y_true) > 1:
        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            # acc_list.append(float(np.sum(correct))/len(correct))
            correct_list.append(correct)
    else:
        is_labeled = ~torch.isnan(torch.tensor(y_true))
        _, predicted = torch.max(y_pred, 1)
        correct_list.append(predicted[is_labeled] == y_true[is_labeled])
    return correct_list


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
        if np.sum(y_true == 1) > 0 and np.sum(y_true == 0) > 0:
            is_labeled = ~torch.isnan(torch.tensor(y_true))
            score = roc_auc_score(y_true[is_labeled], y_pred[is_labeled, 1])
            rocauc_list.append(score)

    # if len(rocauc_list) == 0:
    #     raise RuntimeError(
    #         "No positively labeled data available. Cannot compute ROC-AUC.")

    # return sum(rocauc_list)/len(rocauc_list)
    return torch.tensor(rocauc_list)


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


def get_label_number(dataloader):
    """Return the label number of dataset."""
    for _, labels in dataloader:
        if len(labels.shape) > 1:
            return labels.shape[1]
        else:
            return 1


def makedirs_rm_exist(dir_name):
    """Make a directory, remove any existing data."""
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name, exist_ok=True)