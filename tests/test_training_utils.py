"""Functions used in test_training."""
import torch
import os
import fnmatch
from torch import nn
from dgl.nn.pytorch import GraphConv
import json


def accuracy(logits, labels):
    """Calculate accuracy."""
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def warmup(dataset):
    """Return fixed dict to test_training."""
    args = {
        "model": "GCN",
        "dataset": dataset,
        "task": "task",
        "gpu": -1
    }

    model_cfg = {
        "num_layers": 2,
        "num_hidden": 8,
        "dropout": .6
    }

    train_cfg = {
        "loss_fun": "cross_entropy",
        "dataset": {
            "self_loop": True,
            "to_dense": False
        },
        "optim": {
            "lr": .005,
            "weight_decay": 0.0005
        },
        "num_trials": 1,
        "max_epoch": 3
    }
    return args, model_cfg, train_cfg


def check_multiple_split_v2(dataset):
    """Check whether the dataset has multiple splits."""
    print()
    dataset_directory = os.getcwd() \
        + "/datasets/" + dataset
    for file in os.listdir(dataset_directory):
        if fnmatch.fnmatch(file, "task*.json"):
            with open(dataset_directory + "/" + file,  encoding="utf-8") as f:
                task_dict = json.load(f)
                if "num_splits" in task_dict and task_dict["num_splits"] > 1:
                    return 1
                else:
                    return 0


class GCN(nn.Module):
    """GCN network."""

    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        """Initiate model."""
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden,
                                     activation=activation,
                                     norm="none"))
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden,
                                         activation=activation,
                                         norm="none"))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes,
                                     norm="none"))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        """Forward."""
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
