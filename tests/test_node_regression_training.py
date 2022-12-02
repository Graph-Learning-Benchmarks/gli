"""Test if the dataset can be used to train."""
import pytest
import re
import gli
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import dgl
import time
from utils import find_datasets
from gli.utils import to_dense
from training_utils import get_cfg, check_dataset_task, \
                           check_multiple_split_v2
from benchmarks.NodeClassification.models.gcn import GCN


def evaluate(model, features, labels, mask, eval_func):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return eval_func(logits.squeeze(), labels.float())


@pytest.mark.parametrize("dataset_name", find_datasets())
def test_training(dataset_name):
    """
    Test if the dataset can be trained for two epochs.

    If True, return.
    Else, assert False.
    Use model GCN to do test training.
    """
    # only do the test on NodeRegression datasets
    if not check_dataset_task(dataset_name, "NodeRegression"):
        return

    args, model_cfg, train_cfg = get_cfg(dataset_name)
    device = "cpu"

    data = gli.dataloading.get_gli_dataset(args["dataset"], "NodeRegression",
                                           1, device=device)

    g = data[0]
    g = to_dense(g)
    # convert to undirected set
    if train_cfg["dataset"]["self_loop"]:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    feature_name = re.search(r".*Node/(\w+)", data.features[0]).group(1)
    label_name = re.search(r".*Node/(\w+)", data.target).group(1)
    features = g.ndata[feature_name]
    labels = g.ndata[label_name]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # for multi-split dataset, choose 0-th split for now
    if check_multiple_split_v2(args["dataset"]):
        print("Need to choose one set from multiple split.")
        train_mask = train_mask[:, 0]
        val_mask = val_mask[:, 0]
        test_mask = test_mask[:, 0]

    # When labels contains -1, modify masks
    if min(labels) < 0:
        train_mask = train_mask * (labels >= 0)
        val_mask = val_mask * (labels >= 0)
        test_mask = test_mask * (labels >= 0)

    in_feats = features.shape[1]
    n_edges = g.number_of_edges()

    print(f"""----Data statistics------
      #Edges {n_edges}
      #Train samples {train_mask.int().sum().item()}
      #Val samples {val_mask.int().sum().item()}
      #Test samples {test_mask.int().sum().item()}""")

    # create model
    model = GCN(g,
                in_feats,
                model_cfg["num_hidden"],
                1,  # output size is 1 in regression, instead of n_classes
                model_cfg["num_layers"],
                F.relu,
                model_cfg["dropout"])

    print(model)

    eval_func = loss_fcn = nn.MSELoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["optim"]["lr"],
        weight_decay=train_cfg["optim"]["weight_decay"])

    # initialize graph
    dur = []
    for epoch in range(train_cfg["max_epoch"]):
        model.train()
        t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask].squeeze(),
                        labels[train_mask].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)

        val_loss = evaluate(model, features, labels, val_mask, eval_func)
        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} "
              f"| Loss {loss.item():.4f} | "
              f" Val Loss {val_loss.item():.4f} | "
              f"ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

    print("Test passed for dataset", args["dataset"])
