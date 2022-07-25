"""
Train for node classification dataset.

References:
https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/train.py
https://github.com/pyg-team/pytorch_geometric/blob/master/graphgym/main.py
"""


import time
import os
import fnmatch
import json
import torch
import numpy as np
import dgl
import glb
from config import (
    CFG,
    load_cfg,
    set_out_dir,
)
from utils import generate_model, parse_args, Models_need_to_be_densed
from glb.utils import to_dense


def accuracy(logits, labels):
    """Calculate accuracy."""
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(cfg, model, features, labels, mask, pseudo=None):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        if cfg.model.name == "MoNet":
            logits = model(features, pseudo)
        else:
            logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def check_multiple_split(dataset):
    """Chceck whether the dataset has multiple splits."""
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


def main(cfg):
    """Load dataset and train the model."""
    # load and preprocess dataset
    if cfg.train.gpu < 0:
        device = "cpu"
        cuda = False
    else:
        device = cfg.train.gpu
        cuda = True

    data = glb.dataloading.get_glb_dataset(cfg.dataset.name, cfg.dataset.task,
                                           device=device)
    g = data[0]
    if cfg.dataset.to_dense or cfg.model.name in Models_need_to_be_densed:
        g = to_dense(g)
    # add self loop
    if cfg.dataset.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    features = g.ndata["NodeFeature"]
    labels = g.ndata["NodeLabel"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # for multi-split dataset, choose 0-th split for now
    if check_multiple_split(cfg.dataset.name):
        train_mask = train_mask[:, 0]
        val_mask = val_mask[:, 0]
        test_mask = test_mask[:, 0]

    # When labels contains -1, modify masks
    if min(labels) < 0:
        train_mask = train_mask * (labels >= 0)
        val_mask = val_mask * (labels >= 0)
        test_mask = test_mask * (labels >= 0)

    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = g.number_of_edges()

    # calculate normalization factor (MoNet)
    if cfg.model.name == "MoNet":
        us, vs = g.edges(order="eid")
        udeg, vdeg = 1 / torch.sqrt(g.in_degrees(us).float()), 1 / \
            torch.sqrt(g.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)

    print(f"""----Data statistics------'
      #Edges {n_edges}
      #Classes {n_classes}
      #Train samples {train_mask.int().sum().item()}
      #Val samples {val_mask.int().sum().item()}
      #Test samples {test_mask.int().sum().item()}""")

    # create model
    model = generate_model(cfg, g, in_feats, n_classes)

    print(model)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(cfg.train.epochs):
        model.train()
        if epoch >= 3:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.time()
        # forward
        if cfg.model.name == "MoNet":
            logits = model(features, pseudo)
        else:
            logits = model(features)

        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            if cuda:
                torch.cuda.synchronize()
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if cfg.train.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        elif cfg.model.name == "MoNet":
            val_acc = evaluate(cfg, model, features, labels, val_mask, pseudo)
        else:
            val_acc = evaluate(cfg, model, features, labels, val_mask)

        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f}"
              f"| Loss {loss.item():.4f} | TrainAcc {train_acc:.4f} |"
              f" ValAcc {val_acc:.4f} | "
              f"ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

    print()
    if cfg.model.name == "MoNet":
        acc = evaluate(cfg, model, features, labels, test_mask, pseudo)
    else:
        acc = evaluate(cfg, model, features, labels, test_mask)
    print(f"Test Accuracy {acc:.4f}")


if __name__ == "__main__":
    # Load cmd line args
    Args = parse_args()
    print(Args)
    # Load config file
    load_cfg(CFG, Args)
    set_out_dir(CFG.out_dir, Args.cfg_file)
    print(CFG)
    CFG.dataset.name = Args.dataset
    CFG.model.name = Args.model
    CFG.dataset.task = Args.task
    CFG.train.gpu = Args.gpu
    CFG.train.epochs = Args.epochs
    main(CFG)
