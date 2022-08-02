"""Test if the dataset can be used to train."""
import glb
import torch
import torch.nn.functional as F
import numpy as np
import dgl
import time
from test_data_loading import test_data_loading
from test_training_utils import accuracy, evaluate, warmup, \
                                check_multiple_split_v2, GCN


def test_training(dataset):
    """
    Test if the dataset can be trained for two epochs.

    If True, return.
    Else, assert False.
    Use model GCN to do test training
    """
    print("First test whether the format of the dataset is correct")
    test_data_loading(dataset)

    args, model_cfg, train_cfg = warmup(dataset)
    device = "cpu"

    data = glb.dataloading.get_glb_dataset(args["dataset"], args["task"],
                                           device=device)
    g = data[0]
    if train_cfg["dataset"]["self_loop"]:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    features = g.ndata["NodeFeature"]
    labels = g.ndata["NodeLabel"]
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
    n_classes = data.num_labels
    n_edges = g.number_of_edges()

    print(f"""----Data statistics------'
      #Edges {n_edges}
      #Classes {n_classes}
      #Train samples {train_mask.int().sum().item()}
      #Val samples {val_mask.int().sum().item()}
      #Test samples {test_mask.int().sum().item()}""")

    # create model
    model = GCN(g,
                in_feats,
                model_cfg["num_hidden"],
                n_classes,
                model_cfg["num_layers"],
                F.relu,
                model_cfg["dropout"])

    print(model)

    loss_fcn = torch.nn.CrossEntropyLoss()

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

        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = evaluate(model, features, labels, val_mask)
        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f}"
              f"| Loss {loss.item():.4f} | TrainAcc {train_acc:.4f} |"
              f" ValAcc {val_acc:.4f} | "
              f"ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

        print("The dataset has successfully trained \
               on GCN model for 3 epoches.")
        print("Test passed.")
