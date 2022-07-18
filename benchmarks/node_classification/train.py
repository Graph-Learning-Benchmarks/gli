"""
Train for node classification dataset.

References:
https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/train.py
"""

import argparse
import time
import os
import fnmatch
import json
import torch
import numpy as np
import dgl
import glb

from utils import generate_model
from glb.utils import to_dense


def accuracy(logits, labels):
    """Calculate accuracy."""
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(args, model, features, labels, mask, pseudo):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        if args.model == "MoNet":
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


def main(args):
    """Load dataset and train the model."""
    # load and preprocess dataset
    if args.gpu < 0:
        device = "cpu"
        cuda = False
    else:
        device = args.gpu
        cuda = True

    data = glb.dataloading.get_glb_dataset(args.dataset, args.task,
                                           device=device)
    g = data[0]
    if args.to_dense:
        g = to_dense(g)
    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    features = g.ndata["NodeFeature"]
    labels = g.ndata["NodeLabel"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # for multi-split dataset, choose 0-th split for now
    if check_multiple_split(args.dataset):
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
    model = generate_model(args, g, in_feats, n_classes)

    print(model)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.time()
        # forward
        if args.model == "MoNet":
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

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(args, model, features, labels, val_mask, pseudo)

        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f}"
              f"| Loss {loss.item():.4f} | TrainAcc {train_acc:.4f} |"
              f" ValAcc {val_acc:.4f} | "
              f"ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

    print()
    acc = evaluate(args, model, features, labels, test_mask, pseudo)
    print(f"Test Accuracy {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train for node\
                                                  classification")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="dataset to be trained")
    parser.add_argument("--task", type=str,
                        default="NodeClassification",
                        help="task for NodeClassification")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative-slope", type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--fastmode", action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument("--pseudo-dim", type=int, default=2,
                        help="Pseudo coordinate dimensions in GMMConv,\
                              2 for cora and 3 for pubmed")
    parser.add_argument("--num-kernels", type=int, default=3,
                        help="Number of kernels in GMMConv layer")
    parser.add_argument("--self-loop", action="store_true",
                        help="graph self-loop (default=False)")
    parser.add_argument("--to-dense", action="store_true",
                        help="whether change the model into dense \
                              (default=False)")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    Args = parser.parse_args()
    print(Args)

    main(Args)
