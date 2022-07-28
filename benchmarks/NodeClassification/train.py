"""
Train for node classification dataset.

References:
https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/train.py
https://github.com/pyg-team/pytorch_geometric/blob/master/graphgym/main.py
"""


import time
import torch
import numpy as np
import dgl
import glb
from utils import generate_model, parse_args, Models_need_to_be_densed,\
                  load_config_file, check_multiple_split
from glb.utils import to_dense


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


def main(args, model_cfg, train_cfg):
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
    if train_cfg["dataset"]["to_dense"] or \
       args.model in Models_need_to_be_densed:
        g = to_dense(g)
    # add self loop
    if train_cfg["dataset"]["self_loop"]:
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

    print(f"""----Data statistics------'
      #Edges {n_edges}
      #Classes {n_classes}
      #Train samples {train_mask.int().sum().item()}
      #Val samples {val_mask.int().sum().item()}
      #Test samples {test_mask.int().sum().item()}""")

    # create model
    model = generate_model(args, g, in_feats, n_classes, **model_cfg)

    print(model)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["optim"]["lr"],
        weight_decay=train_cfg["optim"]["weight_decay"])

    # initialize graph
    dur = []
    for epoch in range(train_cfg["max_epoch"]):
        model.train()
        if epoch >= 3:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.time()
        # forward
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
        val_acc = evaluate(model, features, labels, val_mask)
        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f}"
              f"| Loss {loss.item():.4f} | TrainAcc {train_acc:.4f} |"
              f" ValAcc {val_acc:.4f} | "
              f"ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

    print()

    acc = evaluate(model, features, labels, test_mask)
    print(f"Test Accuracy {acc:.4f}")


if __name__ == "__main__":
    # Load cmd line args
    Args = parse_args()
    print(Args)
    # Load config file
    Model_cfg = load_config_file(Args.model_cfg)
    Train_cfg = load_config_file(Args.train_cfg)

    main(Args, Model_cfg, Train_cfg)
