"""GCN train."""
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import glb

from models.gcn import GCN
# from gcn_mp import GCN
# from gcn_spmv import GCN


def evaluate(model, features, labels, mask):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    """Load dataset and train model."""
    # load and preprocess dataset
    if args.gpu < 0:
        device = "cpu"
        cuda = False
    else:
        device = args.gpu
        cuda = True
    g = glb.dataloading.get_glb_graph(args.dataset, device=device)
    task = glb.dataloading.get_glb_task(args.dataset, args.task)
    data = glb.dataloading.get_glb_dataset(args.dataset, args.task, \
                                           device=device)

    features = g.ndata["NodeFeature"]
    print("features: ", features)
    labels = g.ndata["NodeLabel"]
    print("labels: ", labels)
    print("g.ndata: ", g.ndata)
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    # g.ndata["norm"] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.hidden,
                n_classes,
                args.layers,
                F.relu,
                args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f}"
              f" | Loss {loss.item():.4f} | Accuracy {acc:.4f}"
              f" | ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

    print()
    acc = evaluate(model, features, labels, test_mask)
    print(f"Test accuracy {acc:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="provided datasets: cora, citeseer, \
                            pubmed")
    parser.add_argument("--task", type=str,\
                        default="NodeClassification",
                        help="task for NodeClassification")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action="store_true",
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    Args = parser.parse_args()
    print(Args)

    main(Args)
