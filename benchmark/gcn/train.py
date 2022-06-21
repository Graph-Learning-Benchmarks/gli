"""GCN train."""
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import glb

from gcn import GCN
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
    g = glb.graph.read_glb_graph(metadata_path=metadata_path[args.dataset],
                                 device=device)
    task = glb.task.read_glb_task(task_path=task_path[args.dataset])
    data = glb.dataloading.combine_graph_and_task(g, task)

    features = g.ndata["NodeFeature"]
    labels = g.ndata["NodeLabel"]
    train_mask = g.ndata["train_set"]
    val_mask = g.ndata["val_set"]
    test_mask = g.ndata["test_set"]

    in_feats = features.shape[1]
    n_classes = data.num_labels
    # n_edges = data.graph.number_of_edges()
    n_edges = 0
    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes,
    #           train_mask.int().sum().item(),
    #           val_mask.int().sum().item(),
    #           test_mask.int().sum().item()))

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    # n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata["norm"] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
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
    for epoch in range(args.n_epochs):
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
    print("Test accuracy {acc:.2%}")


if __name__ == "__main__":
    metadata_path = {
        "citeseer": "../../examples/citeseer/metadata.json",
        "cora": "../../examples/cora/metadata.json",
        "pubmed": "../../examples/pubmed/metadata.json",
        "ogbn_arxiv":
        "../../examples/ogb_data/node_prediction/ogbn-arxiv/metadata.json",
        "ogbn_mag":
        "../../examples/ogb_data/node_prediction/ogbn-mag/metadata.json"
    }
    task_path = {
        "citeseer": "../../examples/citeseer/task.json",
        "cora": "../../examples/cora/task.json",
        "pubmed": "../../examples/pubmed/task.json",
        "ogbn_arxiv":
        "../../examples/ogb_data/node_prediction/ogbn-arxiv/task.json",
        "ogbn_mag":
        "../../examples/ogb_data/node_prediction/ogbn-mag/task.json"
    }
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="provided datasets: cora, citeseer, \
                            pubmed, ogbn_arxiv, ogbn_mag")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action="store_true",
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    Args = parser.parse_args()
    print(Args)

    main(Args)
