"""GAT train."""
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import dgl
import glb

from models.gcn import GCN
from models.gat import GAT


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
    features = g.ndata["NodeFeature"]
    labels = g.ndata["NodeLabel"]
    train_mask = g.ndata["train_set"]
    val_mask = g.ndata["val_set"]
    test_mask = g.ndata["test_set"]

    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = g.number_of_edges()
    print(f"""----Data statistics------'
      #Edges {n_edges}
      #Classes {n_classes}
      #Train samples {train_mask.int().sum().item()}
      #Val samples {val_mask.int().sum().item()}
      #Test samples {test_mask.int().sum().item()}""")

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    # create model
    if args.model == "GCN":
        model = GCN(g,
                    in_feats,
                    args.num_hidden,
                    n_classes,
                    args.num_layers,
                    F.relu,
                    args.in_drop)
    elif args.model == "GAT":
        heads = ([args.num_heads] * (args.num_layers-1)) + [args.num_out_heads]
        model = GAT(g,
                    args.num_layers,
                    in_feats,
                    args.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual)

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
            val_acc = evaluate(model, features, labels, val_mask)

        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f}"
              f"| Loss {loss.item():.4f} | TrainAcc {train_acc:.4f} |"
              f" ValAcc {val_acc:.4f} | "
              f"ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

    print()
    acc = evaluate(model, features, labels, test_mask)
    print(f"Test Accuracy {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train for node\
                                                  classification")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="provided datasets: cora, citeseer, pubmed, \
                            ogbn_arxiv, ogbn_mag")
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
    parser.add_argument("--self-loop", action="store_true",
                        help="graph self-loop (default=False)")
    Args = parser.parse_args()
    print(Args)

    main(Args)
