"""GAT train."""
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import dgl
import glb

from gat import GAT
from utils import EarlyStopping


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
    g = glb.graph.read_glb_graph(metadata_path=metadata_path[args.dataset],
                                 device=device)
    task = glb.task.read_glb_task(task_path=task_path[args.dataset])
    data = glb.dataloading.combine_graph_and_task(g, task)

    features = g.ndata["NodeFeature"]
    labels = g.ndata["NodeLabel"]
    train_mask = g.ndata["train_set"]
    val_mask = g.ndata["val_set"]
    test_mask = g.ndata["test_set"]

    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes,
    #        train_mask.int().sum().item(),
    #        val_mask.int().sum().item(),
    #        test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * (args.num_layers-1)) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
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
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break

        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f}"
              f"| Loss {loss.item():.4f} | TrainAcc {train_acc:.4f} |"
              f" ValAcc {val_acc:.4f} | "
              f"ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

    print()
    if args.early_stop:
        model.load_state_dict(torch.load("es_checkpoint.pt"))
    acc = evaluate(model, features, labels, test_mask)
    print(f"Test Accuracy {acc:.4f}")


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
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="provided datasets: cora, citeseer, pubmed, \
                            ogbn_arxiv, ogbn_mag")
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
    parser.add_argument("--early-stop", action="store_true", default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--fastmode", action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    Args = parser.parse_args()
    print(Args)

    main(Args)
