"""
Train for node classification dataset.

References:
https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/train.py
https://github.com/pyg-team/pytorch_geometric/blob/master/graphgym/main.py
https://docs.dgl.ai/guide/minibatch-node.html?highlight=sampling
"""


import time
import torch
import numpy as np
import dgl
import glb
from utils import generate_model, parse_args, Models_need_to_be_densed,\
                  load_config_file, check_multiple_split,\
                  EarlyStopping
from glb.utils import to_dense
from dgl.dataloading import MultiLayerFullNeighborSampler as Sampler


def accuracy(logits, labels):
    """Calculate accuracy."""
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, dataloader):
    """Evaluate model."""
    model.eval()
    ys = []
    y_hats = []
    for _, _, blocks in dataloader:
        with torch.no_grad():
            input_features = blocks[0].srcdata["NodeFeature"]
            ys.append(blocks[-1].dstdata["NodeLabel"])
            y_hats.append(model(blocks, input_features))
    return accuracy(torch.cat(y_hats), torch.cat(ys))


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
    # check EdgeFeature and multi-modal node features
    edge_cnt = node_cnt = 0
    if len(data.features) > 1:
        for _, element in enumerate(data.features):
            if "Edge" in element:
                edge_cnt += 1
            if "Node" in element:
                node_cnt += 1
        if edge_cnt >= 1:
            raise NotImplementedError("Edge feature is not supported yet.")
        elif node_cnt >= 2:
            raise NotImplementedError("Multi-modal node features\
                                       is not supported yet.")

    g = data[0]
    indice = data.get_node_indices()

    if train_cfg["to_dense"] or \
       args.model in Models_need_to_be_densed:
        g = to_dense(g)
    # add self loop
    if train_cfg["self_loop"]:
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

    sampler = Sampler(model_cfg["num_layers"] + 1)
    train_dataloader = dgl.dataloading.DataLoader(
        g, indice["train_set"], sampler,
        batch_size=16,
        device=device,
        shuffle=True,
        drop_last=False)

    valid_dataloader = dgl.dataloading.DataLoader(
            g, indice["val_set"], sampler,
            device=device,
            batch_size=16,
            shuffle=True,
            drop_last=False)

    test_dataloader = dgl.dataloading.DataLoader(
            g, indice["test_set"], sampler,
            device=device,
            batch_size=16,
            shuffle=True,
            drop_last=False)

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
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"])

    if train_cfg["early_stopping"]:
        ckpt_name = args.model + "_" + args.dataset
        stopper = EarlyStopping(ckpt_name=ckpt_name, patience=50)

    # initialize graph
    dur = []
    for epoch in range(train_cfg["max_epoch"]):
        model.train()
        if epoch >= 3:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.time()

        for it, (_, _, blocks) in enumerate(train_dataloader):
            if cuda:
                blocks = [b.to(torch.device("cuda")) for b in blocks]
            input_features = blocks[0].srcdata["NodeFeature"]
            output_labels = blocks[-1].dstdata["NodeLabel"]
            logits = model(blocks, input_features)
            loss = loss_fcn(logits, output_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if it % 20 == 0:
                train_acc = accuracy(logits, output_labels)
                print("Loss", loss, "Acc", train_acc)

        if epoch >= 3:
            if cuda:
                torch.cuda.synchronize()
            dur.append(time.time() - t0)

        # train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = evaluate(model, valid_dataloader)
        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f}"
              f"| Loss {loss:.4f}"
              f" ValAcc {val_acc:.4f} | "
              f"ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

        if train_cfg["early_stopping"]:
            if stopper.step(val_acc, model):
                break

    print()

    if train_cfg["early_stopping"]:
        model.load_state_dict(torch.load("es_checkpoint.pt"))

    acc = evaluate(model, test_dataloader)
    print(f"Test Accuracy {acc:.4f}")


if __name__ == "__main__":
    # Load cmd line args
    Args = parse_args()
    print(Args)
    # Load config file
    Model_cfg = load_config_file(Args.model_cfg)
    Train_cfg = load_config_file(Args.train_cfg)

    main(Args, Model_cfg, Train_cfg)
