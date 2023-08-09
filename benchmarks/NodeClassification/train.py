"""
Train for node classification dataset.

References:
https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/train.py
https://github.com/pyg-team/pytorch_geometric/blob/master/graphgym/main.py
"""


import time
import re
import torch
from torch import nn
import numpy as np
import dgl
import gli
from utils import generate_model, parse_args, Models_need_to_be_densed, \
    load_config_file, check_multiple_split, \
    EarlyStopping, set_seed, check_binary_classification, \
    eval_rocauc, Datasets_need_to_be_undirected, \
    get_label_number
from gli.utils import to_dense


def accuracy(logits, labels):
    """Calculate accuracy."""
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask, eval_func):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return eval_func(logits, labels)


def main():
    """Load dataset and train the model."""
    # Load cmd line args
    args = parse_args()
    print(args)
    # Load config file
    model_cfg = load_config_file(args.model_cfg)
    train_cfg = load_config_file(args.train_cfg)
    set_seed(train_cfg["seed"])

    # load and preprocess dataset
    if args.gpu < 0:
        device = "cpu"
        cuda = False
    else:
        device = args.gpu
        cuda = True

    data = gli.dataloading.get_gli_dataset(args.dataset, args.task,
                                           args.task_id, device,
                                           args.verbose)
    # check EdgeFeature and multi-modal node features
    # edge_cnt = node_cnt = 0
    # if len(data.features) > 1:
    #     for _, element in enumerate(data.features):
    #         if "Edge" in element:
    #             edge_cnt += 1
    #         if "Node" in element:
    #             node_cnt += 1
    #     if edge_cnt >= 1:
    #         raise NotImplementedError("Edge feature is not supported yet.")
    #     elif node_cnt >= 2:
    #         raise NotImplementedError("Multi-modal node features\
    #                                    is not supported yet.")
    g = data[0]
    if train_cfg["to_dense"] or \
       args.model in Models_need_to_be_densed:
        g = to_dense(g)
    # add self loop
    if train_cfg["self_loop"]:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    # convert to undirected set
    if train_cfg["to_undirected"] or \
       args.dataset in Datasets_need_to_be_undirected:
        g = g.to("cpu")
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = g.to(device)

    feature_name = re.search(r".*Node/(\w+)", data.features[0]).group(1)
    label_name = re.search(r".*Node/(\w+)", data.target).group(1)
    features = g.ndata[feature_name]
    labels = g.ndata[label_name].squeeze()
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # for multi-split dataset, choose 0-th split for now
    if check_multiple_split(args.dataset):
        train_mask = train_mask[:, 0]
        val_mask = val_mask[:, 0]
        test_mask = test_mask[:, 0]

    # When labels contains -1, modify masks
    if labels.min() < 0:
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
    label_number = get_label_number(labels)
    if label_number > 1:
        # When binary multi-label, output shape is (batchsize, label_num)
        model = generate_model(args, g, in_feats, label_number, **model_cfg)
        loss_fcn = nn.BCEWithLogitsLoss()
    else:
        # When single-label, output shape is (batchsize, num_classes)
        model = generate_model(args, g, in_feats, n_classes, **model_cfg)
        loss_fcn = nn.CrossEntropyLoss()

    print(model)

    if cuda:
        model.cuda()

    # use optimizer
    if train_cfg["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"])
    elif train_cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"])
    else:
        raise NotImplementedError(f"Optimizer \
            {train_cfg['optimizer']} is not supported.")

    ckpt_name = args.model + "_" + args.dataset + "_"
    ckpt_name += args.train_cfg
    stopper = EarlyStopping(ckpt_name=ckpt_name,
                            early_stop=train_cfg["early_stopping"],
                            patience=50)

    # use rocauc for binary classification
    if check_binary_classification(args.dataset):
        eval_func = eval_rocauc
    else:
        eval_func = accuracy

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

        train_acc = eval_func(logits[train_mask], labels[train_mask])
        val_acc = evaluate(model, features, labels, val_mask, eval_func)
        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f}"
              f"| Loss {loss.item():.4f} | TrainAcc {train_acc:.4f} |"
              f" ValAcc {val_acc:.4f} | "
              f"ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

        if stopper.step(val_acc, model):
            break

    print()

    model.load_state_dict(torch.load(stopper.ckpt_dir))

    acc = evaluate(model, features, labels, test_mask, eval_func)
    val_acc = stopper.best_score
    print(f"Test{acc:.4f},Val{val_acc:.4f}")


if __name__ == "__main__":
    main()
