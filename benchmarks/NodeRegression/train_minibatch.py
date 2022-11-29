"""
Train for node classification dataset.

References:
https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/train.py
https://github.com/pyg-team/pytorch_geometric/blob/master/graphgym/main.py
https://docs.dgl.ai/guide/minibatch-node.html?highlight=sampling
"""


import time
import re
import torch
from torch import nn
import numpy as np
import dgl
import gli
from utils import generate_model, parse_args, \
                  load_config_file, check_multiple_split,\
                  EarlyStopping, set_seed
from gli.utils import to_dense
from dgl.dataloading import MultiLayerFullNeighborSampler as Sampler


# def accuracy(logits, labels):
#     """Calculate accuracy."""
#     _, indices = torch.max(logits, dim=1)
#     correct = torch.sum(indices == labels)
#     return correct.item() * 1.0 / len(labels)


def evaluate(model, dataloader, eval_func):
    """Evaluate model."""
    model.eval()
    ys = []
    y_hats = []
    for _, _, blocks in dataloader:
        with torch.no_grad():
            input_features = blocks[0].srcdata["NodeFeature"]
            ys.append(blocks[-1].dstdata["NodeLabel"])
            y_hats.append(model(blocks, input_features))
    return eval_func(torch.cat(y_hats).squeeze(), torch.cat(ys).float())


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

    g = to_dense(g)
    # add self loop
    if train_cfg["self_loop"]:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    feature_name = re.search(r".*Node/(\w+)", data.features[0]).group(1)
    features = g.ndata[feature_name]

    # for multi-split dataset, choose 0-th split for now
    if check_multiple_split(args.dataset):
        train_mask = train_mask[:, 0]
        val_mask = val_mask[:, 0]
        test_mask = test_mask[:, 0]

    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = g.number_of_edges()

    sampler = Sampler(model_cfg["num_layers"])
    train_dataloader = dgl.dataloading.DataLoader(
        g, indice["train_set"], sampler,
        batch_size=train_cfg["batch_size"],
        device=device,
        shuffle=True,
        drop_last=False)

    valid_dataloader = dgl.dataloading.DataLoader(
            g, indice["val_set"], sampler,
            device=device,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            drop_last=False)

    test_dataloader = dgl.dataloading.DataLoader(
            g, indice["test_set"], sampler,
            device=device,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            drop_last=False)

    print(f"""----Data statistics------'
      #Edges {n_edges}
      #Classes {n_classes}""")

    # create model, supporting only single label task
    label_number = 1
    model = generate_model(args, g, in_feats, label_number, **model_cfg)

    print(model)
    if cuda:
        model.cuda()

    # create loss function and evalution function
    if train_cfg["loss_fcn"] == "mse":
        eval_func = loss_fcn = nn.MSELoss()
    elif train_cfg["loss_fcn"] == "mae":
        eval_func = loss_fcn = nn.L1Loss()
    else:
        raise NotImplementedError(f"Loss function \
            {train_cfg['loss_fcn']} is not supported.")

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"])

    ckpt_name = args.model + "_" + args.dataset + "_"
    ckpt_name += args.train_cfg
    stopper = EarlyStopping(ckpt_name=ckpt_name,
                            early_stop=train_cfg["early_stopping"],
                            patience=50)

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

            # When labels contains -1, modify labels
            if min(output_labels) < 0:
                output_labels = output_labels * (output_labels >= 0)

            logits = model(blocks, input_features)
            loss = loss_fcn(logits.squeeze(), output_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if it % 20 == 0:
                # train_acc = loss_fcn(logits.squeeze(), output_labels.float())
                print("Loss", loss.item())

        if epoch >= 3:
            if cuda:
                torch.cuda.synchronize()
            dur.append(time.time() - t0)

        val_loss = evaluate(model, valid_dataloader, eval_func)
        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f}"
              f"| Loss {loss:.4f} | "
              f" Val Loss {val_loss.item():.4f} | "
              f"ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

        if stopper.step(val_loss.item(), model):
            break

    print()

    model.load_state_dict(torch.load(stopper.ckpt_dir))

    loss = evaluate(model, test_dataloader, eval_func)
    val_loss = stopper.best_score
    print(f"Test loss {loss:.4f}, Val loss {val_loss:.4f}")


if __name__ == "__main__":
    main()
