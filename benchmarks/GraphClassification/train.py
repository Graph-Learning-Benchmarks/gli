"""
Training function in gli for graph classification.

References:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin
"""

import torch
from torch import nn
from torch import optim
import gli
from utils import generate_model, load_config_file, \
                  set_seed, parse_args, EarlyStopping, \
                  check_binary_classification, eval_rocauc, eval_acc, \
                  get_label_number, eval_ap
from dgl.dataloading import GraphDataLoader
from dgl import add_self_loop


def evaluate(dataloader, device, model, eval_func):
    """Evaluate model."""
    model.eval()
    y_true = []
    y_pred = []
    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        feat = batched_graph.ndata["NodeFeature"].float()
        logits = model(batched_graph, feat)

        y_true.append(labels.detach().cpu())
        y_pred.append(logits.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    return eval_func(y_pred, y_true)


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

    print("Training with a fixed epsilon = 0")

    dataset = gli.dataloading.get_gli_dataset(args.dataset, args.task,
                                              args.task_id, device,
                                              args.verbose)
    train_dataset = dataset[0]
    val_dataset = dataset[1]
    test_dataset = dataset[2]

    in_feats = train_dataset[0][0].ndata["NodeFeature"].shape[1]
    n_classes = train_dataset.num_labels

    if train_cfg["self_loop"]:
        train_dataset.graphs = [add_self_loop(x) for x in train_dataset.graphs]
        val_dataset.graphs = [add_self_loop(x) for x in val_dataset.graphs]
        test_dataset.graphs = [add_self_loop(x) for x in test_dataset.graphs]

    # create dataloader
    train_loader = GraphDataLoader(train_dataset,
                                   batch_size=train_cfg["batch_size"],
                                   shuffle=True,
                                   pin_memory=torch.cuda.is_available())
    val_loader = GraphDataLoader(val_dataset,
                                 batch_size=train_cfg["batch_size"],
                                 shuffle=False,
                                 pin_memory=torch.cuda.is_available())
    test_loader = GraphDataLoader(test_dataset,
                                  batch_size=train_cfg["batch_size"],
                                  shuffle=False,
                                  pin_memory=torch.cuda.is_available())

    # create model

    label_number = get_label_number(train_loader)
    if label_number > 1:
        # When binary multi-label, output shape is (batchsize, label_num)
        model = generate_model(args, in_feats, label_number, **model_cfg)
        loss_fcn = nn.BCEWithLogitsLoss()
    else:
        # When single-label, output shape is (batchsize, num_classes)
        model = generate_model(args, in_feats, n_classes, **model_cfg)
        loss_fcn = nn.CrossEntropyLoss()

    if cuda:
        model.cuda()

    # model training/validating
    print("Training...")

    # loss function, optimizer, scheduler and early stopping
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    ckpt_name = args.model + "_" + args.dataset + "_"
    ckpt_name += args.train_cfg
    stopper = EarlyStopping(ckpt_name=ckpt_name,
                            early_stop=train_cfg["early_stopping"],
                            patience=50)

    if check_binary_classification(args.dataset):
        if label_number > 1:
            eval_func = eval_ap
        else:
            eval_func = eval_rocauc
    else:
        eval_func = eval_acc

    # training loop
    for epoch in range(train_cfg["max_epoch"]):
        model.train()
        total_loss = 0
        batch = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            # batched_graph = dgl.add_self_loop(batched_graph)
            labels = labels.to(device)
            feat = batched_graph.ndata["NodeFeature"].float()
            logits = model(batched_graph, feat)
            if label_number > 1:
                # When binary multi-label, use BCE loss
                is_labeled = ~torch.isnan(torch.tensor(labels))
                loss = loss_fcn(logits[is_labeled], labels.float()[is_labeled])
            else:
                # Otherwise, use CE loss
                loss = loss_fcn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_acc = evaluate(train_loader, device, model, eval_func)
        valid_acc = evaluate(val_loader, device, model, eval_func)
        print(f"Epoch {epoch:05d} | Loss {total_loss / (batch + 1):.4f} | \
                Train Acc. {train_acc:.4f} | Validation Acc. {valid_acc:.4f}")

        if stopper.step(valid_acc, model):
            break

    model.load_state_dict(torch.load(stopper.ckpt_dir))
    acc = evaluate(test_loader, device, model, eval_func)
    val_acc = stopper.best_score
    print(f"Test{acc:.4f},Val{val_acc:.4f}")


if __name__ == "__main__":
    main()
