"""
Training function in gli for graph classification.

References:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin
"""

import torch
from torch import nn
from torch import optim
import gli
from utils import generate_model, load_config_file,\
                  set_seed, parse_args
from dgl.dataloading import GraphDataLoader


def evaluate(dataloader, device, model):
    """Evaluate model."""
    model.eval()
    total = 0
    total_correct = 0
    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        feat = batched_graph.ndata["NodeFeature"].float()
        total += len(labels)
        logits = model(batched_graph, feat)
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
    acc = 1.0 * total_correct / total
    return acc


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

    dataset = gli.dataloading.get_gli_dataset(args.dataset, args.task)
    train_dataset = dataset[0]
    val_dataset = dataset[1]
    # test_dataset = dataset[2]

    # create dataloader
    train_loader = GraphDataLoader(train_dataset,
                                   batch_size=128,
                                   pin_memory=torch.cuda.is_available())
    val_loader = GraphDataLoader(val_dataset,
                                 batch_size=128,
                                 pin_memory=torch.cuda.is_available())

    # create GIN model
    in_feats = train_dataset[0][0].ndata["NodeFeature"].shape[1]
    n_classes = train_dataset.num_labels
    model = generate_model(args, in_feats, n_classes, **model_cfg)
    if cuda:
        model.cuda()

    # model training/validating
    print("Training...")

    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # training loop
    for epoch in range(train_cfg["max_epoch"]):
        model.train()
        total_loss = 0
        batch = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            feat = batched_graph.ndata["NodeFeature"].float()
            logits = model(batched_graph, feat)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_acc = evaluate(train_loader, device, model)
        valid_acc = evaluate(val_loader, device, model)
        print(f"Epoch {epoch:05d} | Loss {total_loss / (batch + 1):.4f} | \
                Train Acc. {train_acc:.4f} | Validation Acc. {valid_acc:.4f}")


if __name__ == "__main__":
    main()
