"""Test if the Knowledge Graph (KG) dataset can be trained two epochs."""
import gli
import pytest
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from utils import find_datasets
from kg_utils import TransE, KGDataset
from training_utils import check_dataset_task


@pytest.mark.parametrize("dataset_name", find_datasets())
def test_relation_prediction(dataset_name):
    """Test if the KG dataset can be trained for two epochs."""
    # only do the test on KG Datasets
    if not check_dataset_task(dataset_name, "KGRelationPrediction"):
        return

    device = "cpu"

    data = gli.dataloading.get_gli_dataset(
        dataset_name, "KGRelationPrediction", device=device)

    graph = data[0]
    if "NodeFeature" in graph.ndata:
        raise NotImplementedError(
            "KG with node features is not supported yet.")
    if "EdgeFeature" in graph.ndata:
        raise NotImplementedError(
            "KG with edge features is not supported yet.")
    n_negatives = 5
    batch_size = 32
    n_entities = graph.num_nodes()
    n_edges = graph.num_edges()
    n_relations = data.num_relations
    margin = torch.nn.Parameter(torch.Tensor([5.0]))

    train_relations = data.get_train_graph().edata["EdgeClass"]
    train_heads, train_tails = data.get_train_graph().edges()
    training_data = KGDataset(train_heads, train_tails, train_relations)
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"""----Data statistics------'
      #Edges {n_edges}
      #Edge Classes {n_relations}
      #Negative samples {n_negatives}
      #Train samples {len(train_relations)}
      """)

    model = TransE(n_entities, n_relations, dim=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    dur = []
    for epoch in range(2):
        model.train()
        t0 = time.time()

        # forward
        for data in train_dataloader:

            # n_relations - 1 as it indexes from 0
            negatives = torch.randint(
                low=0, high=n_relations-1, size=(n_negatives * batch_size,))
            data["batch_h"] = data["batch_h"].repeat(n_negatives+1)
            data["batch_t"] = data["batch_t"].repeat(n_negatives+1)
            data["batch_r"] = torch.cat((data["batch_r"], negatives))

            score = model(data)
            p_score = score[:batch_size]
            n_score = score[batch_size:]
            p_score = p_score.view(-1, batch_size).permute(1, 0)
            n_score = n_score.view(-1, batch_size).permute(1, 0)

            loss = (torch.max(p_score - n_score, -margin)).mean() + margin

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dur.append(time.time() - t0)

        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} ")
        print(f"| TrainLoss {loss.item():.4f} |")

    print("The dataset has successfully trained \
        on TransE model for two epoches.")
    print("Test passed.")


@pytest.mark.parametrize("dataset_name", find_datasets())
def test_entity_prediction(dataset_name):
    """Test if the KG dataset can be trained for two epochs."""
    # only do the test on KG Datasets
    if not check_dataset_task(dataset_name, "KGEntityPrediction"):
        return

    device = "cpu"

    data = gli.dataloading.get_gli_dataset(
        dataset_name, "KGEntityPrediction", device=device)

    graph = data[0]
    if "NodeFeature" in graph.ndata:
        raise NotImplementedError(
            "KG with node features is not supported yet.")
    if "EdgeFeature" in graph.ndata:
        raise NotImplementedError(
            "KG with edge features is not supported yet.")
    n_negatives = 5
    batch_size = 32
    n_entities = graph.num_nodes()
    n_edges = graph.num_edges()
    n_relations = data.num_relations
    margin = torch.nn.Parameter(torch.Tensor([5.0]))

    train_relations = data.get_train_graph().edata["EdgeClass"]
    train_heads, train_tails = data.get_train_graph().edges()
    training_data = KGDataset(train_heads, train_tails, train_relations)
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"""----Data statistics------'
      #Edges {n_edges}
      #Edge Classes {n_relations}
      #Negative samples {n_negatives}
      #Train samples {len(train_relations)}
      """)

    model = TransE(n_entities, n_relations, dim=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    dur = []
    for epoch in range(2):
        model.train()
        t0 = time.time()

        # forward
        for data in train_dataloader:

            # n_entities - 1 as it indexes from 0
            negatives = torch.randint(
                low=0, high=n_entities-1, size=(n_negatives * batch_size,))
            data["batch_h"] = data["batch_h"].repeat(n_negatives+1)
            data["batch_t"] = torch.cat((data["batch_t"], negatives))
            data["batch_r"] = data["batch_r"].repeat(n_negatives+1)

            score = model(data)
            p_score = score[:batch_size]
            n_score = score[batch_size:]
            p_score = p_score.view(-1, batch_size).permute(1, 0)
            n_score = n_score.view(-1, batch_size).permute(1, 0)

            loss = (torch.max(p_score - n_score, -margin)).mean() + margin

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dur.append(time.time() - t0)

        print(f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} ")
        print(f"| TrainLoss {loss.item():.4f} |")

    print("The dataset has successfully trained \
        on TransE model for two epoches.")
    print("Test passed.")
