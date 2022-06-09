from dgl.nn.pytorch import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../..')
import glb
import time
import numpy as np

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.g = g
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads)
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(self.g, h)
        h = F.elu(h)
        h = h.flatten(1)
        h = self.layer2(self.g, h)
        h = h.mean(1)
        return h

def train(g, model):
    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    features = g.ndata['NodeFeature']
    labels = g.ndata['NodeLabel']
    train_mask = g.ndata['train_set']
    val_mask = g.ndata['val_set']
    test_mask = g.ndata['test_set']

    # main loop
    dur = []
    for epoch in range(30):
        if epoch >= 3:
            t0 = time.time()

        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))


citeseer_metadata_path = "../../examples/citeseer/metadata.json"
citeseer_task_path = "../../examples/citeseer/task.json"
cora_metadata_path = "../../examples/cora/metadata.json"
cora_task_path = "../../examples/cora/task.json"
pubmed_metadata_path = "../../examples/pubmed/metadata.json"
pubmed_task_path = "../../examples/pubmed/task.json"
ogbn_arxiv_metadata_path = "../../examples/ogb_data/node_prediction/ogbn-arxiv/metadata.json"
ogbn_arxiv_task_path = "../../examples/ogb_data/node_prediction/ogbn-arxiv/task.json"
ogbn_mag_metadata_path = "../../examples/ogb_data/node_prediction/ogbn-mag/metadata.json"
ogbn_mag_task_path = "../../examples/ogb_data/node_prediction/ogbn-mag/task.json"

g = glb.graph.read_glb_graph(metadata_path=cora_metadata_path)
task = glb.task.read_glb_task(task_path=cora_task_path)

dataset = glb.dataloading.combine_graph_and_task(g, task)
g = dataset[0]

# create the model, 2 heads, each head has hidden size 8

# train with cpu
model = GAT(g,
          in_dim=g.ndata['NodeFeature'].shape[1],
          hidden_dim=8,
          out_dim=7,
          num_heads=2)
train(g, model)

# train with gpu
# g = g.to('cuda')
# model = GAT(g,
#           in_dim=g.ndata['NodeFeature'].shape[1],
#           hidden_dim=8,
#           out_dim=7,
#           num_heads=2)
# train(g, model)
