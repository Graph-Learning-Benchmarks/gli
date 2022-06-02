import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/huangjin/GLB-Repo')
import glb
from dgl.nn import GraphConv

metadata_path = "/home/huangjin/GLB-Repo/examples/cora/metadata.json"
task_path = "/home/huangjin/GLB-Repo/examples/cora/task.json"
g = glb.graph.read_glb_graph(metadata_path=metadata_path)
task = glb.task.read_glb_task(task_path=task_path)

dataset = glb.dataloading.combine_graph_and_task(g, task)
g = dataset[0]

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# Create the model with given dimensions
model = GCN(g.ndata['NodeFeature'].shape[1], 16, dataset._num_labels)

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['NodeFeature']
    labels = g.ndata['NodeLabel']
    train_mask = g.ndata['train_set']
    val_mask = g.ndata['val_set']
    test_mask = g.ndata['test_set']
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

# train with cpu
# model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
# train(g, model)

# train with gpu
g = g.to('cuda')
model = GCN(g.ndata['NodeFeature'].shape[1], 16, dataset._num_labels).to('cuda')
train(g, model)