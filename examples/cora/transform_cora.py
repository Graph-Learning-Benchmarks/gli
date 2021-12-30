import os
import json
from dgl.data import CoraGraphDataset
from scipy.sparse import csr_matrix, save_npz


root = "data"
if not os.path.exists(root):
    os.makedirs(root)

dataset = CoraGraphDataset()
graph = dataset[0]

meta_data = {}
meta_data["description"] = "CORA dataset."
meta_data["attributes"] = {}
meta_data["attributes"]["node_feature"] = {}
meta_data["attributes"]["node_feature"]["type"] = "SparseTensor"
meta_data["attributes"]["node_feature"]["file"] = "cora_feat.npz"
# meta_data["attributes"]["node_feature"]["index"] = "cora_feat_index.csv"  # optional
meta_data["attributes"]["edge"] = {}
meta_data["attributes"]["edge"]["type"] = "Edge"
meta_data["attributes"]["node_list"] = {}
meta_data["attributes"]["node_list"]["type"] = "NodeList"
with open(os.path.join(root, "metadata.json"), "w") as fout:
    json.dump(meta_data, fout)

task_config = {}
task_config["description"] = "Node classification on CORA dataset. Planetoid split."
task_config["type"] = "NodeClassification"
task_config["feature"] = ["node_feature"]
task_config["target"] = "class"
task_config["num_classes"] = 7  # optional
train_set = graph.ndata["train_mask"].nonzero().squeeze().tolist()
task_config["train_set"] = ["n%d" % i for i in train_set]
val_set = graph.ndata["val_mask"].nonzero().squeeze().tolist()
task_config["val_set"] = ["n%d" % i for i in val_set]
test_set = graph.ndata["test_mask"].nonzero().squeeze().tolist()
task_config["test_set"] = ["n%d" % i for i in test_set]
with open(os.path.join(root, "task_planetoid.json"), "w") as fout:
    json.dump(task_config, fout)

feat = csr_matrix(graph.ndata["feat"].numpy())
save_npz(os.path.join(root, "cora_feat.npz"), feat)

objects = []
for i in range(graph.num_nodes()):
    obj = {}
    obj["id"] = "n%d" % i
    obj["node_feature"] = i
    obj["class"] = graph.ndata["label"][i].item()
    objects.append(obj)

n1, n2 = graph.edges()
for i in range(n1.size(0)):
    obj = {}
    obj["id"] = "e%d" % i
    obj["edge"] = ["n%d" % n1[i].item(), "n%d" % n2[i].item()]
    objects.append(obj)

obj = {}
obj["id"] = "g0"
obj["node_list"] = ["n%d" % i for i in range(graph.num_nodes())]
objects.append(obj)

with open(os.path.join(root, "data.json"), "w") as fout:
    json.dump(objects, fout)
