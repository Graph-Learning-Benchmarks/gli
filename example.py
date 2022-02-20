"""TwoLayerGCN example."""
import glb


METADATA_PATH = "examples/citeseer/metadata.json"
TASK_PATH = "examples/citeseer/task.json"

# pylint: disable=unbalanced-tuple-unpacking
train_set, val_set, test_set = glb.dataset.get_split_dataset(METADATA_PATH,
                                                             TASK_PATH,
                                                             verbose=True)
train_loader = glb.dataloader.NodeDataLoader(train_set)
val_loader = glb.dataloader.NodeDataLoader(val_set)
test_loader = glb.dataloader.NodeDataLoader(test_set)
