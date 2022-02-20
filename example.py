"""TwoLayerGCN example."""
import glb

metadata_path = "examples/citeseer/metadata.json"
task_path = "examples/citeseer/task.json"
train_set, val_set, test_set = glb.dataset.get_split_dataset(metadata_path, task_path, verbose=True)
