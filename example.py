"""Example of data loading for users."""
import glb

g = glb.graph.read_glb_graph(metadata_path="./examples/cora/metadata.json")
task = glb.task.read_glb_task(task_path="./examples/cora/task.json")

dataset = glb.dataloading.combine_graph_and_task(g, task)

print(dataset)
