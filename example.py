"""Example of data loading for users."""
import glb

g = glb.graph.read_glb_graph(metadata_path="examples/ogb_data/graph_prediction/ogbg-molhiv/metadata.json")
task = glb.task.read_glb_task(task_path="examples/ogb_data/graph_prediction/ogbg-molhiv/task.json")

datasets = glb.dataloading.combine_graph_and_task(g, task)

print(datasets)
