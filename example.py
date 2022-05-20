"""Example of data loading for users."""
import glb
import argparse

TASKS = [
    "NodeClassification", "TimeDepenedentLinkPrediction", "GraphClassification"
]
PATHS = [
    ("examples/cora/metadata.json", "examples/cora/task.json"),
    ("examples/ogb_data/link_prediction/ogbl-collab/metadata.json",
     "examples/ogb_data/link_prediction/ogbl-collab/task_runtime_sampling.json"
     ),
    ("examples/ogb_data/graph_prediction/ogbg-molhiv/metadata.json",
     "examples/ogb_data/graph_prediction/ogbg-molhiv/task.json")
]

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=TASKS, default=TASKS[0])
args = parser.parse_args()
task_name = args.task


def prepare_dataset(metadata_path, task_path):
    """Prepare dataset."""
    g = glb.graph.read_glb_graph(metadata_path=metadata_path)
    task = glb.task.read_glb_task(task_path=task_path)
    datasets = glb.dataloading.combine_graph_and_task(g, task)
    return g, task, datasets


def main():
    """Run main function."""
    path_dict = dict(zip(TASKS, PATHS))
    g, task, datasets = prepare_dataset(*path_dict[task_name])
    print(g)
    print(task)
    print(datasets)


if __name__ == "__main__":
    main()
