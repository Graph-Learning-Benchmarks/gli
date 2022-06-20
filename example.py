"""Example of data loading for users."""
import argparse
import time
import glb
import tracemalloc

TASKS = [
    "NodeClassification", "TimeDepenedentLinkPrediction", "GraphClassification"
]
PATHS = [("examples/cora/metadata.json", "examples/cora/task.json"),
         ("examples/ogbl-collab/metadata.json",
          "examples/ogbl-collab/task_runtime_sampling.json"),
         ("examples/ogbg-molhiv/metadata.json",
          "examples/ogbg-molhiv/task.json")]


class Timer:
    """Tic-Toc timer."""
    def __init__(self):
        """Initialize tic by current time."""
        self._tic = time.time()

    def tic(self):
        """Reset tic."""
        self._tic = time.time()
        return self._tic

    def toc(self):
        """Return time elaspe between tic and toc, and reset tic."""
        last_tic = self._tic
        self._tic = time.time()
        return self._tic - last_tic


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=TASKS, default=TASKS[2])
args = parser.parse_args()
task_name = args.task
clock = Timer()


def prepare_dataset(metadata_path, task_path):
    """Prepare dataset."""
    clock.tic()
    tracemalloc.start()
    g = glb.graph.read_glb_graph(metadata_path=metadata_path)
    print(f"Read graph data from {metadata_path} in {clock.toc():.2f}s.")
    task = glb.task.read_glb_task(task_path=task_path)
    print(f"Read task specification from {task_path} in {clock.toc():.2f}s.")
    datasets = glb.dataloading.combine_graph_and_task(g, task)
    print(f"Combine graph and task into dataset(s) in {clock.toc():.2f}s.")
    mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    print(f"Peak memory usage: {mem:.2f}MB.")
    tracemalloc.stop()
    return g, task, datasets


def main():
    """Run main function."""
    path_dict = dict(zip(TASKS, PATHS))
    g, task, datasets = prepare_dataset(*path_dict[task_name])
    if isinstance(g, list):
        print(f"Dataset contains {len(g)} graphs.")
    else:
        print(g)
    print(task)
    print(datasets)


if __name__ == "__main__":
    main()
