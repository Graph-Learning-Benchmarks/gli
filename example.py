"""usage: example.py [-h] [-g GRAPH] [-t TASK] [-d DEVICE] [-v].

Demo data loading.

optional arguments:
  -h, --help            show this help message and exit
  -g GRAPH, --graph GRAPH
                        The graph to be loaded.
  -t TASK, --task TASK  The predefined task type.
  -d DEVICE, --device DEVICE
  -v, --verbose
"""
import argparse
import os
import time
import tracemalloc

import gli


def main():
    """Run main function."""
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Demo data loading.")
    parser.add_argument("-g",
                        "--graph",
                        type=str,
                        default="cora",
                        help="The graph to be loaded.")
    parser.add_argument("-t",
                        "--task",
                        type=str,
                        default="NodeClassification",
                        help="The predefined task type.")
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    args = parser.parse_args()

    is_get = not (args.graph.endswith(".json") and args.task.endswith(".json"))

    # Find a proper task if user does not specify.
    if args.task is None:
        for f in os.listdir(os.path.join("datasets", args.graph)):
            if f.endswith(".json") and f.startswith("task"):
                args.task = f[:-5]

    # Download graph and task data, with profiling.
    # The following three commands are equivalent to
    # dataset = gli.dataloading.get_gli_dataset(args.graph, args.task,
    #                                           args.device, args.verbose)
    with Profiler("> Graph(s) loading"):
        if is_get:
            g = gli.dataloading.get_gli_graph(args.graph,
                                              device=args.device,
                                              verbose=args.verbose)
        else:
            g = gli.dataloading.read_gli_graph(args.graph,
                                               device=args.device,
                                               verbose=args.verbose)

    with Profiler("> Task loading"):
        if is_get:
            task = gli.dataloading.get_gli_task(args.graph,
                                                args.task,
                                                verbose=args.verbose)
        else:
            task = gli.dataloading.read_gli_task(args.task,
                                                 verbose=args.verbose)

    with Profiler("> Combining graph(s) and task"):
        dataset = gli.dataloading.combine_graph_and_task(g, task)

    print(dataset)


class Profiler:
    """Tic-Toc timer."""

    def __init__(self, func_name):
        """Initialize tic by current time."""
        self.t = time.time()
        self.func_name = func_name

    def __enter__(self):
        """Reset tic."""
        self.t = time.time()
        tracemalloc.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Return time elaspe between tic and toc, and reset tic."""
        elapse = time.time() - self.t
        mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()
        print(f"{self.func_name} takes {elapse:.4f} seconds and"
              f" uses {mem:.4f} MB.")


if __name__ == "__main__":
    main()
