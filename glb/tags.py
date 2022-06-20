"""Tagging network datasets"""
import dgl
import argparse
import networkx as nx
import numpy as np
import time
import glb
import tracemalloc

TASKS = [
    "NodeClassification", "TimeDepenedentLinkPrediction", "GraphClassification"
]
PATHS = [
    ("../examples/cora/metadata.json", "../examples/cora/task.json"),
    ("../examples/ogb_data/link_prediction/ogbl-collab/metadata.json",
     "../examples/ogb_data/link_prediction/ogbl-collab/task_runtime_sampling.json"
     ),
    ("../examples/ogb_data/graph_prediction/ogbg-molhiv/metadata.json",
     "../examples/ogb_data/graph_prediction/ogbg-molhiv/task.json")
]


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


def edge_density(g):
    nx_g = dgl.to_networkx(g)
    edge_den = nx.density(nx_g)

    return edge_den


def avg_degree(g):
    nx_g = dgl.to_networkx(g)
    degree = nx_g.degree()
    degree_list = []
    for (n, d) in degree:
        degree_list.append(d)
    av_degree = sum(degree_list) / len(degree_list)

    return av_degree


def avg_cluster_coefficient(g):
    nx_g = dgl.to_networkx(g)
    # to Digraph
    nx_g = nx.DiGraph(nx_g)
    av_local_clustering_coeff = nx.average_clustering(nx_g)
    # av_local_clustering_coeff = sum(local_clustering_coefficient.values()) / len(local_clustering_coefficient)

    return av_local_clustering_coeff


def diameter(g):
    nx_g = dgl.to_networkx(g)
    nx_g = nx.Graph(nx_g)
    if nx.is_connected(nx_g):
        return nx.diameter(nx_g)
    else:
        return max([max(j.values()) for (i, j) in nx.shortest_path_length(g)])


def avg_shortest_path(g):
    nx_g = dgl.to_networkx(g)
    return nx.average_shortest_path_length(g)


def edge_reciprocity(g):
    nx_g = dgl.to_networkx(g)

    return nx.reciprocity(nx_g)


def gini_array(array):
    array += np.finfo(np.float32).eps
    array = np.sort(array)
    n = array.shape[0]
    index = np.arange(1, n + 1)

    return np.sum((2 * index - n - 1) * array) / (n * np.sum(array))


def gini_degree(g):
    nx_g = dgl.to_networkx(g)
    degree_sequence = [d for n, d in nx_g.degree()]
    return gini_array(degree_sequence)


# def gini_coreness(g):

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
    mem = tracemalloc.get_traced_memory()[1]/(1024*1024)
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
        print(f"edge density: {edge_density(g):.6f}")
        print(f"average degree: {avg_degree(g):.6f}")
        print(f"average local clustering coefficient: {avg_cluster_coefficient(g):.6f}")
        #print(f"diameter: {glb.metric.diameter(g)}")
        #print(f"edge reciprocity: {glb.metric.edge_reciprocity(g)}")
        print(f"gini degree: {gini_degree(g):.6f}")
    #print(task)
    #print(datasets)


if __name__ == "__main__":
    main()