"""Tagging graph datasets."""
import dgl
import argparse
import networkx as nx
import numpy as np
import glb


def edge_density(g):
    """Compute the edge density."""
    nx_g = dgl.to_networkx(g)
    edge_den = nx.density(nx_g)

    return edge_den


def avg_degree(g):
    """Compute the average degree."""
    nx_g = dgl.to_networkx(g)

    degree = nx_g.in_degree()
    degree_list = []
    for _, d in degree:
        degree_list.append(d)
    av_degree = sum(degree_list) / len(degree_list)

    return av_degree


def avg_cluster_coefficient(g):
    """Compute the average clustering coefficient."""
    nx_g = dgl.to_networkx(g)
    # to Digraph
    nx_g = nx.DiGraph(nx_g)
    av_local_clustering_coeff = nx.average_clustering(nx_g)
    # av_local_clustering_coeff = sum(local_clustering_coefficient.values())
    # / len(local_clustering_coefficient)

    return av_local_clustering_coeff


def diameter(g):
    """Compute the diameters (need to be connected)."""
    nx_g = dgl.to_networkx(g)
    nx_g = nx.Graph(nx_g)
    if nx.is_connected(nx_g):
        return nx.diameter(nx_g)
    # else:
    #     return max(max(j.values()) for (i, j) in nx.shortest_path_length(g))


def avg_shortest_path(g):
    """Compute the average shortest path (need to be connected)."""
    # nx_g = dgl.to_networkx(g)
    return nx.average_shortest_path_length(g)


def edge_reciprocity(g):
    """Compute the edge reciprocity (need to be connected)."""
    nx_g = dgl.to_networkx(g)

    return nx.overall_reciprocity(nx_g)


def relative_largest_cc(g):
    """Compute relative size of the largest connected component."""
    # will disregard the edge direction!
    nx_g = dgl.to_networkx(g)
    nx_g = nx.Graph(nx_g)
    lcc = sorted(nx.connected_components(nx_g), key=len, reverse=True)
    lcc_size = nx.number_of_nodes(nx_g.subgraph(lcc[0]))
    return lcc_size / nx.number_of_nodes(nx_g)


def relative_largest_scc(g):
    """Compute relative size of the largest strongly connected component."""
    # consider directed network only
    nx_g = dgl.to_networkx(g)
    lcc = sorted(nx.strongly_connected_components(nx_g), key=len, reverse=True)
    lcc_size = nx.number_of_nodes(nx_g.subgraph(lcc[0]))
    return lcc_size / nx.number_of_nodes(nx_g)


def gini_array(array):
    """Compute the gini index of a given array."""
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    return np.sum((2 * index - array.shape[0] - 1) * array) / \
                 (array.shape[0] * np.sum(array))


def gini_degree(g):
    """Compute the gini index of the degree sequence."""
    nx_g = dgl.to_networkx(g)
    degree_sequence = [d for n, d in nx_g.degree()]
    return gini_array(degree_sequence)


def gini_coreness(g):
    """Compute the gini index of the coreness sequence."""
    nx_g = dgl.to_networkx(g)
    # convert the MultiDiGraph to Digraph
    nx_g = nx.DiGraph(nx_g)
    # remove potential self-loops
    nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
    core_sequence = list(nx.core_number(nx_g).values())
    return gini_array(core_sequence)


def degeneracy(g):
    """Compute the Degeneracy."""
    nx_g = dgl.to_networkx(g)
    # convert the MultiDiGraph to Digraph
    nx_g = nx.DiGraph(nx_g)
    # remove potential self-loops
    nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
    return max(nx.core_number(nx_g).values())


def degree_assortativity(g):
    """Compute the degree assortativity coefficient."""
    nx_g = dgl.to_networkx(g)
    return nx.degree_assortativity_coefficient(nx_g)


def check_direct(g):
    """Check the graph is directed or not."""
    # to examine whether all the edges are bi-directed
    # print("dgl edge number: ", g.edges()[0].size(dim=0))
    nx_g = dgl.to_networkx(g)
    node_num = nx_g.number_of_nodes()
    count = 0
    for i in range(node_num):
        for _, neighbors in nx_g.edges(i):
            if (neighbors, i) in nx_g.edges(neighbors):
                # print("bi-directed edges exist")
                count += 1
    # print("number of indirected edges: ", count)
    return g.edges()[0].size(dim=0) != count


def prepare_dataset(metadata_path, task_path):
    """Prepare dataset."""
    g = glb.graph.read_glb_graph(metadata_path=metadata_path)
    task = glb.task.read_glb_task(task_path=task_path)
    datasets = glb.dataloading.combine_graph_and_task(g, task)
    return g, task, datasets


def main():
    """Run main function."""
    # parsing the input command
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    metadata_name, path_name = args.metadata, args.task
    # read in the graph dataset
    g, task, datasets = prepare_dataset(metadata_name, path_name)
    # print input graph, task, and dataset information
    print(g)
    print(task)
    print(datasets)
    # print graph tags
    print(f"Directed: {check_direct(g)}")
    print(f"Edge Density: {edge_density(g):.6f}")
    print(f"Average Degree: {avg_degree(g):.6f}")
    print(f"Relative Size of Largest Connected Component: "
          f"{relative_largest_cc(g):.6f}")
    print(f"Relative Size of Largest Strongly Connected "
          f"Component: {relative_largest_cc(g):.6f}")
    print(f"Average Clustering Coefficient: "
          f"{avg_cluster_coefficient(g):.6f}")
    print(f"Diameter: {diameter(g)}")
    print(f"Edge Reciprocity: {edge_reciprocity(g)}")
    print(f"Gini Coefficient of Degree: {gini_degree(g):.6f}")
    print(f"Gini Coefficient of Coreness: {gini_coreness(g):.6f}")
    print(f"Degeneracy: {degeneracy(g)}")
    print(f"Degree Assortativity: {degree_assortativity(g):.6f}")


if __name__ == "__main__":
    main()
