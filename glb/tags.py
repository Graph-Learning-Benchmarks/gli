"""Tagging graph datasets."""
import dgl
import argparse
import networkx as nx
import numpy as np
import glb
import powerlaw


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
    # transform from MultiDigraph to Digraph
    nx_g = nx.DiGraph(nx_g)
    av_local_clustering_coeff = nx.average_clustering(nx_g)
    return av_local_clustering_coeff


def diameter(g):
    """Compute the diameters (need to be connected)."""
    nx_g = dgl.to_networkx(g)
    # nx_g = nx.Graph(nx_g)
    if nx.is_strongly_connected(nx_g):
        return nx.diameter(nx_g)
    else:
        # if the graph is not strongly connected,
        # we report the diameter in the LSCC
        cc = sorted(nx.strongly_connected_components(nx_g), key=len,
                    reverse=True)
        lcc = nx_g.subgraph(cc[0])
        return nx.diameter(lcc)


def pseudo_diameter(g):
    """Compute a lower bound on the diameter."""
    nx_g = dgl.to_networkx(g)
    if nx.is_strongly_connected(nx_g):
        return nx.algorithms.approximation.diameter(nx_g)
    else:
        # if the graph is not strongly connected,
        # we report the diameter in the LCC
        cc = sorted(nx.strongly_connected_components(nx_g), key=len,
                    reverse=True)
        lcc = nx_g.subgraph(cc[0])
        return nx.algorithms.approximation.diameter(lcc)


def avg_shortest_path(g):
    """Compute the average shortest path."""
    nx_g = dgl.to_networkx(g)
    if nx.is_strongly_connected(nx_g):
        return nx.average_shortest_path_length(nx_g)
    else:
        # if the graph is not strongly connected,
        # we report the avg shortest path in the LSCC
        cc = sorted(nx.strongly_connected_components(nx_g),
                    key=len, reverse=True)
        lcc = nx_g.subgraph(cc[0])
        return nx.average_shortest_path_length(lcc)


def edge_reciprocity(g):
    """Compute the edge reciprocity."""
    nx_g = dgl.to_networkx(g)
    return nx.overall_reciprocity(nx_g)


def relative_largest_cc(g):
    """Compute relative size of the largest connected component."""
    # will disregard the edge direction
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
    """Compute the Gini Index of a given array."""
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    return np.sum((2 * index - array.shape[0] - 1)
                  * array) / (array.shape[0] * np.sum(array))


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
    # to examine whether all the edges are bi-directed edges
    nx_g = dgl.to_networkx(g)
    for e in nx_g.edges():
        if (e[1], e[0]) not in nx_g.edges():
            return True
    return False


def edge_homogeneity(g):
    """Compute the edge homogeneity."""
    # proportion of the edges that connect nodes with the same class label
    count = 0
    nx_g = dgl.to_networkx(g, node_attrs=["NodeLabel"])
    edge_num = nx_g.number_of_edges()
    for e in nx_g.edges(data=True):
        if nx_g.nodes[e[0]]["NodeLabel"] == nx_g.nodes[e[1]]["NodeLabel"]:
            count += 1

    return count / edge_num


def power_law_expo(g):
    """Fit the Power-Law Distribution on degree sequence."""
    nx_g = dgl.to_networkx(g)
    degree_sequence = [d for n, d in nx_g.degree()]
    degree_sequence = np.sort(degree_sequence)
    fit = powerlaw.Fit(degree_sequence, verbose=False)
    return fit.power_law.alpha


def pareto_expo(g):
    """Get the Pareto Exponent."""
    nx_g = dgl.to_networkx(g)
    # remove nodes that have 0 degree
    remove_nodes = [node for node, degree in
                    dict(nx_g.degree()).items() if degree == 0]
    nx_g.remove_nodes_from(remove_nodes)
    degree_sequence = [d for n, d in nx_g.degree()]
    degree_sequence = np.sort(degree_sequence)
    dmin = np.min(degree_sequence)
    alpha = len(degree_sequence) / np.sum(np.log(degree_sequence / dmin))
    return alpha


def prepare_dataset(dataset, task):
    """Prepare datasets."""
    glb_graph = glb.dataloading.get_glb_graph(dataset)
    glb_task = glb.dataloading.get_glb_task(dataset, task)
    glb_graph_task = glb.dataloading.combine_graph_and_task(
        glb_graph, glb_task)
    return glb_graph, glb_task, glb_graph_task


def main():
    """Run main function."""
    # parsing the input command
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    dataset_name, task_name = args.dataset, args.task
    # read in the graph dataset
    g, task, datasets = prepare_dataset(dataset_name, task_name)
    # print input graph, task, and dataset information
    print(g)
    print(task)
    print(datasets)
    print(f"Directed: {check_direct(g)}")
    print(f"Edge Density: {edge_density(g):.6f}")
    print(f"Average Degree: {avg_degree(g):.6f}")
    # print(f"Diameter: {diameter(g)}")
    print(f"Pseudo Diameter: {pseudo_diameter(g)}")
    # print(f"Average Shortest Path Length: {avg_shortest_path(g):.6f}")
    print(f"Relative Size of Largest Connected Component: "
          f"{relative_largest_cc(g):.6f}")
    print(f"Relative Size of Largest Strongly Connected "
          f"Component: {relative_largest_cc(g):.6f}")
    print(f"Average Clustering Coefficient: "
          f"{avg_cluster_coefficient(g):.6f}")
    print(f"Edge Reciprocity: {edge_reciprocity(g):.6f}")
    print(f"Gini Coefficient of Degree: {gini_degree(g):.6f}")
    print(f"Gini Coefficient of Coreness: {gini_coreness(g):.6f}")
    print(f"Degeneracy: {degeneracy(g)}")
    print(f"Degree Assortativity: {degree_assortativity(g):.6f}")
    print(f"Edge Homogeneity: {edge_homogeneity(g):.6f}")
    print(f"Power Law Exponent: {power_law_expo(g):.6f}")
    print(f"Pareto Exponent: {pareto_expo(g):.6f}")


if __name__ == "__main__":
    main()
