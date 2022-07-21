"""Tagging graph datasets."""
import dgl
import argparse
import networkx as nx
import numpy as np
import torch

import glb
import powerlaw


def edge_density(g):
    """Compute the edge density."""
    # depends on direct/indirect
    nx_g = dgl.to_networkx(g)
    edge_den = nx.density(nx_g)
    if check_direct(g):
        return edge_den
    else:
        return 2 * edge_den


def avg_degree(g):
    """Compute the average degree."""
    # depends on direct/indirect
    nx_g = dgl.to_networkx(g)
    edge_num = nx.number_of_edges(nx_g)
    node_num = nx.number_of_nodes(nx_g)
    if check_direct(g):
        return edge_num / node_num
    else:
        return 2 * edge_num / node_num


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


def core_number_related(g):
    """Compute 2 tags related to coreness."""
    nx_g = dgl.to_networkx(g)
    # convert the MultiDiGraph to Digraph
    nx_g = nx.DiGraph(nx_g)
    # remove potential self-loops
    nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
    core_list = list(nx.core_number(nx_g).values())
    return gini_coreness(core_list), degeneracy(core_list)


def gini_coreness(nx_core_list):
    """Compute the gini index of the coreness sequence."""
    return gini_array(nx_core_list)


def degeneracy(nx_core_list):
    """Compute the Degeneracy."""
    return max(nx_core_list)


def degree_assortativity(g):
    """Compute the degree assortativity coefficient."""
    nx_g = dgl.to_networkx(g)
    return nx.degree_assortativity_coefficient(nx_g)


def check_direct(g):
    """Check the graph is directed or not."""
    # to examine whether all the edges are bi-directed edges
    nx_g = dgl.to_networkx(g)
    # remove self-loop before edge computation
    nx_g.remove_edges_from(list(nx.selfloop_edges(nx_g)))
    n_all_edge = nx_g.number_of_edges()
    n_un_edge = nx_g.to_undirected().number_of_edges()
    return n_all_edge != 2 * n_un_edge


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


def transitivity(g):
    """Compute the transitivity of the graph."""
    # only work for in-directed graphs
    # will disregard the edge direction
    nx_g = dgl.to_networkx(g)
    nx_g = nx.Graph(nx_g)
    return nx.transitivity(nx_g)


def prepare_dataset(dataset, task):
    """Prepare datasets."""
    glb_graph = glb.dataloading.get_glb_graph(dataset)
    glb_task = glb.dataloading.get_glb_task(dataset, task)
    glb_graph_task = glb.dataloading.combine_graph_and_task(
        glb_graph, glb_task)
    return glb_graph, glb_task, glb_graph_task


def dict_to_tensor(feature_dict, tensor_size):
    """Transform dict of tensor to 2-D tensor."""
    out_tensor = torch.zeros(size=tensor_size)
    for v in feature_dict:
        out_tensor[v] = feature_dict[v]
    return out_tensor


def matrix_row_norm(feature_matrix):
    """Normalize the node feature tensor."""
    return feature_matrix / torch.linalg.norm(feature_matrix, axis=1)[:, None]


def get_feature_label(g):
    """Compute the feature homogeneity."""
    # convert sparse tensor to dense tensor
    g.ndata["NodeFeature"] = g.ndata["NodeFeature"].to_dense()
    # get attribute size
    feature_size = g.ndata["NodeFeature"].size()
    label_size = g.ndata["NodeLabel"].size()
    nx_g = dgl.to_networkx(g, node_attrs=["NodeFeature", "NodeLabel"])
    # get attribute dictionary
    feature_dict = nx.get_node_attributes(nx_g, "NodeFeature")
    label_dict = nx.get_node_attributes(nx_g, "NodeLabel")
    # get feature matrix and label matrix
    feature_matrix = dict_to_tensor(feature_dict, feature_size)
    normed_feature_matrix = matrix_row_norm(feature_matrix)
    label_matrix = dict_to_tensor(label_dict, label_size)
    return normed_feature_matrix, label_matrix


def sum_angular_distance_matrix_nan(x, y):
    """Compute summation of angular distance."""
    inner_prod = x.matmul(y.T)
    inner_prod = torch.clip(inner_prod, -1.0, 1.0)
    angular_dist = 1.0 - torch.arccos(inner_prod) / torch.pi
    angular_dist[torch.where(torch.isnan(angular_dist))] = 1.0
    return torch.sum(angular_dist)


def feature_homogeneity(g):
    """Compute feature homogeneity."""
    normed_feature_matrix, label_matrix = get_feature_label(g)
    # get a sorted list of all labels
    # convert labels to integer
    label_list = [int(x) for x in list(set(label_matrix.tolist()))]
    all_labels = sorted(label_list)
    label_num = len(all_labels)
    sum_matrix = torch.zeros((label_num, label_num))
    count_matrix = torch.zeros((label_num, label_num))

    # find relationship between each pair of labels
    for label_idx, i in enumerate(all_labels):
        idx_i = torch.where(label_matrix == i)[0]
        vec_i = normed_feature_matrix[idx_i, :]
        for j in all_labels[label_idx:]:
            idx_j = torch.where(label_matrix == j)[0]
            vec_j = normed_feature_matrix[idx_j, :]
            the_sum = sum_angular_distance_matrix_nan(vec_i, vec_j)
            # the total number of pairs
            the_count = len(idx_j) * len(idx_i)
            if i == j:
                the_sum -= float(len(idx_j))
                the_sum /= 2.0
                the_count -= float(len(idx_j))
                the_count /= 2
            sum_matrix[i, j] = the_sum
            count_matrix[i, j] = the_count
    out_avg = torch.sum(sum_matrix[torch.triu_indices(
        sum_matrix.shape[0], sum_matrix.shape[0])]) / (
                  torch.sum(count_matrix[torch.triu_indices(
                      count_matrix.shape[0], sum_matrix.shape[0])]))
    in_avg = torch.sum(torch.diag(
        sum_matrix)) / torch.sum(torch.diag(count_matrix))
    return in_avg, out_avg


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
    # Related to coreness
    gini_core, degen_core = core_number_related(g)
    print(f"Gini Coefficient of Coreness: {gini_core:.6f}")
    print(f"Degeneracy: {degen_core}")
    print(f"Degree Assortativity: {degree_assortativity(g):.6f}")
    print(f"Edge Homogeneity: {edge_homogeneity(g):.6f}")
    print(f"Power Law Exponent: {power_law_expo(g):.6f}")
    print(f"Pareto Exponent: {pareto_expo(g):.6f}")
    print(f"Transitivity: {transitivity(g):.6f}")
    in_avg, out_avg = feature_homogeneity(g)
    print(f"Average In-Feature Angular Distance: {in_avg:.6f}")
    print(f"Average Out-Feature Angular Distance: {out_avg:.6f}")
    print(f"Feature Angular SNR: {in_avg / out_avg:.6f}")


if __name__ == "__main__":
    main()
