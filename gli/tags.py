"""Tagging graph datasets."""
import dgl
import argparse
import networkx as nx
import numpy as np
import torch
from scipy.sparse.linalg import norm
from scipy import sparse

import gli
import powerlaw


def check_direct(nx_g):
    """Check the graph is directed or not."""
    # to examine whether all the edges are bi-directed edges
    # nx_g = dgl.to_networkx(g)
    # remove self-loop before edge computation
    # nx_g.remove_edges_from(list(nx.selfloop_edges(nx_g)))
    nx_g_ = nx_g.copy()
    n_all_edge = nx_g_.number_of_edges()
    n_un_edge = nx_g_.to_undirected().number_of_edges()
    return n_all_edge != 2 * n_un_edge


def directed(nx_g):
    """Check the graph is directed or not and output."""
    if check_direct(nx_g):
        return "Yes"
    else:
        return "No"


def edge_density(nx_g):
    """Compute the edge density."""
    # depends on direct/indirect
    edge_den = nx.density(nx_g)
    if check_direct(nx_g):
        return edge_den
    else:
        return 2 * edge_den


def avg_degree(nx_g):
    """Compute the average degree."""
    # depends on direct/indirect
    # nx_g = dgl.to_networkx(g)
    edge_num = nx.number_of_edges(nx_g)
    node_num = nx.number_of_nodes(nx_g)
    if check_direct(nx_g):
        return edge_num / node_num
    else:
        return 2 * edge_num / node_num


def degree_assortativity(nx_g):
    """Compute the degree assortativity coefficient."""
    if not check_direct(nx_g):
        nx_g_ = nx.Graph(nx_g)
        out = nx.degree_pearson_correlation_coefficient(nx_g_)
        return np.round(out, 6)
    else:
        dic = {"in", "out"}
        out = []
        for i in dic:
            for j in dic:
                out.append(np.round(nx.degree_pearson_correlation_coefficient(
                    nx_g, x=i, y=j), 6))
        return out[0], out[1], out[2], out[3]


def edge_reciprocity(nx_g):
    """Compute the edge reciprocity."""
    # nx_g = dgl.to_networkx(g)
    return nx.overall_reciprocity(nx_g)


def pseudo_diameter(nx_g):
    """Compute a lower bound on the diameter."""
    # nx_g = dgl.to_networkx(g)
    if nx.is_strongly_connected(nx_g):
        return nx.algorithms.approximation.diameter(nx_g)
    else:
        # if the graph is not strongly connected,
        # we report the diameter in the LCC
        cc = sorted(nx.strongly_connected_components(nx_g), key=len,
                    reverse=True)
        lcc = nx_g.subgraph(cc[0])
        return nx.algorithms.approximation.diameter(lcc)


def relative_largest_cc(nx_g):
    """Compute relative size of the largest connected component."""
    # will disregard the edge direction
    # nx_g = dgl.to_networkx(g)
    nx_g_ = nx.Graph(nx_g)
    lcc = sorted(nx.connected_components(nx_g_), key=len, reverse=True)
    lcc_size = nx.number_of_nodes(nx_g_.subgraph(lcc[0]))
    return lcc_size / nx.number_of_nodes(nx_g_)


def relative_largest_scc(nx_g):
    """Compute relative size of the largest strongly connected component."""
    # consider directed network only
    # nx_g = dgl.to_networkx(g)
    lcc = sorted(nx.strongly_connected_components(nx_g), key=len, reverse=True)
    lcc_size = nx.number_of_nodes(nx_g.subgraph(lcc[0]))
    return lcc_size / nx.number_of_nodes(nx_g)


def avg_cluster_coefficient(nx_g):
    """Compute the average clustering coefficient."""
    # nx_g = dgl.to_networkx(g)
    # transform from MultiDigraph to Digraph
    nx_g_ = nx.DiGraph(nx_g)
    av_local_clustering_coeff = nx.average_clustering(nx_g_)
    return av_local_clustering_coeff


def transitivity(nx_g):
    """Compute the transitivity of the graph."""
    # only work for in-directed graphs
    # will disregard the edge direction
    # nx_g = dgl.to_networkx(g)
    nx_g_ = nx.Graph(nx_g)
    return nx.transitivity(nx_g_)


def degeneracy(nx_core_list):
    """Compute the Degeneracy."""
    return max(nx_core_list)


def power_law_expo(nx_g):
    """Fit the Power-Law Distribution on degree sequence."""
    # nx_g = dgl.to_networkx(g)
    degree_sequence = [d for n, d in nx_g.degree()]
    degree_sequence = np.sort(degree_sequence)
    fit = powerlaw.Fit(degree_sequence, verbose=False)
    return fit.power_law.alpha


def pareto_expo(nx_g):
    """Get the Pareto Exponent."""
    # nx_g = dgl.to_networkx(g)
    # remove nodes that have 0 degree
    remove_nodes = [node for node, degree in
                    dict(nx_g.degree()).items() if degree == 0]
    nx_g_ = nx_g.copy()
    nx_g_.remove_nodes_from(remove_nodes)
    degree_sequence = [d for n, d in nx_g.degree()]
    degree_sequence = np.sort(degree_sequence)
    dmin = np.min(degree_sequence)
    alpha = len(degree_sequence) / np.sum(np.log(degree_sequence / dmin))
    return alpha


def gini_array(array):
    """Compute the Gini Index of a given array."""
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    return np.sum((2 * index - array.shape[0] - 1)
                  * array) / (array.shape[0] * np.sum(array))


def core_number_related(nx_g):
    """Compute 2 tags related to coreness."""
    # nx_g = dgl.to_networkx(g)
    # convert the MultiDiGraph to Digraph
    nx_g_ = nx.DiGraph(nx_g)
    # remove potential self-loops
    core_list = list(nx.core_number(nx_g_).values())
    return core_list


def gini_degree(nx_g):
    """Compute the gini index of the degree sequence."""
    # nx_g = dgl.to_networkx(g)
    degree_sequence = [d for n, d in nx_g.degree()]
    return gini_array(degree_sequence)


def gini_coreness(nx_core_list):
    """Compute the gini index of the coreness sequence."""
    return gini_array(nx_core_list)


def edge_homogeneity(nx_g_attr):
    """Compute the edge homogeneity."""
    # proportion of the edges that connect nodes with the same class label
    count = 0
    # nx_g = dgl.to_networkx(g, node_attrs=["NodeLabel"])
    edge_num = nx_g_attr.number_of_edges()
    for e in nx_g_attr.edges(data=True):
        if nx_g_attr.nodes[e[0]]["NodeLabel"] == \
                nx_g_attr.nodes[e[1]]["NodeLabel"]:
            count += 1

    return count / edge_num


def matrix_row_norm(feature_matrix):
    """Normalize the node feature tensor."""
    row_sums = norm(feature_matrix, axis=1)
    row_ind, _ = feature_matrix.nonzero()
    feature_matrix.data /= row_sums[row_ind]
    return feature_matrix


def get_feature_label(g):
    """Compute the feature homogeneity."""
    sparse_feat_ts = g.ndata["NodeFeature"].to_sparse_coo().coalesce()
    sp_ind = sparse_feat_ts.indices().numpy()
    sp_val = sparse_feat_ts.values().numpy() * 1.0
    sparse_feat = sparse.csr_matrix((sp_val, sp_ind))
    normed_feature_matrix = matrix_row_norm(sparse_feat)
    label = g.ndata["NodeLabel"].numpy().squeeze()
    return normed_feature_matrix, label


def sum_angular_distance_matrix_nan(x, y, batch_size):
    """Compute summation of angular distance."""
    x_dim = x.shape[0]
    y_dim = y.shape[0]
    fsum = 0.0
    x_start = 0
    while x_start < x_dim:
        # print("x pos: ", x_start)
        x_end = min(x_start + batch_size, x_dim)
        x_batch = x[x_start: x_end, :]

        y_start = 0
        while y_start < y_dim:
            # print("y pos: ", y_start)
            y_end = min(y_start + batch_size, y_dim)
            y_batch = y[y_start: y_end, :]
            inner_prod = np.dot(x_batch, y_batch.transpose())
            inner_prod.data = np.clip(inner_prod.data, -1.0, 1.0)
            inner_prod.data = 1.0 - np.arccos(inner_prod.data) / np.pi
            # remaining numbers are zeros with arccos value = 0.5
            fsum += (inner_prod.sum() + (np.prod(inner_prod.get_shape())
                                         - inner_prod.nnz) * 0.5)
            y_start += batch_size
        x_start += batch_size
    return fsum


def feature_homogeneity(g):
    """Compute feature homogeneity."""
    normed_feature_matrix, label_matrix = get_feature_label(g)
    # get a sorted list of all labels
    # convert labels to integer
    label_list = [int(x) for x in list(set(label_matrix.tolist()))]
    all_labels = sorted(label_list)
    label_num = len(all_labels)
    sum_matrix = np.zeros((label_num, label_num))
    count_matrix = np.zeros((label_num, label_num))

    for label_idx, i in enumerate(all_labels):
        idx_i = np.where(label_matrix == i)[0]
        vec_i = normed_feature_matrix[idx_i, :]
        for j in all_labels[label_idx:]:
            idx_j = np.where(label_matrix == j)[0]
            vec_j = normed_feature_matrix[idx_j, :]
            batch_size = 10000
            the_sum = sum_angular_distance_matrix_nan(vec_i, vec_j,
                                                      batch_size=batch_size)
            # the total number of pairs
            the_count = len(idx_j) * len(idx_i)
            if i == j:
                the_sum -= float(len(idx_j))
                the_sum /= 2.0
                the_count -= float(len(idx_j))
                the_count /= 2
            sum_matrix[i, j] = the_sum
            count_matrix[i, j] = the_count
    out_avg = np.sum(sum_matrix[torch.triu_indices(
        sum_matrix.shape[0], sum_matrix.shape[0])]) / (
                  np.sum(count_matrix[torch.triu_indices(
                      count_matrix.shape[0], sum_matrix.shape[0])]))
    in_avg = np.sum(np.diag(
        sum_matrix)) / np.sum(np.diag(count_matrix))
    return in_avg, out_avg


def avg_in_feature_dist(g):
    """Compute the average in-feature angular distance."""
    return feature_homogeneity(g)[0]


def avg_out_feature_dist(g):
    """Compute the average out-feature angular distance."""
    return feature_homogeneity(g)[1]


def feature_snr(g):
    """Compute the feature angular SNR."""
    return avg_in_feature_dist(g) / avg_out_feature_dist(g)


def homophily_hat(nx_g_attr):
    """Compute the modified homophily measure."""
    # "Large Scale Learning on Non-Homophilous Graphs:
    # New Benchmarks and Strong Simple Methods"
    # For directed graphs, only outgoing neighbors /
    # adjacencies are included.

    label_dict = nx.get_node_attributes(nx_g_attr, "NodeLabel")
    all_label = list(set(label_dict.values()))
    label_edge_dict = {}
    label_same_edge_dict = {}
    # Initialize
    for label in all_label:
        label_edge_dict[label] = 0
        label_same_edge_dict[label] = 0

    # compute C_k
    label_num_dict = {}
    node_num = nx_g_attr.number_of_nodes()
    for n in nx_g_attr.nodes(data=True):
        label_name = n[1]["NodeLabel"]
        if label_name not in label_num_dict:
            label_num_dict[label_name] = 1
        else:
            label_num_dict[label_name] += 1

    for n in nx_g_attr.nodes(data=True):
        # find n's neighbor
        core_label = n[1]["NodeLabel"]
        # print("core label: ", core_label)
        for nb in nx_g_attr.neighbors(n[0]):
            label_edge_dict[core_label] += 1
            nb_label = nx_g_attr.nodes[nb]["NodeLabel"]
            if core_label == nb_label:
                label_same_edge_dict[core_label] += 1
    # output value
    counter = 0
    for label in all_label:
        if label_edge_dict[label] > 0:
            h_k = label_same_edge_dict[label] * 1.0 / label_edge_dict[label]
            if h_k > label_num_dict[label] / node_num:
                counter += (h_k - label_num_dict[label] / node_num)

    return counter / (len(all_label) - 1)


def attribute_assortativity(nx_g_attr):
    """Compute the attribute(label) assortativity coefficient."""
    # nx_g = dgl.to_networkx(g, node_attrs=["NodeLabel"])
    return nx.attribute_assortativity_coefficient(nx_g_attr, "NodeLabel")


def prepare_dataset(dataset, task):
    """Prepare datasets."""
    gli_graph = gli.dataloading.get_gli_graph(dataset)
    gli_task = gli.dataloading.get_gli_task(dataset, task)
    gli_graph_task = gli.dataloading.combine_graph_and_task(
        gli_graph, gli_task)
    return gli_graph, gli_task, gli_graph_task


def make_metric_dict():
    """Construct groups of graph metrics."""
    output_dict = {"Basic": [directed, edge_density, avg_degree,
                             edge_reciprocity],
                   "Distance": [pseudo_diameter],
                   "Connectivity": [relative_largest_cc,
                                    relative_largest_scc],
                   "Clustering": [avg_cluster_coefficient,
                                  transitivity, degeneracy],
                   "Distribution": [power_law_expo, pareto_expo,
                                    gini_degree, gini_coreness],
                   "Attribute": [edge_homogeneity, avg_in_feature_dist,
                                 avg_out_feature_dist, feature_snr,
                                 homophily_hat, attribute_assortativity]}
    return output_dict


def make_metric_quote():
    """Construct quotes of each metric group."""
    output_dict = {"Basic": ">These are metrics associated "
                            "with basic graph characteristics,"
                            "such as the number of nodes / edges, "
                            "node degrees, and "
                            "whether the graph is directed / undirected.\n",
                   "Distance": ">These are metrics associated "
                               "with (geodestic) distances on the graph.\n",
                   "Connectivity": ">These are metrics associated with the "
                                   "connectedness and connected "
                                   "component of the graph.\n",
                   "Clustering": ">These are metrics associated with sparsity "
                                 "and closeness of the graph.\n",
                   "Distribution": ">These are metrics assoicated with the "
                                   "characteristics of the distribution of "
                                   "the sequence of node-level properties.\n",
                   "Attribute": ">These are metrics associated with both "
                                "graph structure and node features/labels.\n"}

    return output_dict


def make_metric_names():
    """Construct names of each graph metric."""
    output_dict = {"Basic": ["Directed", "Edge Density", "Average Degree",
                             "Edge Reciprocity"],
                   "Distance": ["Pseudo Diameter"],
                   "Connectivity": ["Relative Size of Largest "
                                    "Connected Component",
                                    "Relative Size of Largest "
                                    "Strongly Connected Component"],
                   "Clustering": ["Average Clustering Coefficient",
                                  "Transitivity", "Degeneracy"],
                   "Distribution": ["Power Law Exponent", "Pareto Exponent",
                                    "Gini Coefficient of Degree",
                                    "Gini Coefficient of Coreness"],
                   "Attribute": ["Edge Homogeneity",
                                 "Average In-Feature Angular Distance",
                                 "Average Out-Feature Angular Distance",
                                 "Feature Angular SNR", "Homophily Hat",
                                 "Attribute Assortativity"]}
    return output_dict


def output_markdown_file(file_name, g, metric_dict, metric_quote, metric_name):
    """Output the tags of a dataset into txt with Markdown format."""
    group_dict = ["Basic", "Distance", "Connectivity", "Clustering",
                  "Distribution", "Attribute"]

    # skipped metric: diameter, efficiency,
    # avg_shortest_path, degree_assortativity
    # since these will run forever on large datasets
    nx_g = dgl.to_networkx(dgl.to_homogeneous(g))
    nx_g_rem = nx_g.copy()
    nx_g_rem.remove_edges_from(list(nx.selfloop_edges(nx_g_rem)))
    core_list = core_number_related(nx_g_rem)

    with open(file_name, "w", encoding="utf-8") as f:
        f.write("## **Graph Metrics**\n")
        for group_name in group_dict:
            f.write("### " + str(group_name) + " Metrics\n")
            f.write(metric_quote[group_name])
            f.write("\n")
            f.write("| Metric | Quantity |\n")
            f.write("| ------ | ------ |\n")
            if group_name != "Attribute":
                for i in range(len(metric_dict[group_name])):
                    print(metric_dict[group_name][i].__name__)
                    if metric_dict[group_name][i].__name__ in ("gini_coreness",
                                                               "degeneracy"):
                        var = metric_dict[group_name][i](core_list)
                    else:
                        var = metric_dict[group_name][i](nx_g)
                    if not isinstance(var, str):
                        if torch.is_tensor(var):
                            var = var.item()
                        var = round(var, 6)
                    f.write("| " + str(metric_name[group_name][i]) +
                            " | " + str(var) + " |")
                    f.write("\n")
            else:
                nx_g_attr = dgl.to_networkx(g, node_attrs=["NodeLabel"])
                # convert from tensor to numerical value
                for n in nx_g_attr:
                    nx_g_attr.nodes[n]["NodeLabel"] = \
                        nx_g_attr.nodes[n]["NodeLabel"].item()
                nx_g_attr_rem = nx_g_attr.copy()
                nx_g_attr_rem.remove_edges_from(
                    list(nx.selfloop_edges(nx_g_attr_rem)))
                in_avg, out_avg = feature_homogeneity(g)

                for i in range(len(metric_dict[group_name])):
                    print(metric_dict[group_name][i].__name__)
                    if metric_dict[group_name][i].__name__ \
                            == "avg_in_feature_dist":
                        var = in_avg
                    elif metric_dict[group_name][i].__name__ \
                            == "avg_out_feature_dist":
                        var = out_avg
                    elif metric_dict[group_name][i].__name__ \
                            == "feature_snr":
                        var = in_avg / out_avg
                    elif metric_dict[group_name][i].__name__ \
                            == "homophily_hat":
                        var = metric_dict[group_name][i](nx_g_attr_rem)
                    else:
                        var = metric_dict[group_name][i](nx_g_attr)
                    if not isinstance(var, str):
                        if torch.is_tensor(var):
                            var = var.item()
                        var = round(var, 6)
                    f.write("| " + str(metric_name[group_name][i]) +
                            " | " + str(var) + " |")
                    f.write("\n")

    f.close()


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

    metric_dict = make_metric_dict()
    metric_quote = make_metric_quote()
    metric_name = make_metric_names()
    output_markdown_file("markdown_file_" + str(dataset_name) + ".txt",
                         g, metric_dict, metric_quote, metric_name)


if __name__ == "__main__":
    main()
