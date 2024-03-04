"""The ``gli.raw_text_utils`` module provides functions to process raw text."""
import json
import numpy as np
import os
import pandas as pd
import random
import sys
import torch
from datasets import load_dataset
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.preprocessing import normalize
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

sys.path.append("../")



DATASET_W_RAW_TEXT = ["cora", "pubmed", "ogbn-arxiv",
                      "arxiv-2023", "ogbn-products"]


def load_data(dataset, use_text=False, seed=0):
    """
    Load data based on the dataset name.

    Parameters:
        dataset (str): Name of the dataset to be loaded. 
            Options are "cora", "pubmed", "arxiv", "arxiv_2023", and "product".
        use_text (bool, optional): Whether to use text data. Default is False.
        seed (int, optional): Random seed for data loading. Default is 0.

    Returns:
        Tuple: Loaded data and text information.

    Raises:
        ValueError: If the dataset name is not recognized.
    """

    if dataset == "cora":
        data, text = get_raw_text_cora(use_text, seed)
    elif dataset == "pubmed":
        data, text = get_raw_text_pubmed(use_text, seed)
    elif dataset == "arxiv":
        data, text = get_raw_text_arxiv(use_text)
    elif dataset == "arxiv_2023":
        data, text = get_raw_text_arxiv_2023(use_text)
    elif dataset == "product":
        data, text = get_raw_text_products(use_text)
    else:
        raise ValueError("Dataset must be one of: cora, pubmed, arxiv")
    return data, text


################# Ogbn-arxiv #################


def get_raw_text_arxiv(use_text=False):
    """
    Reference: https://github.com/XiaoxinHe/TAPE/blob/
    main/core/data_utils/load_arxiv.py
    """

    dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits["train"]] = True
    val_mask[idx_splits["valid"]] = True
    test_mask[idx_splits["test"]] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    nodeidx2paperid_path = "dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz"
    nodeidx2paperid = pd.read_csv(nodeidx2paperid_path, compression="gzip")

    raw_text = pd.read_csv("dataset/ogbn_arxiv/titleabs.tsv", sep="\t")
    raw_text.columns = ["paper id", "title", "abs"]

    df = pd.merge(nodeidx2paperid, raw_text, on="paper id")

    text = {"title": [], "abs": [], "label": []}

    for ti, ab in zip(df["title"], df["abs"]):
        text["title"].append(ti)
        text["abs"].append(ab)

    # Load the label index to arXiv category mapping data
    label_mapping_path = "dataset/ogbn_arxiv/mapping/"\
                        "labelidx2arxivcategeory.csv.gz"
    label_mapping_data = pd.read_csv(label_mapping_path)
    label_mapping_data.columns = ["label_idx", "arxiv_category"]

    for i in range(len(data.y)):
        row = label_mapping_data.loc[
            label_mapping_data["label_idx"].isin(data.y[i].numpy())]
        # If the row doesn"t exist, return a message indicating this
        if len(row) == 0:
            raise ValueError("No matching arXiv category found for this label.")

        # Parse the arXiv category string to be in the desired format "cs.XX"
        arxiv_category = "cs." + row["arxiv_category"]\
                        .values[0].split()[-1].upper()
        text["label"].append(arxiv_category)

    return data, text


def generate_arxiv_keys_list():
    label_mapping_path = "dataset/ogbn_arxiv/mapping/"\
                        "labelidx2arxivcategeory.csv.gz"
    label_mapping_data = pd.read_csv(label_mapping_path, compression="gzip")
    label_mapping_data.columns = ["label_idx", "arxiv_category"]
    arxiv_categories = label_mapping_data["arxiv_category"].unique()
    return ["cs." + category.split()[-1].upper()
            for category in arxiv_categories]



################# Arxiv-2023 #################

def get_raw_text_arxiv_2023(use_text=True, base_path="dataset/arxiv_2023"):
    # Load processed data
    edge_index = torch.load(os.path.join(base_path,
                                         "processed", "edge_index.pt"))
    # Load raw data
    titles_df = pd.read_csv(os.path.join(base_path,
                            "raw", "titles.csv.gz"), compression="gzip")
    abstracts_df = pd.read_csv(os.path.join(base_path,
                                "raw", "abstracts.csv.gz"), compression="gzip")
    ids_df = pd.read_csv(os.path.join(base_path, "raw", "ids.csv.gz"),
                         compression="gzip")
    labels_df = pd.read_csv(os.path.join(base_path, "raw", "labels.csv.gz"),
                            compression="gzip")

    # Load split data
    train_id_df = pd.read_csv(os.path.join(base_path, "split", "train.csv.gz"),
                              compression="gzip")
    val_id_df = pd.read_csv(os.path.join(base_path, "split", "valid.csv.gz"),
                            compression="gzip")
    test_id_df = pd.read_csv(os.path.join(base_path, "split", "test.csv.gz"),
                             compression="gzip")

    num_nodes = len(ids_df)
    titles = titles_df["titles"].tolist()
    abstracts = abstracts_df["abstracts"].tolist()
    ids = ids_df["ids"].tolist()
    labels = labels_df["labels"].tolist()
    train_id = train_id_df["train_id"].tolist()
    val_id = val_id_df["val_id"].tolist()
    test_id = test_id_df["test_id"].tolist()

    features = torch.load(os.path.join(base_path, "processed", "features.pt"))

    y = torch.load(os.path.join(base_path, "processed", "labels.pt"))

    train_mask = torch.tensor([x in train_id for x in range(num_nodes)])
    val_mask = torch.tensor([x in val_id for x in range(num_nodes)])
    test_mask = torch.tensor([x in test_id for x in range(num_nodes)])

    data = Data(
        x=features,
        y=y,
        paper_id=ids,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
    )

    data.train_id = train_id
    data.val_id = val_id
    data.test_id = test_id

    if not use_text:
        return data, None

    text = {"title": titles, "abs": abstracts, "label": labels, "id": ids}

    return data, text

################# Cora #################

cora_mapping = {
    0: "Case Based",
    1: "Genetic Algorithms",
    2: "Neural Networks",
    3: "Probabilistic Methods",
    4: "Reinforcement Learning",
    5: "Rule Learning",
    6: "Theory"
}

def get_cora_casestudy(seed=0):
    """
    Reference: https://github.com/XiaoxinHe/TAPE/blob/main/
    core/data_utils/load_cora.py
    """
    data_x, data_y, data_citeid, data_edges = parse_cora()
    # data_x = sklearn.preprocessing.normalize(data_x, norm="l1")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # load data
    data_name = "cora"
    # path = osp.join(osp.dirname(osp.realpath(__file__)), "dataset")
    dataset = Planetoid("dataset", data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_x).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_y).long()
    data.num_nodes = len(data_y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.1)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_citeid

# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun
def parse_cora():
    path = "cora_raw/cora"
    idx_features_labels = np.genfromtxt(
        f"{path}.content", dtype=np.dtype(str))
    data_x = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(["Case_Based",
                                             "Genetic_Algorithms",
                                             "Neural_Networks",
                                             "Probabilistic_Methods",
                                             "Reinforcement_Learning",
                                             "Rule_Learning",
                                             "Theory"])}
    data_y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        f"{path}.cites", dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges is None).max(1)], dtype="int")
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_x, data_y, data_citeid, \
           np.unique(data_edges, axis=0).transpose()

def get_raw_text_cora(use_text=False, seed=0):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open("cora_raw/mccallum/cora/papers", encoding="UTF-8")as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split("\t")[0]
        fn = line.split("\t")[1].replace(":", "_")
        pid_filename[pid] = fn

    path = "cora_raw/mccallum/cora/extractions/"

    text = {"title": [], "abs": [], "label": []}

    # Assuming path is given
    all_files = {f.lower(): f for f in os.listdir(path)}

    for pid in data_citeid:
        expected_fn = pid_filename[pid].lower()
        # fn = pid_filename[pid]
        if expected_fn in all_files:
            real_fn = all_files[expected_fn]
            with open(path+real_fn, encoding="UTF-8") as f:
                lines = f.read().splitlines()

            if expected_fn in all_files:
                real_fn = all_files[expected_fn]

            for line in lines:
                if "Title:" in line:
                    ti = line
                if "Abstract:" in line:
                    ab = line
            text["title"].append(ti)
            text["abs"].append(ab)

    for i in range(len(data.y)):
        text["label"].append(cora_mapping[data.y[i].item()])

    return data, text


################# Ogbn-product #################



def get_raw_dataset(raw_train="dataset/ogbn_products/Amazon-3M.raw/trn.json.gz",
                    raw_test="dataset/ogbn_products/"\
                             "Amazon-3M.raw/tst.json.gz",
                    label2cat="dataset/ogbn_products/mapping/"\
                              "labelidx2productcategory.csv.gz",
                    idx2asin="dataset/ogbn_products/mapping/"\
                             "nodeidx2asin.csv.gz"):
    """
    mapping references:
    https://github.com/CurryTang/Graph-LLM/blob/master/utils.py
    """

    train_part = load_dataset("json", data_files=raw_train)
    test_part = load_dataset("json", data_files=raw_test)
    train_df = train_part["train"].to_pandas()
    test_df = test_part["train"].to_pandas()
    combine_df = pd.concat([train_df, test_df], ignore_index=True)

    label2cat_df = pd.read_csv(label2cat, compression="gzip")
    idx2asin_df = pd.read_csv(idx2asin, compression="gzip")

    idx_mapping = {row[0]: row[1] for row in idx2asin_df.values}
    label_mapping = {row["label idx"]: row["product category"]
                     for _, row in label2cat_df.iterrows()}
    content_mapping = {row[0]: (row[1], row[2]) for row in combine_df.values}

    return idx_mapping, content_mapping, label_mapping

def get_raw_text_products(use_text=False):
    dataset = PygNodePropPredDataset(name="ogbn-products")
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits["train"]] = True
    val_mask[idx_splits["valid"]] = True
    test_mask[idx_splits["test"]] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    if not use_text:
        return data, None

    idx_mapping, content_mapping, label_mapping = get_raw_dataset()

    text = {"title": [], "content": [], "label": []}

    for i in range(len(data.y)):
        uid = idx_mapping.get(i, None)
        if uid:
            title, content = content_mapping.get(uid, (None, None))
            label = label_mapping.get(data.y[i].item(), None)

            text["title"].append(title)
            text["content"].append(content)

            mapped_label = products_mapping.get(label, None)
            if mapped_label is None:
                text["label"].append("label 25")
            else:
                text["label"].append(mapped_label)

    return data, text


products_mapping = {"Home & Kitchen": "Home & Kitchen",
        "Health & Personal Care": "Health & Personal Care",
        "Beauty": "Beauty",
        "Sports & Outdoors": "Sports & Outdoors",
        "Books": "Books",
        "Patio, Lawn & Garden": "Patio, Lawn & Garden",
        "Toys & Games": "Toys & Games",
        "CDs & Vinyl": "CDs & Vinyl",
        "Cell Phones & Accessories": "Cell Phones & Accessories",
        "Grocery & Gourmet Food": "Grocery & Gourmet Food",
        "Arts, Crafts & Sewing": "Arts, Crafts & Sewing",
        "Clothing, Shoes & Jewelry": "Clothing, Shoes & Jewelry",
        "Electronics": "Electronics",
        "Movies & TV": "Movies & TV",
        "Software": "Software",
        "Video Games": "Video Games",
        "Automotive": "Automotive",
        "Pet Supplies": "Pet Supplies",
        "Office Products": "Office Products",
        "Industrial & Scientific": "Industrial & Scientific",
        "Musical Instruments": "Musical Instruments",
        "Tools & Home Improvement": "Tools & Home Improvement",
        "Magazine Subscriptions": "Magazine Subscriptions",
        "Baby Products": "Baby Products",
        "label 25": "label 25",
        "Appliances": "Appliances",
        "Kitchen & Dining": "Kitchen & Dining",
        "Collectibles & Fine Art": "Collectibles & Fine Art",
        "All Beauty": "All Beauty",
        "Luxury Beauty": "Luxury Beauty",
        "Amazon Fashion": "Amazon Fashion",
        "Computers": "Computers",
        "All Electronics": "All Electronics",
        "Purchase Circles": "Purchase Circles",
        "MP3 Players & Accessories": "MP3 Players & Accessories",
        "Gift Cards": "Gift Cards",
        "Office & School Supplies": "Office & School Supplies",
        "Home Improvement": "Home Improvement",
        "Camera & Photo": "Camera & Photo",
        "GPS & Navigation": "GPS & Navigation",
        "Digital Music": "Digital Music",
        "Car Electronics": "Car Electronics",
        "Baby": "Baby",
        "Kindle Store": "Kindle Store",
        "Buy a Kindle": "Buy a Kindle",
        "Furniture & D&#233;cor": "Furniture & Decor",
        "#508510": "#508510"}

products_keys_list = list(products_mapping.keys())



################# Pubmed #################

"""
Reference: https://github.com/XiaoxinHe/TAPE/blob/main/core/
data_utils/load_pubmed.py
"""

pubmed_mapping = {
    0: "Experimentally induced diabetes",
    1: "Type 1 diabetes",
    2: "Type 2 diabetes",
}

def get_pubmed_casestudy(corrected=False, seed=0):
    _, data_x, data_y, data_pubid, data_edges = parse_pubmed()
    data_x = normalize(data_x, norm="l1")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load data
    data_name = "PubMed"
    # path = osp.join(osp.dirname(osp.realpath(__file__)), "dataset")
    dataset = Planetoid("dataset", data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    # replace dataset matrices with the PubMed-Diabetes data,
    # for which we have the original pubmed IDs
    data.x = torch.tensor(data_x)
    data.edge_index = torch.tensor(data_edges)
    data.y = torch.tensor(data_y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    if corrected:
        is_mistake = np.loadtxt(
            "pubmed_casestudy/pubmed_mistake.txt", dtype="bool")
        data.train_id = [i for i in data.train_id if not is_mistake[i]]
        data.val_id = [i for i in data.val_id if not is_mistake[i]]
        data.test_id = [i for i in data.test_id if not is_mistake[i]]

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_pubid


def parse_pubmed():
    path = "dataset/PubMed/data/"

    n_nodes = 19717
    n_features = 500

    data_x = np.zeros((n_nodes, n_features), dtype="float32")
    data_y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}
    feature_to_index = {}

    # parse nodes
    with open(path + "Pubmed-Diabetes.NODE.paper.tab", "r", encoding="UTF-8")\
        as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split("\t")

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split("=")[-1]) - \
                1  # subtract 1 to zero-count
            data_y[i] = label

            # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
            features = items[2:-1]
            for feature in features:
                parts = feature.split("=")
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_x[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_a = np.zeros((n_nodes, n_nodes), dtype="float32")

    with open(path + "Pubmed-Diabetes.DIRECTED.cites.tab",
              "r", encoding="UTF-8") as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split("\t")

            tail = items[1].split(":")[-1]
            head = items[3].split(":")[-1]

            data_a[paper_to_index[tail], paper_to_index[head]] = 1.0
            data_a[paper_to_index[head], paper_to_index[tail]] = 1.0
            if head != tail:
                data_edges.append(
                    (paper_to_index[head], paper_to_index[tail]))
                data_edges.append(
                    (paper_to_index[tail], paper_to_index[head]))

    return data_a, data_x, data_y, data_pubid,\
           np.unique(data_edges, axis=0).transpose()


def get_raw_text_pubmed(use_text=False, seed=0):
    data, _ = get_pubmed_casestudy(seed=seed)
    if not use_text:
        return data, None
    with open("dataset/PubMed/pubmed.json", encoding="UTF-8") as f:
        pubmed = json.load(f)
        df_pubmed = pd.DataFrame.from_dict(pubmed)

    ab = df_pubmed["AB"].fillna("")
    ti = df_pubmed["TI"].fillna("")
    text = {"title": [], "abs": [], "label": []}
    for ti, ab in zip(ti, ab):
        text["title"].append(ti)
        text["abs"].append(ab)

    for i in range(len(data.y)):
        text["label"].append(pubmed_mapping[data.y[i].item()])

    return data, text
