"""Utils functions for gli.io module."""
import json
import os
import numpy as np
from scipy.sparse import isspmatrix

from gli.utils import save_data


def detect_array_type(array):
    """
    Detect the type of the data in the array.

    :param array: The input array.
    :type array: scipy.sparse array or numpy array

    :raises ValueError: If the input array is empty.
    :raises TypeError: If the input array is not a scipy sparse array or numpy
        array.
    :raises TypeError: If the input array contains unsupported data types.

    :return: Data type of the elements in the array.
    :rtype: str
    """
    if isspmatrix(array) or isinstance(array, np.ndarray):
        if array.size == 0:
            raise ValueError("The input array is empty.")

        # Check for the first non-null element's type
        if isspmatrix(array):
            data = array.data
        else:
            # In case array is a multi-dimensional numpy array, flatten it
            # Otherwise we will not be able to iterate data in the for loop
            data = array.flatten().data
        for element in data:
            if element is not None:
                element_type = type(element)
                break

        if issubclass(element_type, int):
            return "int"
        elif issubclass(element_type, float) or element_type is np.float32:
            # np.float64 is a subclass of float but np.float32 is not
            return "float"
        elif issubclass(element_type, str):
            return "str"
        else:
            raise TypeError("The input array contains unsupported data types.")
    else:
        raise TypeError("The input array must be a scipy sparse array or numpy"
                        " array.")


def check_data_splits(train_set, val_set, test_set, train_ratio, val_ratio,
                      test_ratio, num_samples):
    """Check the input arguments of data splits are valid."""
    if train_set is None:
        assert num_samples is not None, \
            "If `train_set` is not provided, `num_samples` must be provided."
        assert 0 < train_ratio + val_ratio + test_ratio <= 1, \
            "The sum of `train_ratio`, `val_ratio`, and `test_ratio` must " \
            "be in (0, 1]."
    else:
        assert val_set is not None and test_set is not None, \
            "If `train_set` is provided, `val_set` and `test_set` must be " \
            "provided."
        if isinstance(train_set, np.ndarray):
            train_set = train_set.tolist()
        if isinstance(val_set, np.ndarray):
            val_set = val_set.tolist()
        if isinstance(test_set, np.ndarray):
            test_set = test_set.tolist()
        assert isinstance(train_set, list) and \
            isinstance(val_set, list) and isinstance(test_set, list), \
            "`train_set`, `val_set`, and `test_set` must be lists or numpy" \
            " arrays."
        if isinstance(train_set[0], list):  # Multiple splits.
            assert len(train_set) == len(val_set) == len(test_set), \
                "If `train_set`, `val_set`, and `test_set` are lists of " \
                "lists, they must have the same length (number of splits)."
            # Check the split ratio's of different splits are the same.
            split_num = (len(train_set[0]), len(val_set[0]), len(test_set[0]))
            for i in range(len(train_set)):
                assert len(train_set[i]) == split_num[0] and \
                    len(val_set[i]) == split_num[1] and \
                    len(test_set[i]) == split_num[2], \
                    "The split ratio's of different splits are not the same."

    # Return `train_set`, `val_set`, and `test_set` as lists of lists.
    return train_set, val_set, test_set


def check_feature(feature):
    """Check the input argument `feature` is valid."""
    assert isinstance(feature, list), \
        "`feature` must be a list of strings."
    for feat in feature:
        assert isinstance(feat, str), \
            "Each element in `feature` must be a string."
        assert feat.startswith("Node/") or feat.startswith("Edge/") or \
            feat.startswith("Graph/"), \
            "Each element in `feature` must be a node/edge/graph attribute."


def save_task_reg_or_cls(task_type,
                         name,
                         description,
                         feature,
                         target,
                         num_classes=None,
                         train_set=None,
                         val_set=None,
                         test_set=None,
                         train_ratio=0.8,
                         val_ratio=0.1,
                         test_ratio=0.1,
                         num_samples=None,
                         task_id=1,
                         save_dir="."):
    """Save the information of a regression or classification task into task json and data files.

    :param task_type: The type of the task. It should be either
        "NodeRegression" or "NodeClassification".
    :type task_type: str
    :param name: The name of the dataset.
    :type name: str
    :param description: The description of the task.
    :type description: str
    :param feature: The list of feature names to be used in the task. The
        features could be node attributes, edge attributes, or graph
        attributes. For homogeneous graphs, the feature names should be in the
        format of "Node/{node_attr_name}", "Edge/{edge_attr_name}", or
        "Graph/{graph_attr_name}". For heterogeneous graphs, the feature names
        should be in the format of "Node/{node_type}/{node_attr_name}",
        "Edge/{edge_type}/{edge_attr_name}", or
        "Graph/{graph_type}/{graph_attr_name}". The node/edge/graph_attr_name,
        and the node/edge/graph_type should be the ones declared in the
        metadata.json file.
    :type feature: list of str
    :param target: The attribute name as prediction target in the task.
    :type target: str
    :param num_classes: The number of classes in the task.
    :type num_classes: int
    :param train_set: A list of training sample IDs or a list of list training
        sample IDs. In the latter case, each inner list is the sample IDs of
        one data split. If not None, `train_ratio`, `val_ratio`, and
        `test_ratio` will be ignored while `val_set` and `test_set` must
        present. If None, the task json file will store `train_ratio`,
        `val_ratio`, and `test_ratio` and random splits will be generated at
        run time. Default: None.
    :type train_set: list/array of int or list of lists/2-d array of int
    :param val_set: A list of validation sample IDs or a list of list
        validation sample IDs. See `train_set` for more details. Default: None.
    :type val_set: list/array of int or list of lists/2-d array of int
    :param test_set: A list of test sample IDs or a list of list test sample
        IDs. See `train_set` for more details. Default: None.
    :type test_set: list/array of int or list of lists/2-d array of int
    :param train_ratio: The ratio of training samples. See `train_set` for more
        details. Default: 0.8.
    :type train_ratio: float
    :param val_ratio: The ratio of validation samples. See `train_set` for more
        details. Default: 0.1.
    :type val_ratio: float
    :param test_ratio: The ratio of test samples. See `train_set` for more
        details. Default: 0.1.
    :type test_ratio: float
    :param num_samples: The total number of samples in the dataset. This needs
        to be provided if `train_set`, `val_set`, and `test_set` are not
        provided. Default: None.
    :type num_samples: int
    :param task_id: The task ID. This is needed when there are multiple tasks
        of the same task type are defined on the dataset. Default: 1.
    :type task_id: int
    :param save_dir: The directory to save the task json and data files.
        Default: ".".
    :type save_dir: str

    :raises ValueError: If `task_type` is not "NodeRegression" or
        "NodeClassification".
    :raises ValueError: If `description` is not a string.
    :raises ValueError: If `feature` is not a list of strings.
    :raises ValueError: If elements in `feature` do not correspond to
        node/edge/graph attributes.
    :raises ValueError: If `target` is not a string.
    :raises ValueError: If `target` does not correspond to a node/graph
        attribute.
    :raises ValueError: If `num_classes` is not None for regression tasks.
    :raises ValueError: If `train_set`, `val_set`, and `test_set` are not
        provided and `num_samples` is not provided.
    :raises ValueError: If `train_set`, `val_set`, and `test_set` are not
        provided and `train_ratio`, `val_ratio`, and `test_ratio` do not sum
        up to 1.
    :raises ValueError: If `train_set`, `val_set`, and `test_set` are not
        provided at the same time.
    :raises ValueError: If `train_set`, `val_set`, and `test_set` are provided
        but they are not lists or numpy arrays.
    :raises ValueError: If `train_set`, `val_set`, and `test_set` contain
        multiple splits but the split ratio of different splits is different.

    :return: The dictionary of the content in the task json file.
    :rtype: dict
    """  # noqa: E501,E262  #pylint: disable=line-too-long
    # Check the input arguments.
    assert isinstance(description, str), \
        "`description` must be a string."
    check_feature(feature)
    assert isinstance(target, str), \
        "`target` must be a string."
    train_set, val_set, test_set = check_data_splits(train_set, val_set,
                                                     test_set, train_ratio,
                                                     val_ratio, test_ratio,
                                                     num_samples)

    # Task-dependent checks.
    if task_type in ("NodeClassification", "NodeRegression"):
        assert target.startswith("Node/"), \
            "`target` must be a node attribute."
    if task_type in ("GraphClassification", "GraphRegression"):
        assert target.startswith("Graph/"), \
            "`target` must be a graph attribute."
    if task_type == "NodeClassification":
        task_str = "node_classification"
    elif task_type == "NodeRegression":
        task_str = "node_regression"
        assert num_classes is None, \
            "`num_classes` must be None for regression tasks."
    elif task_type == "GraphClassification":
        task_str = "graph_classification"
    elif task_type == "GraphRegression":
        task_str = "graph_regression"
        assert num_classes is None, \
            "`num_classes` must be None for regression tasks."
    else:
        raise NotImplementedError(f"Task type {task_type} is not supported.")
    # Create the dictionary for task json file.
    task_dict = {
        "description": description,
        "type": task_type,
        "feature": feature,
        "target": target
    }
    if num_classes is not None:
        task_dict["num_classes"] = num_classes
    if train_set is not None:
        # Save the task data files, i.e., the data splits in this task.
        data_dict = {
            "train_set": np.array(train_set, dtype=np.int64),
            "val_set": np.array(val_set, dtype=np.int64),
            "test_set": np.array(test_set, dtype=np.int64)
        }
        key_to_loc = save_data(f"{name}__task_{task_str}_{task_id}",
                               save_dir=save_dir,
                               **data_dict)
        # Update the task dictionary with the data file names and keys.
        task_dict.update(key_to_loc)
        if isinstance(train_set[0], list):
            task_dict["num_splits"] = len(train_set)
    else:
        task_dict["train_ratio"] = train_ratio
        task_dict["val_ratio"] = val_ratio
        task_dict["test_ratio"] = test_ratio
        task_dict["num_samples"] = num_samples

    # Save the task json file.
    with open(os.path.join(save_dir, f"task_{task_str}_{task_id}.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(task_dict, f, indent=4)

    return task_dict
