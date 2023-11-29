"""Edge task related helper functions."""

import os
import json
import numpy as np
from .utils import check_feature
from ..utils import save_data


def save_task_link_prediction(name,
                              description,
                              feature,
                              train_set,
                              val_set,
                              test_set,
                              val_neg=None,
                              test_neg=None,
                              task_id=1,
                              save_dir="."):
    """Save the link prediction task information into task json and data files.

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
    :param train_set: A list of training edge IDs or a list of list training
        edge IDs.
    :type train_set: list/array of int or list of lists/2-d array of int
    :param val_set: A list of validation edge IDs or a list of list validation
        edge IDs.
    :type val_set: list/array of int or list of lists/2-d array of int
    :param test_set: A list of test edge IDs or a list of list test edge IDs.
    :type test_set: list/array of int or list of lists/2-d array of int
    :param val_neg: Negative samples of edges to validate. Default: None.
    :type val_neg: list/array of int or list of lists/2-d array of int
    :param test_neg: Negative samples of edges to test. Default: None.
    :type test_neg: list/array of int or list of lists/2-d array of int
    :param task_id: The task ID. This is needed when there are multiple tasks
        of the same task type are defined on the dataset. Default: 1.
    :type task_id: int
    :param save_dir: The directory to save the task json and data files.
        Default: ".".
    :type save_dir: str

    :raises ValueError: If `description` is not a string.
    :raises ValueError: If `feature` is not a list of strings.
    :raises ValueError: If elements in `feature` do not correspond to
        node/edge/graph attributes.

    :return: The dictionary of the content in the task json file.
    :rtype: dict

    Example
    -------
    .. code-block:: python

        train_set = [0, 1]
        val_set = [2, 3]
        test_set = [4, 5]

        # Save the task information.
        save_task_link_prediction(
            name="example_dataset",
            description="A link prediction task for the example dataset.",
            feature=["Node/DenseNodeFeature", "Node/SparseNodeFeature"],
            train_set=train_set,
            val_set=val_set,
            test_set=test_set)
        # This function will save the task information into a json file named
        # `task_node_classification_1.json` and one numpy data file storing the
        # data splits, `train_set`, `val_set`, and `test_set`. The json file
        # will look like the following.

    .. code-block:: json

        {
            "description": "A link prediction task for the example dataset.",
            "type": "LinkPrediction",
            "feature": [
                "Node/DenseNodeFeature",
                "Node/SparseNodeFeature"
            ],
            "train_set": {
                "file": "example_dataset__task_link_prediction_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "train_set"
            },
            "val_set": {
                "file": "example_dataset__task_link_prediction_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "val_set"
            },
            "test_set": {
                "file": "example_dataset__task_link_prediction_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "test_set"
            }
        }
    """  # noqa: E501,E262  #pylint: disable=line-too-long
    # Check the input arguments.
    assert isinstance(description, str), \
        "`description` must be a string."
    check_feature(feature)
    task_type = "LinkPrediction"
    task_str = "link_prediction"

    task_dict = {
        "description": description,
        "type": task_type,
        "feature": feature,
    }
    data_dict = {
        "train_set": np.array(train_set, dtype=np.int64),
        "val_set": np.array(val_set, dtype=np.int64),
        "test_set": np.array(test_set, dtype=np.int64),
    }
    if val_neg is not None:
        data_dict["val_neg"] = val_neg
    if test_neg is not None:
        data_dict["test_neg"] = test_neg
    key_to_loc = save_data(f"{name}__task_{task_str}_{task_id}",
                           save_dir=save_dir,
                           **data_dict)
    task_dict.update(key_to_loc)

    # Save the task json file.
    with open(os.path.join(save_dir, f"task_{task_str}_{task_id}.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(task_dict, f, indent=4)

    return task_dict


def save_task_time_dependent_link_prediction(name,
                                             description,
                                             feature,
                                             time,
                                             train_time_window,
                                             val_time_window,
                                             test_time_window,
                                             val_neg=None,
                                             test_neg=None,
                                             task_id=1,
                                             save_dir="."):
    """Save the time dependent link prediction task information into task json and data files.

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
    :param time: The time attribute that indicates edge order in the format of "Edge/{edge_attr_name}"
    :type time: str
    :param train_time_window: Training time window (left inclusive and right exclusive).
    :type train_time_window: list of two float
    :param val_time_window: Validation time window (left inclusive and right exclusive).
    :type val_time_window: list of two float
    :param test_time_window: Testing time window (left inclusive and right exclusive).
    :type test_time_window: list of two float
    :param val_neg: Negative samples of edges to validate. Default: None.
    :type val_neg: list/array of int or list of lists/2-d array of int
    :param test_neg: Negative samples of edges to test. Default: None.
    :type test_neg: list/array of int or list of lists/2-d array of int
    :param task_id: The task ID. This is needed when there are multiple tasks
        of the same task type are defined on the dataset. Default: 1.
    :type task_id: int
    :param save_dir: The directory to save the task json and data files.
        Default: ".".
    :type save_dir: str

    :raises ValueError: If `description` is not a string.
    :raises ValueError: If `time` is not a string.
    :raises ValueError: If `time` do not correspond to edge attributes.
    :raises ValueError: If `train_time_window` is not a list.
    :raises ValueError: If `val_time_window` is not a list.
    :raises ValueError: If `test_time_window` is not a list.
    :raises ValueError: If `feature` is not a list of strings.
    :raises ValueError: If elements in `feature` do not correspond to
        node/edge/graph attributes.

    :return: The dictionary of the content in the task json file.
    :rtype: dict

    Example
    -------
    .. code-block:: python

        # Save the task information.
        save_task_time_dependent_link_prediction(
            name="example_dataset",
            description="A time dependent link prediction task for the example dataset.",
            feature=["Node/DenseNodeFeature", "Node/SparseNodeFeature"],
            time="Edge/EdgeYear",train_time_window=[1, 2],val_time_window=[3, 4],test_time_window[5, 6],)
        # This function will save the task information into a json file named
        # `task_node_classification_1.json` and one numpy data file storing the
        # data splits, `train_set`, `val_set`, and `test_set`. The json file
        # will look like the following.

    .. code-block:: json

        {
            "description": "A time dependent link prediction task for the example dataset.",
            "type": "TimeDependentLinkPrediction",
            "feature": [
                "Node/DenseNodeFeature",
                "Node/SparseNodeFeature"
            ],
            "time": "Edge/EdgeYear",
            "train_time_window": [
                1,
                2
            ],
            "val_time_window": [
                3,
                4
            ],
            "test_time_window": [
                5,
                6
            ]
        }
    """  # noqa: E501,E262  #pylint: disable=line-too-long
    # Check the input arguments.
    assert isinstance(description, str), \
        "`description` must be a string."
    check_feature(feature)
    assert isinstance(time, str), \
        "`time` must be a string."
    assert time.startswith("Edge/"), \
        "`time` must be an edge attribute."
    assert isinstance(train_time_window, list), \
        "`train_time_window` must be a list."
    assert isinstance(val_time_window, list), \
        "`val_time_window` must be a list."
    assert isinstance(test_time_window, list), \
        "`test_time_window` must be a list."
    assert len(train_time_window) == 2, \
        "`train_time_window` must be a list of 2."
    assert len(val_time_window) == 2, \
        "`val_time_window` must be a list of 2."
    assert len(test_time_window) == 2, \
        "`test_time_window` must be a list of 2."
    # print("train_time_window[0]: ", train_time_window[0])
    assert isinstance(train_time_window[0], (float, int)) \
        and isinstance(train_time_window[1], (float, int)), \
        "`train_time_window` must be a list of numbers."
    assert isinstance(val_time_window[0], (float, int)) \
        and isinstance(val_time_window[1], (float, int)), \
        "`val_time_window` must be a list of numbers."
    assert isinstance(test_time_window[0], (float, int)) \
        and isinstance(test_time_window[1], (float, int)), \
        "`test_time_window` must be a list of numbers."
    assert train_time_window[0] < train_time_window[1], \
        "`train_time_window` must not overlap."
    assert val_time_window[0] < val_time_window[1], \
        "`val_time_window` must not overlap."
    assert test_time_window[0] < test_time_window[1], \
        "`test_time_window` must not overlap."
    task_type = "TimeDependentLinkPrediction"
    task_str = "time_dependent_link_prediction"
    task_dict = {
        "description": description,
        "type": task_type,
        "feature": feature,
        "time": time,
        "train_time_window": train_time_window,
        "val_time_window": val_time_window,
        "test_time_window": test_time_window
    }
    data_dict = {}
    if val_neg is not None:
        data_dict["val_neg"] = val_neg
    if test_neg is not None:
        data_dict["test_neg"] = test_neg
    key_to_loc = save_data(f"{name}__task_{task_str}_{task_id}",
                           save_dir=save_dir,
                           **data_dict)
    task_dict.update(key_to_loc)

    # Save the task json file.
    with open(os.path.join(save_dir, f"task_{task_str}_{task_id}.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(task_dict, f, indent=4)

    return task_dict
