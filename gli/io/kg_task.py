"""Knowledge graph task related helper functions."""
import os
import json
import numpy as np
from .utils import check_feature
from ..utils import save_data


def save_task_kg_entity_prediction(name,
                                   description,
                                   feature,
                                   train_triplet_set,
                                   val_triplet_set,
                                   test_triplet_set,
                                   num_relations=0,
                                   predict_tail=True,
                                   task_id=1,
                                   save_dir="."):
    """Save the kg entity prediction task information into task json and data files.

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
    :param train_triplet_set: A list of training edge indices.
    :type train_triplet_set: list/array of int
    :param val_triplet_set: A list of validation edge indices.
    :type val_triplet_set: list/array of int
    :param val_triplet_set: A list of testing edge indices.
    :type val_triplet_set: list/array of int
    :param num_relations: The total number of different relations between entities.
        Default: 0.
    :type num_relations: int
    :param predict_tail: The bool determines whether it predicts tail.
        Default: True.
    :type predict_tail: Bool

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

        train_triplet_set = [0, 1]
        val_triplet_set = [2, 3]
        test_triplet_set = [4, 5]

        # Save the task information.
        save_task_kg_entity_prediction(
            name="example_dataset",
            description="A kg entity prediction task for the example dataset.",
            feature=["Node/DenseNodeFeature", "Node/SparseNodeFeature"],
            train_triplet_set=train_triplet_set,
            val_triplet_set=val_triplet_set,
            test_triplet_set=test_triplet_set)
        # This function will save the task information into a json file named
        # `task_kg_entity_prediction_1.json` and one numpy data file storing the
        # data splits, `train_triplet_set`, `val_triplet_set`, and `test_triplet_set`. The json file
        # will look like the following.

    .. code-block:: json

        {
            "description": "A kg entity prediction task for the example dataset.",
            "type": "KGEntityPrediction",
            "feature": [
                "Node/DenseNodeFeature",
                "Node/SparseNodeFeature"
            ],
            "num_relations": 0,
            "predict_tail": True,
            "train_triplet_set": {
                "file": "example_dataset__task_kg_entity_prediction_1__d4e39475e078fa07e18e57fdda149d36.npz",
                "key": "train_triplet_set"
            },
            "val_triplet_set": {
                "file": "example_dataset__task_kg_entity_prediction_1__d4e39475e078fa07e18e57fdda149d36.npz",
                "key": "val_triplet_set"
            },
            "test_triplet_set": {
                "file": "example_dataset__task_kg_entity_prediction_1__d4e39475e078fa07e18e57fdda149d36.npz",
                "key": "test_triplet_set"
            }
        }
    """  # noqa: E501,E262  #pylint: disable=line-too-long
    assert isinstance(description, str), \
        "`description` must be a string."
    check_feature(feature)
    task_type = "KGEntityPrediction"
    task_str = "kg_entity_prediction"

    task_dict = {
        "description": description,
        "type": task_type,
        "feature": feature,
        "num_relations": num_relations,
        "predict_tail": predict_tail
    }
    data_dict = {
        "train_triplet_set": np.array(train_triplet_set),
        "val_triplet_set": np.array(val_triplet_set),
        "test_triplet_set": np.array(test_triplet_set)
    }
    key_to_loc = save_data(f"{name}__task_{task_str}_{task_id}",
                           save_dir=save_dir,
                           **data_dict)
    # Update the task dictionary with the data file names and keys.
    task_dict.update(key_to_loc)

    # Save the task json file.
    with open(os.path.join(save_dir, f"task_{task_str}_{task_id}.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(task_dict, f, indent=4)

    return task_dict


def save_task_kg_relation_prediction(name,
                                     description,
                                     feature,
                                     target,
                                     train_triplet_set,
                                     val_triplet_set,
                                     test_triplet_set,
                                     num_relations=0,
                                     task_id=1,
                                     save_dir="."):
    """Save the kg relation prediction task information into task json and data files.

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
    :param target: The attribute name as prediction target in the task. For
        a kg relation prediction task, the attribute should be a edge attribute
        with format "Edge/{edge_attr_name}".
    :type target: str
    :param train_triplet_set: A list of training edge indices.
    :type train_triplet_set: list/array of int
    :param val_triplet_set: A list of validation edge indices.
    :type val_triplet_set: list/array of int
    :param val_triplet_set: A list of testing edge indices.
    :type val_triplet_set: list/array of int
    :param num_relations: The total number of different relations between entities.
        Default: 0.
    :type num_relations: int
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

        train_triplet_set = [0, 1]
        val_triplet_set = [2, 3]
        test_triplet_set = [4, 5]

        # Save the task information.
        save_task_kg_relation_prediction(
            name="example_dataset",
            description="A kg entity prediction task for the example dataset.",
            feature=["Node/DenseNodeFeature", "Node/SparseNodeFeature"],
            target="Edge/EdgeClass",
            train_triplet_set=train_triplet_set,
            val_triplet_set=val_triplet_set,
            test_triplet_set=test_triplet_set)
        # This function will save the task information into a json file named
        # `task_kg_entity_prediction_1.json` and one numpy data file storing the
        # data splits, `train_triplet_set`, `val_triplet_set`, and `test_triplet_set`. The json file
        # will look like the following.

    .. code-block:: json

        {
            "description": "A kg entity prediction task for the example dataset.",
            "type": "KGRelationPrediction",
            "feature": [
                "Node/DenseNodeFeature",
                "Node/SparseNodeFeature"
            ],
            "target": "Edge/EdgeClass",
            "num_relations": 0,
            "train_triplet_set": {
                "file": "example_dataset__task_kg_relation_prediction_1__d4e39475e078fa07e18e57fdda149d36.npz",
                "key": "train_triplet_set"
            },
            "val_triplet_set": {
                "file": "example_dataset__task_kg_relation_prediction_1__d4e39475e078fa07e18e57fdda149d36.npz",
                "key": "val_triplet_set"
            },
            "test_triplet_set": {
                "file": "example_dataset__task_kg_relation_prediction_1__d4e39475e078fa07e18e57fdda149d36.npz",
                "key": "test_triplet_set"
            }
        }
    """  # noqa: E501,E262  #pylint: disable=line-too-long
    assert isinstance(description, str), \
        "`description` must be a string."
    assert isinstance(target, str), \
        "`target` must be a string."
    check_feature(feature)
    task_type = "KGRelationPrediction"
    task_str = "kg_relation_prediction"

    task_dict = {
        "description": description,
        "type": task_type,
        "feature": feature,
        "target": target,
        "num_relations": num_relations
    }
    data_dict = {
        "train_triplet_set": np.array(train_triplet_set),
        "val_triplet_set": np.array(val_triplet_set),
        "test_triplet_set": np.array(test_triplet_set)
    }
    key_to_loc = save_data(f"{name}__task_{task_str}_{task_id}",
                           save_dir=save_dir,
                           **data_dict)
    # Update the task dictionary with the data file names and keys.
    task_dict.update(key_to_loc)

    # Save the task json file.
    with open(os.path.join(save_dir, f"task_{task_str}_{task_id}.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(task_dict, f, indent=4)

    return task_dict
