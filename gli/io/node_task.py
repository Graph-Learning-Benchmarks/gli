"""Node task related helper functions."""
from .utils import save_task_reg_or_cls


def save_task_node_regression(name,
                              description,
                              feature,
                              target,
                              train_set=None,
                              val_set=None,
                              test_set=None,
                              train_ratio=0.8,
                              val_ratio=0.1,
                              test_ratio=0.1,
                              num_samples=None,
                              task_id=1,
                              save_dir="."):
    """Save the node regression task information into task json and data files.

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
        this node regression task, the attribute should be a node attribute.
        For homogeneous graphs, the attribute name should be in the format of
        "Node/{node_attr_name}". For heterogeneous graphs, the attribute name
        should be in the format of "Node/{node_type}/{node_attr_name}". The
        node_attr_name and the node_type should be the ones declared in the
        metadata.json file.
    :type target: str
    :param train_set: A list of training node IDs or a list of list training
        node IDs. In the latter case, each inner list is the node IDs of one
        data split. If not None, `train_ratio`, `val_ratio`, and `test_ratio`
        will be ignored while `val_set` and `test_set` must present. If None,
        the task json file will store `train_ratio`, `val_ratio`, and
        `test_ratio` and random splits will be generated at run time. Default:
        None.
    :type train_set: list/array of int or list of lists/2-d array of int
    :param val_set: A list of validation node IDs or a list of list validation
        node IDs. See `train_set` for more details. Default: None.
    :type val_set: list/array of int or list of lists/2-d array of int
    :param test_set: A list of test node IDs or a list of list test node IDs.
        See `train_set` for more details. Default: None.
    :type test_set: list/array of int or list of lists/2-d array of int
    :param train_ratio: The ratio of training nodes. See `train_set` for more
        details. Default: 0.8.
    :type train_ratio: float
    :param val_ratio: The ratio of validation nodes. See `train_set` for more
        details. Default: 0.1.
    :type val_ratio: float
    :param test_ratio: The ratio of test nodes. See `train_set` for more
        details. Default: 0.1.
    :type test_ratio: float
    :param num_samples: The total number of nodes in the dataset. This needs to
        be provided if `train_set`, `val_set`, and `test_set` are not provided.
        Default: None.
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

    Example
    -------
    .. code-block:: python

        train_set = [0, 1]
        val_set = [2, 3]
        test_set = [4, 5]

        # Save the task information.
        save_task_node_regression(
            name="example_dataset",
            description="A node regression task for the example dataset.",
            feature=["Node/DenseNodeFeature", "Node/SparseNodeFeature"],
            target="Node/NodeLabel",
            train_set=train_set,
            val_set=val_set,
            test_set=test_set)
        # This function will save the task information into a json file named
        # `task_node_regression_1.json` and one numpy data file storing the
        # data splits, `train_set`, `val_set`, and `test_set`. The json file
        # will look like the following.

    .. code-block:: json

        {
            "description": "A node regression task for the example dataset.",
            "type": "NodeRegression",
            "feature": [
                "Node/DenseNodeFeature",
                "Node/SparseNodeFeature"
            ],
            "target": "Node/NodeLabel",
            "train_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "train_set"
            },
            "val_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "val_set"
            },
            "test_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "test_set"
            }
        }
    """  # noqa: E501,E262  #pylint: disable=line-too-long
    return save_task_reg_or_cls(task_type="NodeRegression",
                                name=name,
                                description=description,
                                feature=feature,
                                target=target,
                                train_set=train_set,
                                val_set=val_set,
                                test_set=test_set,
                                train_ratio=train_ratio,
                                val_ratio=val_ratio,
                                test_ratio=test_ratio,
                                num_samples=num_samples,
                                task_id=task_id,
                                save_dir=save_dir)


def save_task_node_classification(name,
                                  description,
                                  feature,
                                  target,
                                  num_classes,
                                  train_set=None,
                                  val_set=None,
                                  test_set=None,
                                  train_ratio=0.8,
                                  val_ratio=0.1,
                                  test_ratio=0.1,
                                  num_samples=None,
                                  task_id=1,
                                  save_dir="."):
    """Save the node classification task information into task json and data files.

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
        a node classification task, the attribute should be a node attribute.
        For homogeneous graphs, the attribute name should be in the format of
        "Node/{node_attr_name}". For heterogeneous graphs, the attribute name
        should be in the format of "Node/{node_type}/{node_attr_name}". The
        node_attr_name and the node_type should be the ones declared in the
        metadata.json file.
    :type target: str
    :param num_classes: The number of classes in the task.
    :type num_classes: int
    :param train_set: A list of training node IDs or a list of list training
        node IDs. In the latter case, each inner list is the node IDs of one
        data split. If not None, `train_ratio`, `val_ratio`, and `test_ratio`
        will be ignored while `val_set` and `test_set` must present. If None,
        the task json file will store `train_ratio`, `val_ratio`, and
        `test_ratio` and random splits will be generated at run time. Default:
        None.
    :type train_set: list/array of int or list of lists/2-d array of int
    :param val_set: A list of validation node IDs or a list of list validation
        node IDs. See `train_set` for more details. Default: None.
    :type val_set: list/array of int or list of lists/2-d array of int
    :param test_set: A list of test node IDs or a list of list test node IDs.
        See `train_set` for more details. Default: None.
    :type test_set: list/array of int or list of lists/2-d array of int
    :param train_ratio: The ratio of training nodes. See `train_set` for more
        details. Default: 0.8.
    :type train_ratio: float
    :param val_ratio: The ratio of validation nodes. See `train_set` for more
        details. Default: 0.1.
    :type val_ratio: float
    :param test_ratio: The ratio of test nodes. See `train_set` for more
        details. Default: 0.1.
    :type test_ratio: float
    :param num_samples: The total number of nodes in the dataset. This needs to
        be provided if `train_set`, `val_set`, and `test_set` are not provided.
        Default: None.
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

    Example
    -------
    .. code-block:: python

        train_set = [0, 1]
        val_set = [2, 3]
        test_set = [4, 5]

        # Save the task information.
        save_task_node_classification(
            name="example_dataset",
            description="A node classification task for the example dataset.",
            feature=["Node/DenseNodeFeature", "Node/SparseNodeFeature"],
            target="Node/NodeLabel",
            num_classes=4,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set)
        # This function will save the task information into a json file named
        # `task_node_classification_1.json` and one numpy data file storing the
        # data splits, `train_set`, `val_set`, and `test_set`. The json file
        # will look like the following.

    .. code-block:: json

        {
            "description": "A node classification task for the example dataset.",
            "type": "NodeClassification",
            "feature": [
                "Node/DenseNodeFeature",
                "Node/SparseNodeFeature"
            ],
            "target": "Node/NodeLabel",
            "num_classes": 4,
            "train_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "train_set"
            },
            "val_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "val_set"
            },
            "test_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "test_set"
            }
        }
    """  # noqa: E501,E262  #pylint: disable=line-too-long
    return save_task_reg_or_cls(task_type="NodeClassification",
                                name=name,
                                description=description,
                                feature=feature,
                                target=target,
                                num_classes=num_classes,
                                train_set=train_set,
                                val_set=val_set,
                                test_set=test_set,
                                train_ratio=train_ratio,
                                val_ratio=val_ratio,
                                test_ratio=test_ratio,
                                num_samples=num_samples,
                                task_id=task_id,
                                save_dir=save_dir)
