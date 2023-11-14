"""Root entry for gli.io."""
from .graph import save_graph, save_homograph, save_heterograph, Attribute
from .node_task import save_task_node_classification, save_task_node_regression
from .edge_task import save_task_link_prediction, \
    save_task_time_dependent_link_prediction
from .graph_task import save_task_graph_classification, \
    save_task_graph_regression
from .kg_task import save_task_kg_entity_prediction, \
    save_task_kg_relation_prediction
