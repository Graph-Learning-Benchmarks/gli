"""
Config for GLB.

References:
https://github.com/pyg-team/pytorch_geometric/blob/575611f4f5e2209c7923dba977a1eebc207bd2e2/torch_geometric/graphgym/config.py
"""

import os
import shutil
import warnings


try:  # Define global config object
    from yacs.config import CfgNode as CN
    CFG = CN()
except ImportError:
    CFG = None
    warnings.warn("Could not define global config object. Please install "
                  "'yacs' for using the GraphGym experiment manager via "
                  "'pip install yacs'.")


def set_cfg(cfg):
    """
    Set the default config value.

    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name
    :return: configuration use by the experiment.
    """
    if cfg is None:
        return cfg

    # ----------------------------------------------------------------------- #
    # Basic options
    # ----------------------------------------------------------------------- #

    # Output directory
    cfg.out_dir = "results"

    # Config name (in out_dir)
    cfg.cfg_dest = "config.yaml"

    # ----------------------------------------------------------------------- #
    # Globally shared variables:
    # These variables will be set dynamically based on the input dataset
    # Do not directly set them here or in .yaml files
    # ----------------------------------------------------------------------- #

    cfg.share = CN()

    # Size of input dimension
    cfg.share.dim_in = 1

    # Size of out dimension, i.e., number of labels to be predicted
    cfg.share.dim_out = 1

    # Number of dataset splits: train/val/test
    cfg.share.num_splits = 1

    # ----------------------------------------------------------------------- #
    # Dataset options
    # ----------------------------------------------------------------------- #
    cfg.dataset = CN()

    # Name of the dataset
    cfg.dataset.name = "cora"

    # Whether add self loop
    cfg.dataset.self_loop = 1

    # Whether transform the dataset to dense
    cfg.dataset.to_dense = 0

    # if PyG: look for it in Pytorch Geometric dataset
    # if NetworkX/nx: load data in NetworkX format
    cfg.dataset.format = "PyG"

    # Dir to load the dataset. If the dataset is downloaded, this is the
    # cache dir
    cfg.dataset.dir = "./datasets"

    # Task: node, edge, graph, link_pred
    cfg.dataset.task = "node"

    # Type of task: classification, regression, classification_binary
    # classification_multi
    cfg.dataset.task_type = "classification"

    # ----------------------------------------------------------------------- #
    # Training options
    # ----------------------------------------------------------------------- #
    cfg.train = CN()

    # Skip re-evaluate the validation set
    cfg.train.fastmode = False

    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    cfg.model = CN()

    # Number of hidden layers
    cfg.model.num_layers = 2

    # Number of hidden units
    cfg.model.num_hidden = 8

    # Number of hidden attention heads"
    cfg.model.num_heads = 8

    # Number of output attention heads
    cfg.model.num_out_heads = 2

    # Whether to use residual connection
    cfg.model.residual = False

    # Dropout rate
    cfg.model.dropout = 0.6

    # The negative slope of leaky relu
    cfg.model.Negative_slope = 0.2

    # Pseudo coordinate dimensions in GMMConv
    cfg.model.pseudo_dim = 2

    # Number of kernels in GMMConv layer
    cfg.model.num_kernels = 3

    # Aggregator type: mean/gcn/pool/lstm
    cfg.model.aggregator_type = "gcn"

    # Loss function: cross_entropy, mse
    cfg.model.loss_fun = "cross_entropy"

    # ----------------------------------------------------------------------- #
    # Optimizer options
    # ----------------------------------------------------------------------- #
    cfg.optim = CN()

    # optimizer: sgd, adam
    cfg.optim.optimizer = "adam"

    # Base learning rate
    cfg.optim.lr = 0.01

    # L2 regularization
    cfg.optim.weight_decay = 5e-4

    # Maximal number of epochs
    cfg.optim.max_epoch = 2000


def load_cfg(cfg, args):
    """
    Load configurations from file system and command line.

    Args:
        cfg (CfgNode): Configuration node
        args (ArgumentParser): Command argument parser

    """
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)


def makedirs_rm_exist(dirt):
    """Make directory."""
    if os.path.isdir(dirt):
        shutil.rmtree(dirt)
    os.makedirs(dirt, exist_ok=True)


def get_fname(fname):
    """
    Extract filename from file name path.

    Args:
        fname (string): Filename for the yaml format configuration file
    """
    fname = fname.split("/")[-1]
    if fname.endswith(".yaml"):
        fname = fname[:-5]
    elif fname.endswith(".yml"):
        fname = fname[:-4]
    return fname


def set_out_dir(out_dir, fname):
    """
    Create the directory for full experiment run.

    Args:
        out_dir (string): Directory for output, specified in :obj:`cfg.out_dir`
        fname (string): Filename for the yaml format configuration file

    """
    fname = get_fname(fname)
    CFG.out_dir = os.path.join(out_dir, fname)
    # Make output directory
    makedirs_rm_exist(CFG.out_dir)


def set_run_dir(out_dir):
    """
    Create the directory for each random seed experiment run.

    Args:
        out_dir (string): Directory for output, specified in :obj:`cfg.out_dir`
        fname (string): Filename for the yaml format configuration file

    """
    CFG.run_dir = os.path.join(out_dir, str(CFG.seed))
    # Make output directory
    if CFG.train.auto_resume:
        os.makedirs(CFG.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(CFG.run_dir)


set_cfg(CFG)
