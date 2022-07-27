"""Utility functions."""
import torch.nn.functional as F
from models.gcn import GCN
from models.gat import GAT
from models.monet import MoNet
from models.graph_sage import GraphSAGE
from models.mlp import MLP
from models.mixhop import MixHop
from models.linkx import LINKX


def generate_model(args, g, in_feats, n_classes):
    """Generate required model."""
    # create model
    if args.model == "GCN":
        model = GCN(g,
                    in_feats,
                    args.num_hidden,
                    n_classes,
                    args.num_layers,
                    F.relu,
                    args.in_drop)
    elif args.model == "GAT":
        heads = ([args.num_heads] * (args.num_layers-1)) + [args.num_out_heads]
        model = GAT(g,
                    args.num_layers,
                    in_feats,
                    args.num_hidden,
                    n_classes,
                    heads,
                    F.relu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual)
    elif args.model == "MoNet":
        model = MoNet(g,
                      in_feats,
                      args.num_hidden,
                      n_classes,
                      args.num_layers,
                      args.pseudo_dim,
                      args.num_kernels,
                      args.in_drop)
    elif args.model == "GraphSAGE":
        model = GraphSAGE(g,
                          in_feats,
                          args.num_hidden,
                          n_classes,
                          args.num_layers,
                          F.relu,
                          args.in_drop,
                          args.aggregator_type)
    elif args.model == "MLP":
        model = MLP(in_feats,
                    args.num_hidden,
                    n_classes,
                    args.num_layers,
                    F.relu,
                    args.in_drop)
    elif args.model == "MixHop":
        model = MixHop(g,
                       in_dim=in_feats,
                       hid_dim=args.num_hidden,
                       out_dim=n_classes,
                       p=args.p,
                       num_layers=args.num_layers,
                       input_dropout=args.in_drop,
                       layer_dropout=args.layer_dropout,
                       activation=F.tanh,
                       batchnorm=False)
    elif args.model == "LINKX":
        model = LINKX(in_channels=in_feats,
                      num_nodes=g.ndata["NodeFeature"].shape[0],
                      hidden_channels=args.num_hidden,
                      out_channels=n_classes,
                      num_layers=args.num_layers,
                      dropout=args.in_drop,
                      inner_activation=args.inner_activation,
                      inner_dropout=args.inner_dropout,
                      init_layers_A=args.init_layers_A,
                      init_layers_X=args.init_layers_X)

    return model
