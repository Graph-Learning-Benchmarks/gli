"""Utility functions."""
import torch.nn.functional as F
from models.gcn import GCN
from models.gat import GAT
from models.monet import MoNet
from models.GraphSAGE import GraphSAGE

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
                    F.elu,
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
    return model
