"""
GCNII model in GLI.

References:
https://github.com/dmlc/dgl
Pull request #5008
"""

import math
from torch import nn
import torch.nn.functional as F
from dgl.sparse import from_coo, diag, identity


class GCNIIConvolution(nn.Module):
    """GCNII Conv."""

    def __init__(self, in_size, out_size):
        """Init."""
        super().__init__()
        self.out_size = out_size
        self.weight = nn.Linear(in_size, out_size, bias=False)

    ###########################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement the GCNII
    # forward process.
    ###########################################################################
    def forward(self, a_norm, h, h0, lamda, alpha, index):
        """Forward."""
        beta = math.log(lamda / index + 1)

        # Multiply a sparse matrix by a dense matrix.
        h = a_norm @ h
        h = (1 - alpha) * h + alpha * h0
        h = (1 - beta) * h + beta * self.weight(h)
        return h


class GCNII(nn.Module):
    """GCNII network."""

    def __init__(
        self,
        g,
        in_size,
        out_size,
        hidden_size,
        n_layers,
        lambda_,
        alpha,
        dropout=0.5,
    ):
        """Init."""
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lambda_ = lambda_
        self.alpha = alpha

        # The GCNII model.
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size))
        for _ in range(n_layers):
            self.layers.append(GCNIIConvolution(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, out_size))

        self.activation = nn.ReLU()
        self.dropout = dropout

        # Create the adjacency matrix of graph.
        src, dst = g.edges()
        n = g.num_nodes()
        a = from_coo(dst, src, shape=(n, n))

        #####################################################
        # (HIGHLIGHT) Compute the symmetrically normalized adjacency matrix
        # with Sparse Matrix API
        #####################################################
        i = identity(a.shape)
        a_hat = a + i
        d_hat = diag(a_hat.sum(1)) ** -0.5
        self.a_norm = d_hat @ a_hat @ d_hat

    def forward(self, feature):
        """Fprward."""
        h = feature
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.layers[0](h)
        h = self.activation(h)
        h0 = h

        # The GCNII convolution forward.
        for i, conv in enumerate(self.layers[1:-1]):
            h = F.dropout(h, self.dropout, training=self.training)
            h = conv(self.a_norm, h, h0, self.lambda_, self.alpha, i + 1)
            h = self.activation(h)

        h = F.dropout(h, self.dropout, training=self.training)
        h = self.layers[-1](h)

        return h
