"""
[Simple and Deep Graph Convolutional Networks]
(https://arxiv.org/abs/2007.02133)
"""

import math
import torch.nn as nn
import torch.nn.functional as F
from dgl.sparse import from_coo, diag, identity


class GCNIIConvolution(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.out_size = out_size
        self.weight = nn.Linear(in_size, out_size, bias=False)

    ############################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement the GCNII
    # forward process.
    ############################################################################
    def forward(self, A_norm, H, H0, lamda, alpha, l):
        beta = math.log(lamda / l + 1)

        # Multiply a sparse matrix by a dense matrix.
        H = A_norm @ H
        H = (1 - alpha) * H + alpha * H0
        H = (1 - beta) * H + beta * self.weight(H)
        return H


class GCNII(nn.Module):
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
        N = g.num_nodes()
        A = from_coo(dst, src, shape=(N, N))

        ############################################################################
        # (HIGHLIGHT) Compute the symmetrically normalized adjacency matrix with
        # Sparse Matrix API
        ############################################################################
        I = identity(A.shape)
        A_hat = A + I
        D_hat = diag(A_hat.sum(1)) ** -0.5
        self.A_norm = D_hat @ A_hat @ D_hat

    def forward(self, feature):
        H = feature
        H = F.dropout(H, self.dropout, training=self.training)
        H = self.layers[0](H)
        H = self.activation(H)
        H0 = H

        # The GCNII convolution forward.
        for i, conv in enumerate(self.layers[1:-1]):
            H = F.dropout(H, self.dropout, training=self.training)
            H = conv(self.A_norm, H, H0, self.lambda_, self.alpha, i + 1)
            H = self.activation(H)

        H = F.dropout(H, self.dropout, training=self.training)
        H = self.layers[-1](H)

        return H
