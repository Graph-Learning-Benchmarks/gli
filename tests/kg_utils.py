"""Utils function for KG."""
import torch
from torch import nn
from torch.utils.data import Dataset


def relation_negative_sample(batch, n_neg, nrange):
    """
    Add negative samples.

    Given a randomly sampled batch with shape (batch_size, 3),
    which represents a bunch of (head, tail, relation) triplets.
    Returns an array with shape (batch_size * (n_neg+1), 3) where the
    appended relations are negative_samples in the range of `nrange`.
    """
    batch_size = batch.shape[0]
    # (3, batch_size)
    batch = batch.T
    # (3, batch_size * (n_neg + 1) )
    batch = batch.repeat((1, n_neg+1))
    negatives = torch.randint(low=0, high=nrange, size=(1, n_neg * batch_size))
    batch[1][batch_size:] = negatives
    batch = batch.T


class KGDataset(Dataset):
    """KG Dataset."""

    def __init__(self, heads, tails, rels):
        """KG Dataset function."""
        self.heads = heads
        self.tails = tails
        self.rels = rels

    def __len__(self):
        """KG Dataset function."""
        return len(self.heads)

    def __getitem__(self, idx):
        """KG Dataset function."""
        data = {
            'batch_h': self.heads[idx],
            'batch_t': self.tails[idx],
            'batch_r': self.rels[idx],
        }
        return data


class TransE(nn.Module):
    """TransE network."""  # noqa: D403

    def __init__(
        self, ent_tot, rel_tot, dim=100, p_norm=1,
        norm_flag=True, margin=None, epsilon=None
    ):  # noqa: D403, R1725
        """TransE function."""
        super(TransE, self).__init__()  # pylint: disable=R1725
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        if margin is None or epsilon is None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]),
                requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if margin is not None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _calc(self, h, t, r):  # noqa: D403
        """TransE function."""
        score = h + r - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def forward(self, data):  # noqa: D403
        """TransE function."""
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h, t, r)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):  # noqa: D403
        """TransE function."""
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):  # noqa: D403
        """TransE function."""
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()
