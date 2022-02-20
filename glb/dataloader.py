from torch.utils.data.dataloader import DataLoader
from .dataset import Dataset


class NodeDataLoader(DataLoader):
    """Base class for node dataloader."""

    def __init__(self, dataset, sampler=None, batch_size=0, shuffle=False):
        """Initialize node dataloader.
        
        When sampler and batch_size are not specified, the entire dataset will be fed into model for
        each iteration. Otherwise, the dataloader uses a given node batch sampler to
        sample mini-batches.
        """
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = shuffle
        if sampler:
            raise NotImplementedError
        if batch_size != 0:
            raise NotImplementedError
        if shuffle:
            raise NotImplementedError

    def __iter__(self):
        return iter([self.dataset[0],])