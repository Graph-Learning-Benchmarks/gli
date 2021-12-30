import os
import csv
from scipy.sparse import load_npz


class Attribute(object):
    def __init__(self):
        super().__init__()
        
    def transform(self):
        raise NotImplementedError("Method `transform` not implemented.")


class SparseTensor(Attribute):
    def __init__(self, file, index=None):
        super().__init__()
        self.file = file  # npz file
        self.index = index  # csv file
        
        self.tensor = load_npz(file)
        self.n = self.tensor.shape[0]
        
        if index is not None:
            with open(index) as csv_file:
                csv_reader = csv.reader(csv_file)
                self.index_dict = {}
                for row in csv_reader:
                    self.index[row[0]] = int(row[1])
        else:
            self.index_dict = None
            
    def _get_idx(self, instance):
        if self.index_dict is not None:
            idx = self.index_dict[str(instance)]
        else:
            idx = instance
        assert type(idx) == int, "Index is not integer."
        assert idx < self.n, "Index value exceeds {}.".format(self.n)
        return idx
        
    def transform(self, instance):
        return self.tensor[self._get_idx(instance)]


if __name__ == "__main__":
    path = os.path.abspath(__file__)
    root = os.path.split(os.path.split(path)[0])[0]

    file = os.path.join(root, "examples/cora/data/cora_feat.npz")
    sparse_tensor_attr = SparseTensor(file)
    print(sparse_tensor_attr.transform(0))
