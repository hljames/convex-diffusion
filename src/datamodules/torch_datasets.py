from typing import Tuple, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


class MyTensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self,
                 *tensors: Tensor,
                 dataset_id: str = '',
                 ) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.dataset_id = dataset_id

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


def get_tensor_dataset_from_numpy(*ndarrays, dataset_id="", **kwargs) -> MyTensorDataset:
    tensors = [torch.from_numpy(ndarray.copy()).float() for ndarray in ndarrays]
    return MyTensorDataset(*tensors, dataset_id=dataset_id, **kwargs)


class AutoregressiveDynamicsTensorDataset(Dataset[Tuple[Tensor, ...]]):
    data: Tensor

    def __init__(self, data, horizon: int = 1, **kwargs):
        assert horizon > 0, f"horizon must be > 0, but is {horizon}"
        self.data = data
        self.horizon = horizon

    def __getitem__(self, index):
        # input: index time step
        # output: index + horizon time-steps ahead
        return self.data[index], self.data[index + self.horizon]

    def __len__(self):
        return len(self.data) - self.horizon
