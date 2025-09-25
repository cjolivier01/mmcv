"""Compatibility shims for legacy mmcv.parallel API.

Provides DataContainer, collate, and MMDataParallel wrappers used by mmaction.
The implementations are minimal to satisfy usage in this repo.
"""
from typing import Any, Dict, List

import torch


class DataContainer:
    """A minimal DataContainer to carry data through collation.

    Only supports attributes used in this codebase: .data, indexing and len().
    """

    def __init__(self, data: Any, stack: bool = False, pad_dims: int = None, cpu_only: bool = False):
        self.data = data
        self.stack = stack
        self.pad_dims = pad_dims
        self.cpu_only = cpu_only

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        try:
            return len(self.data)
        except Exception:
            return 1


def collate(batch: List[Dict[str, Any]], samples_per_gpu: int = 1) -> Dict[str, Any]:
    """A very simple collate function that groups DataContainer and tensors.

    - For DataContainer values, returns a new DataContainer with list of items.
    - For tensors, uses default torch collate.
    """
    out: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if isinstance(values[0], DataContainer):
            # gather underlying data into list
            out[key] = DataContainer([v.data for v in values],
                                     stack=values[0].stack,
                                     pad_dims=values[0].pad_dims,
                                     cpu_only=values[0].cpu_only)
        else:
            out[key] = torch.utils.data.dataloader.default_collate(values)
    return out


class MMDataParallel(torch.nn.DataParallel):
    pass


def scatter(data: Dict[str, Any], device_ids: List[int]):
    """Minimal scatter that returns data on the first device without copying.

    For compatibility only; assumes the model handles device placement.
    """
    return [data]
