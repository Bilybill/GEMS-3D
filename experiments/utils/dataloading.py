from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def build_data_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    prefetch_factor: int = 2,
):
    loader_kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
    return DataLoader(dataset, **loader_kwargs)
