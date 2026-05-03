# Copyright (c) The BrokRest Authors - All Rights Reserved

import contextlib as ctxl

import torch
from numpy import typing as npt

_device: torch.device = torch.device("cpu")
"The device to run on. Right now `brokrest` supports running on a single device."


@ctxl.contextmanager
def device(device: str | torch.device):
    """
    Set the global device to the device.
    """

    global _device

    before = _device
    try:
        _device = torch.device(device)

        # Do this s.t. the template functions create tensors directly on the specified devices.
        with _device:
            yield _device
    finally:
        _device = before


def from_numpy(data: npt.NDArray) -> torch.Tensor:
    """
    The "patched" version of `torch.from_numpy`,
    because `torch`'s version does not allocate.

    So here we use `torch.tensor` if the default device is not cpu.
    """

    if _device == torch.device("cpu"):
        return torch.from_numpy(data)

    else:
        return torch.tensor(data)
