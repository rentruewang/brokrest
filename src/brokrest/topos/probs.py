# Copyright (c) The BrokRest Authors - All Rights Reserved

import typing

import torch

__all__ = ["Importance"]


class Importance(typing.Protocol):
    """
    A 1D probabilitiy distribution that works in batches.

    It computes the input distance values, whose +- signs show above / below the lines),
    and re-sample their importances.

    Outputs an importance matrix (0-1 matrix, with 0 meaning no importance)
    that will be elementwise multiplied on the input, all >= 0, and should sum to 1.
    """

    def __call__(self, dists: torch.Tensor, /) -> torch.Tensor: ...
