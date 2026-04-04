# Copyright (c) The BrokRest Authors - All Rights Reserved

"The tensordict adaptor (overcoming the type check issues)."

import typing
from collections import abc as cabc

import tensordict as td
import torch
from numpy import typing as npt

__all__ = ["TensorClass", "tensorclass"]


@typing.dataclass_transform()
def tensorclass[T: type](typ: T) -> T:
    decorator = typing.cast(typing.Any, td.tensorclass)(autocast=True)
    return decorator(typ)


@tensorclass
class TensorClass:
    if typing.TYPE_CHECKING:

        def __contains__(self, key: str) -> bool: ...

        def __repr__(self) -> str: ...

        def __str__(self) -> str: ...

        def __len__(self) -> int: ...

        @typing.overload
        def __getitem__(self, key: str) -> torch.Tensor: ...

        @typing.overload
        @typing.overload
        def __getitem__(
            self, key: int | slice | list[int] | tuple | npt.NDArray | torch.Tensor
        ) -> typing.Self: ...

        def __getitem__(self, key):
            raise NotImplementedError

        def auto_batch_size_(self) -> typing.Self: ...

        def size(self) -> torch.Size: ...

        @property
        def shape(self) -> torch.Size: ...

        def flatten(self) -> typing.Self: ...

        @property
        def ndim(self) -> int: ...

        @property
        def batch_size(self) -> torch.Size: ...

        def numel(self) -> int: ...

        def to_dict(self) -> dict[str, typing.Any]: ...

        def item(self) -> typing.Any: ...

        def to(self, device: str | torch.device) -> typing.Self: ...

        def keys(self) -> cabc.KeysView[str]: ...

        def values(self) -> cabc.ValuesView[torch.Tensor]: ...

        def items(self) -> cabc.ItemsView[str, torch.Tensor]: ...
