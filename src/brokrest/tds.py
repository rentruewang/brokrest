# Copyright (c) The BrokRest Authors - All Rights Reserved

"The tensordict adaptor (overcoming the type check issues)."
import typing
from collections.abc import ItemsView, KeysView, ValuesView
from typing import Any, Self

import tensordict as td
from numpy.typing import NDArray
from tensordict import TensorDict
from torch import Size, Tensor
from torch import device as Device

__all__ = ["TensorClass", "tensorclass"]


@typing.dataclass_transform()
@typing.no_type_check
def tensorclass[T: type](typ: T) -> T:
    return td.tensorclass(autocast=True)(typ)


@tensorclass
class TensorClass:
    if typing.TYPE_CHECKING:

        def __contains__(self, key: str) -> bool: ...

        def __repr__(self) -> str: ...

        def __str__(self) -> str: ...

        def __len__(self) -> int: ...

        @typing.overload
        def __getitem__(self, key: str) -> Tensor: ...

        @typing.overload
        def __getitem__(self, key: list[str]) -> TensorDict: ...

        @typing.overload
        def __getitem__(
            self, key: int | slice | list[int] | tuple | NDArray | Tensor
        ) -> Self: ...

        def __getitem__(self, key):
            raise NotImplementedError

        def auto_batch_size_(self) -> Self: ...

        def size(self) -> Size: ...

        @property
        def shape(self) -> Size: ...

        def flatten(self) -> Self: ...

        @property
        def ndim(self) -> int: ...

        @property
        def batch_size(self) -> Size: ...

        def numel(self) -> int: ...

        def to_dict(self) -> dict[str, Any]: ...

        def item(self) -> Any: ...

        def to(self, device: str | Device) -> Self: ...

        def keys(self) -> KeysView[str]: ...

        def values(self) -> ValuesView[Tensor]: ...

        def items(self) -> ItemsView[str, Tensor]: ...
