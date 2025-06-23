# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Sequence
from datetime import datetime as DateTime

from numpy.typing import NDArray


@dcls.dataclass(frozen=True)
class Order:
    time: DateTime
    price: float


@dcls.dataclass(frozen=True)
class TradeHistory:
    orders: Sequence[Order]
    sample_rate: float

    @property
    def open_signals(self) -> NDArray:
        raise NotImplementedError

    @property
    def close_signals(self) -> NDArray:
        raise NotImplementedError
