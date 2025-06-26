# Copyright (c) The BrokRest Authors - All Rights Reserved

import numpy as np
from numpy.typing import NDArray, ArrayLike
import fire
import pandas as pd


def gains(ticks: ArrayLike):
    ticks = np.array(ticks)
    gains = ticks[1:] / ticks[:-1]

    print(f'Over {len(ticks)} ticks')

    up_start = gains >= 1
    down_start = gains < 1

    up_acc = np.multiply.accumulate([1, *gains[up_start]])
    down_acc = np.multiply.accumulate([1, *gains[down_start]])

    return up_acc[-1], down_acc[-1]


def main(csv: str):
    df = pd.read_csv(csv)
    best, worst = gains([float(p.replace(",", "")) for p in df["Open"]])
    print(f"{best=} {worst=}")


if __name__ == "__main__":
    fire.Fire(main)
