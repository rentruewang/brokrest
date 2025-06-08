# Copyright (c) The BrokRest Authors - All Rights Reserved

import fire
import fs
import numpy as np
import pandas as pd


def main(history_zip: str, pair: str):
    history_fs = fs.open_fs(f"zip://{history_zip}")

    with history_fs.open(f"{pair}.csv") as f:
        price = np.array(pd.read_csv(f))[:, 1]
        print(max(price))


if __name__ == "__main__":
    fire.Fire(main)
