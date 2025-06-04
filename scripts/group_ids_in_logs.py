# Copyright (c) The BrokRest Authors - All Rights Reserved

import re
from collections import defaultdict as DefaultDict
from pathlib import Path
from typing import Any

import alive_progress as ap
import fire
import pandas as pd
from pandas import DataFrame


def main(csv: str, output="ids"):
    parent = Path(csv).parent

    df = pd.read_csv(csv)

    id_regex = re.compile(r"id\=(\d+)")

    result: dict[int, list[dict[str, Any]]] = DefaultDict(list)

    for record in ap.alive_it(df.to_records(index=False)):
        if matched := id_regex.search(record["text"]):
            found = int(matched.group(1))
            result[found].append(record)

    output_path = parent / output
    output_path.mkdir(exist_ok=True)
    gitignore = output_path / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*")

    for i, data in result.items():
        data_df = DataFrame.from_records(data, columns=df.columns)
        data_df.to_csv(output_path / f"{i}.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
