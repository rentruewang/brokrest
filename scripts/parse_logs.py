# Copyright (c) The BrokRest Authors - All Rights Reserved

import functools
import glob
from pathlib import Path

import alive_progress as ap
import fire
from pandas import DataFrame

from brokrest.miners import (
    CompositeParser,
    LogLevelParser,
    ModuleParser,
    StripConsumer,
    TimeParser,
)


@functools.cache
def log_parser():
    return CompositeParser(
        [
            TimeParser(),
            StripConsumer(" -"),
            ModuleParser(),
            StripConsumer(" -"),
            LogLevelParser(),
            StripConsumer(" -"),
        ]
    )


def write_output(df: DataFrame, output: str):
    # Create new folder, and gitignore
    out_path = Path(output)
    out_path.mkdir(exist_ok=True)
    gitignore = out_path / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*")

    for module, group in df.groupby("module"):
        group.to_csv(Path(output) / f"{module}.csv")


def main(pattern: str, output: str = "out"):
    file_matches = glob.glob(pattern)

    lines: list[str] = []
    for file in ap.alive_it(sorted(file_matches)):
        with open(file) as f:
            lines.extend(line.strip() for line in f.readlines())

    df = DataFrame({"text": lines})

    parser = log_parser()

    df = parser(df)
    write_output(df=df, output=output)


if __name__ == "__main__":
    fire.Fire(main)
