import functools
import glob

import alive_progress as ap
import fire
import fire.fire_import_test
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


pattern = "./logs/*.log*"


def main(*patterns: str):
    file_matches = sum((glob.glob(pa) for pa in patterns), [])

    lines: list[str] = []
    for file in ap.alive_it(sorted(file_matches)):
        with open(file) as f:
            lines.extend(line.strip() for line in f.readlines())

    df = DataFrame({"text": lines})

    parser = log_parser()

    df = parser(df)
    print(df)


if __name__ == "__main__":
    fire.Fire(main)
