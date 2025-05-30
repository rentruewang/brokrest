# Copyright (c) The BrokRest Authors - All Rights Reserved

import glob
import logging
import re
from datetime import datetime as DateTime

import alive_progress as ap
import fire
from lark import Lark

LOGGER = logging.getLogger(__name__)

_TIME_REGEX = re.compile(r"\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d")
_TIME_FMT = r"%Y-%m-%d %H:%M:%S,%f"
_MODULE_REGEX = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*")


def parse_time(dt: str):
    return DateTime.strptime(dt, _TIME_FMT)


def parse_lines(lines: list[str]):
    dts = []

    pattern = f"({_TIME_REGEX}) - ({_MODULE_REGEX}) - "
    for line in lines:
        matched = re.match(pattern, line)
        LOGGER.debug("Matching %s on string %s, found %s", pattern, line, matched)

        if not matched:
            print(line, file=open("debug.txt", "w"))
        assert matched, line

        text = matched.group()
        dt = parse_time(text)
        dts.append(dt)

    return dts


def parse_one(fname: str):
    # 2025-05-24 17:00:22,519

    with open(fname) as f:
        lines = f.readlines()

    datetimes = parse_lines(lines)
    print(fname, min(datetimes), max(datetimes))


def main(*patterns: str):
    """
    Parse the files matching the given patterns.
    """

    file_matches = sum((glob.glob(pattern) for pattern in patterns), start=[])

    for file in ap.alive_it(sorted(file_matches)):
        parse_one(file)


if __name__ == "__main__":
    fire.Fire(main)
