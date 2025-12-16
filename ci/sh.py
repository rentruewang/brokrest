# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib
import os
import subprocess as sp
from pathlib import Path


def cmd(command: str, /) -> None:
    "Run a command in the shell."

    print()
    print(f">>> {command}")
    print()

    _ = sp.run(command, shell=True, check=True)


def root() -> Path:
    "The root location of the project."

    return Path(__file__).parent.parent.resolve()


@contextlib.contextmanager
def run_in_root():
    "Run the sub commands in the root directory."

    with chdir(root()):
        yield


@contextlib.contextmanager
def chdir(dest: str | Path):
    "Run the following commands in the ``dest`` folder."

    current = Path.cwd()

    # Execute the rest of the commands in the ``contextmanager`` in the directory.
    os.chdir(dest)

    try:
        yield

    # Go back when done.
    finally:
        os.chdir(current)
