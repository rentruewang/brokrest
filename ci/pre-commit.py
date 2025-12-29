# Copyright (c) AIoWay Authors - All Rights Reserved

from argparse import ArgumentParser

import gha
import pdm


def parse_args() -> str:

    parser = ArgumentParser()
    parser.add_argument("command", type=str)

    flags = vars(parser.parse_args())

    return flags["command"]


def tools_for_command(command: str, /):
    match command:
        case "format":
            yield "autoflake"
            yield "isort"
            yield "black"
        case "typing":
            yield "mypy"
        case "linting":
            yield "pylint"
        case _:
            raise NotImplementedError(
                f"Support for '{command}' command is not yet implemented."
            )


if __name__ == "__main__":
    command = parse_args()

    gha.setup()
    pdm.sync()

    failed: list[str] = []

    for tool in tools_for_command(command):
        try:
            pdm.run(f"pre-commit run --all-files {tool}")
        except Exception:
            failed.append(tool)

    if failed:
        raise RuntimeError("Failed tools: " + ", ".join(f"'{tool}'" for tool in failed))
