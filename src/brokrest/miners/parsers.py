# Copyright (c) The BrokRest Authors - All Rights Reserved

import abc
import dataclasses as dcls
import logging
import re
import typing
from abc import ABC
from collections.abc import Iterable, Sequence
from datetime import datetime as DateTime
from typing import Any, ClassVar, Protocol

from pandas import DataFrame, Series

__all__ = [
    "ConsumeParser",
    "ConsumeParserStep",
    "TimeParser",
    "ModuleParser",
    "LogLevelParser",
    "StripConsumer",
    "CompositeParser",
]


LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class ParserResult[T]:
    _: dcls.KW_ONLY

    parsed: T | None = None
    remains: str

    def __post_init__(self):
        assert isinstance(self.remains, str)


class ConsumeParser(Protocol):
    def __call__(self, df: DataFrame) -> DataFrame: ...


class ConsumeParserStep(ConsumeParser, ABC):
    input: ClassVar[str] = "text"
    """
    The input column to parse.
    """

    output: ClassVar[str] = ""
    """
    The output column to parse.
    If not overwritten (empty string), discards the results.
    """

    @typing.override
    @typing.final
    def __call__(self, df: DataFrame) -> DataFrame:
        input_text: list[str] = df[self.input].to_list()

        parsed = [self._process(t) for t in input_text]

        df = df.copy()
        df[self.input] = Series([p.remains for p in parsed])

        if self.output:
            assert all(p is not None for p in parsed)
            df[self.output] = Series([p.parsed for p in parsed])

        return df

    @abc.abstractmethod
    def _process(self, text: str, /) -> ParserResult: ...


@dcls.dataclass(frozen=True)
class TimeParser(ConsumeParserStep):
    output = "time"

    @typing.override
    def _process(self, text: str) -> ParserResult:
        LOGGER.debug("Calling %s on %s", self, text)
        fmt = r"(\d\d\d\d)-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d),(\d\d\d)"
        found = re.match(fmt, text)
        assert found

        # Using `Any` to silence type checker for now.
        found_groups: Any = [int(found.group(i)) for i in range(1, 8)]
        # Total 7 groups.
        time = DateTime(*found_groups)
        remains = re.sub(fmt, "", text)
        return ParserResult(parsed=time, remains=remains)


@dcls.dataclass(frozen=True)
class ModuleParser(ConsumeParserStep):
    output = "module"

    @typing.override
    def _process(self, text: str) -> ParserResult:
        alpha = f"A-Za-z_"
        var_name = f"[{alpha}][{alpha}0-9]*"
        module = f"{var_name}(.{var_name})*"
        found = re.match(module, text)
        assert found

        remains = re.sub(module, "", text, count=1)

        return ParserResult(parsed=found.group(0), remains=remains)


@dcls.dataclass(frozen=True)
class LogLevelParser(ConsumeParserStep):
    output = "level"

    levels: Iterable[str] = frozenset(
        ["NOTSET", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
    )

    @typing.override
    def _process(self, text: str) -> ParserResult:
        for level in self.levels:
            if not text.startswith(level):
                continue

            return ParserResult(parsed=text[: len(level)], remains=text[len(level) :])
        raise ValueError("No level found.")


@dcls.dataclass(frozen=True)
class StripConsumer(ConsumeParserStep):
    output = ""
    string: str

    @typing.override
    def _process(self, text: str) -> ParserResult:
        text = text.strip(self.string)
        return ParserResult(remains=text)


@dcls.dataclass(frozen=True)
class CompositeParser(ConsumeParser):
    pipeline: Sequence[ConsumeParserStep]

    @typing.override
    def __call__(self, df: DataFrame) -> DataFrame:
        for parser in self.pipeline:
            df = parser(df)
        return df
