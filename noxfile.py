# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import os
import subprocess as sp
import sys
from collections import abc as cabc

import nox

PYTHON_VERSIONS = ["3.14"]


@nox.session
def publish(session: nox.Session):
    "Nox `publish` command. Calls `pdm publish`."
    commands(session).publish()


@nox.session
def build(session: nox.Session):
    "Nox `build` command. Calls `pdm build`."
    commands(session).build()


@nox.session
def pre_commit(session: nox.Session):
    "Runs the pre-commit commands."

    formatting(session)
    typing(session)


@nox.session(python=PYTHON_VERSIONS)
def testing(session: nox.Session):
    "Nox `testing` command. Calls `pytest` command. Runs in multiple python versions."
    commands(session).test()


@nox.session
def formatting(session: nox.Session):
    "Nox `formatting` command. Calls `autoflake`, `isort`, `black`, in that order."
    autoflake(session)
    isort(session)
    black(session)


@nox.session
def autoflake(session: nox.Session):
    "Nox `autoflake` command. Calls `autoflake` command."
    commands(session).autoflake()


@nox.session
def isort(session: nox.Session):
    "Nox `isort` command. Calls `isort` command."
    commands(session).isort()


@nox.session
def black(session: nox.Session):
    "Nox `black` command. Calls `black` command."
    commands(session).black()


@nox.session
def mypy(session: nox.Session):
    "Nox `mypy` command. Calls `mypy` command."
    commands(session).mypy()


@nox.session
def typing(session: nox.Session):
    "Nox `typing` command. Calls `mypy` command."
    mypy(session)


@functools.cache
def github(session: nox.Session):
    "Global singleton of `github`."
    return _Github(session)


@functools.cache
def pdm(session: nox.Session):
    "Global singleton of `pdm`."
    return _Pdm(session)


@functools.cache
def commands(session: nox.Session):
    "Global singleton of `commands`."
    return _Commands(session)


@dcls.dataclass(frozen=True)
class _Github:
    "The manager for setting up github."

    session: nox.Session
    "The nox session to use."

    @functools.cache
    def setup(self) -> None:
        "The shared entrypoint to GitHub Actions scripts"

        # Does nothing outside of GitHub Actions.
        if not in_github_actions():
            return

        self._remove_unwanted_files()
        self._log_storage_usage()

    def _run(self, *args: str):
        self.session.run_install(*args, external=True)

    def _remove_unwanted_files(self) -> None:
        "Remove the files GitHub Actions pre-installed."

        print("Removing files we did not ask for...")

        for folder in [
            "/usr/local/lib/android",
            "/usr/share/dotnet",
            "/usr/local/.ghcup",
        ]:
            self._run("sudo", "rm", "-rf", folder)

        self._run("docker", "system", "prune", "-af", "--volumes")

    def _log_storage_usage(self) -> None:
        "Log how much usage is currently being used by GitHub Actions."
        print("Investigating how much storage is used in GitHub Actions...")

        self._run("df", "-h")


@dcls.dataclass(frozen=True)
class _Pdm:
    session: nox.Session

    def __post_init__(self):
        github(self.session).setup()

        if in_github_actions():
            self._run("pdm", "config", "python.use_venv", "true")

    def sync(self) -> None:
        self._sync_or_install("sync")

    def install(self):
        self._sync_or_install("install")

    def build(self):
        self.install()
        self._run("pdm", "build")

    def publish(self):
        self.install()
        self._run("pdm", "publish")

    def run(self, *args: str):
        self.sync()
        self._run("pdm", "run", *args)

    def _sync_or_install(self, mode: str) -> None:
        # Don't repeatedly reinstall locally.
        if not in_github_actions():
            return

        self.session.run_install("pdm", mode, "-G:all")

    def _run(self, *args: str):
        self.session.run(*args, external=True)


@dcls.dataclass(frozen=True)
class _Commands:
    session: nox.Session

    def __post_init__(self) -> None:
        github(self.session).setup()
        _install_ta_lib(self.session)

    def build(self):
        "`pdm build` command."
        self.pdm.build()

    def publish(self):
        "`pdm publish` command."
        self.pdm.publish()

    def test(self):
        "`pytest` command."
        self.pdm.run("pytest")

    def autoflake(self, path: str = "."):
        "`autoflake` command."
        self.pdm.run("autoflake", path)

    def isort(self, path: str = "."):
        "`isort` command."
        self.pdm.run("isort", path)

    def black(self, path: str = "."):
        "`black` command."
        self.pdm.run("black", path)

    def mypy(self, path: str = "src"):
        "`mypy` command."
        self.pdm.run("mypy", "--non-interactive", "--install-types", path)
        self.pdm.run("mypy", path)

    @property
    def pdm(self):
        return pdm(self.session)

    def _run(self, *args: str):
        self.session.run(*args, external=True)


def _install_talib_macos(session: nox.Session):
    if _has_brew_talib():
        return

    session.run("brew", "install", "ta-lib", external=True)


def _install_talib_linux(session: nox.Session):
    if _has_talib_linux_binding():
        return

    session.run("git", "clone", "https://github.com/ta-lib/ta-lib/", external=True)
    with session.cd("ta-lib"):
        session.run("sudo", "./install", external=True)
    session.run("rm", "-rf", "ta-lib", external=True)


def _install_ta_lib(session: nox.Session):

    match sys.platform:
        case "darwin":
            _install_talib_macos(session)
        case "linux":
            _install_talib_linux(session)
        case _:
            raise RuntimeError(f"Platform '{sys.platform}' is not supported!")


def checking_if(condition: str):
    def decorator(function: cabc.Callable[[], bool]) -> cabc.Callable[[], bool]:
        def wrapper() -> bool:
            print(f"Checking if {condition}...", end=" ")
            answer = function()
            print("Yes" if answer else "No")
            return answer

        return wrapper

    return decorator


@checking_if("we are in GitHub Actions")
def in_github_actions() -> bool:
    "Detect whether or not it is running in GitHub Actions."

    return os.getenv("GITHUB_ACTIONS") == "true"


@checking_if("ta-lib is installed in brew")
def _has_brew_talib():
    result = sp.run(
        ["brew", "list", "--versions", "ta-lib"],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
    )
    return result.stdout.strip() != ""


@checking_if("ta-lib is installed on linux")
def _has_talib_linux_binding():
    result = sp.run(
        ["ldconfig", "-p"],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
    )
    return "ta_lib" in result.stdout.lower()
