#!/usr/bin/env python3
"""A tiny PASS/FAIL recorder shared by the Step A and Step B validators.

The repo's convention is that every validator prints one ``PASS``/``FAIL`` line
per check and exits non-zero if anything failed, so a human can read the output
and a shell script can gate on the exit code. Running the same validators over
30 episodes adds a second requirement: the result has to be *collectable* into a
machine-readable report without re-parsing stdout.

This module is that one abstraction. ``Checker.check()`` prints exactly the line
the existing scripts print and simultaneously records it, so nothing about the
human-facing output changes while the tier report gets structured data.

There is deliberately no "skip" or "warn" state for a failed assertion: a check
either passes or fails. Observations that are not pass/fail go through
``info()``/``metric()`` and are reported, never silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

PASS = "  PASS  "
FAIL = "  FAIL  "
INFO = "  ..    "
WARN = "  NOTE  "


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool


@dataclass
class Checker:
    """Collects check results while printing them in the house format."""

    emit: Callable[[str], None] = print
    results: list[CheckResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str | bool | None] = field(default_factory=dict)

    def check(self, ok: bool, message: str) -> bool:
        """Record and print one pass/fail assertion."""
        self.emit((PASS if ok else FAIL) + message)
        self.results.append(CheckResult(name=message, ok=bool(ok)))
        return bool(ok)

    def info(self, message: str) -> None:
        self.emit(INFO + message)

    def note(self, message: str) -> None:
        """A reportable observation that is not a pass/fail assertion."""
        self.emit(WARN + message)
        self.notes.append(message)

    def section(self, title: str) -> None:
        self.emit("")
        self.emit(f"=== {title} ===")

    def metric(self, name: str, value) -> None:
        """Record a number for the tier report (not printed on its own)."""
        self.metrics[name] = value

    @property
    def failures(self) -> list[str]:
        return [result.name for result in self.results if not result.ok]

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.ok)

    def summary(self) -> None:
        """Print the closing line the verify scripts grep for."""
        self.emit("")
        if self.failed == 0:
            self.emit("ALL CHECKS PASSED")
        else:
            self.emit(f"{self.failed} CHECK(S) FAILED")

    def as_dict(self) -> dict:
        return {
            "checks_total": len(self.results),
            "checks_passed": self.passed,
            "checks_failed": self.failed,
            "failures": self.failures,
            "notes": list(self.notes),
            "metrics": dict(self.metrics),
        }
