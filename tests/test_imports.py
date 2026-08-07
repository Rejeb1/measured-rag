"""Import-order regression tests.

`ragmed.generate` and `ragmed.eval` legitimately depend on each other's concepts: the
generator writes the abstention sentinel, the eval layer reads it. That once formed a
real cycle (generate -> eval.generation_metrics -> eval/__init__ -> ablation -> runner
-> generate) which only surfaced when `ragmed.generate` was imported *first* - so the
whole test suite passed while `ragmed ask` crashed on startup.

A single pytest process caches modules, so the order that breaks is invisible after
anything else has imported. Each check therefore runs in a fresh interpreter.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Every public entry point, each as the very first ragmed import in a clean process.
ENTRY_POINTS = [
    "ragmed",
    "ragmed.generate",
    "ragmed.api",
    "ragmed.cli",
    "ragmed.eval",
    "ragmed.eval.runner",
    "ragmed.eval.ablation",
    "ragmed.eval.generation_metrics",
    "ragmed.retrieve.pipeline",
    "ragmed.index.store",
    "ragmed.ingest",
    "ragmed.llm",
]


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_module_imports_standalone(module: str):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"importing {module} first failed:\n{result.stderr}"
    )


def test_the_sentinel_has_exactly_one_definition():
    """Two copies would drift, and abstention would stop being measurable."""
    from ragmed.eval.generation_metrics import ABSTAIN_SENTINEL as from_eval
    from ragmed.generate import ABSTAIN_SENTINEL as from_generate
    from ragmed.types import ABSTAIN_SENTINEL as from_types

    assert from_types is from_eval is from_generate
