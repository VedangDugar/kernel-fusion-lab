"""Correctness tests for every kernel, over all shapes and dtypes.

Run inside the container with the Triton interpreter enabled:

    docker run --rm -e TRITON_INTERPRET=1 -e PYTHONPATH=/work \
        -v "$PWD:/work" -w /work kfl:cpu pytest -q

A failure here means a kernel is wrong. Tolerances live in harness/config.py
and are derived from dtype epsilon; they are not to be widened to make a test
go green.
"""

from __future__ import annotations

import os

import pytest
import torch

from harness.config import DTYPE_NAMES, DTYPES, SHAPES
from harness.correctness import ASSERTED, run_case

if os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available():
    pytest.skip(
        "needs either a GPU or TRITON_INTERPRET=1 to execute kernels",
        allow_module_level=True,
    )


def _case_id(shape: tuple[int, int], dtype: torch.dtype) -> str:
    return f"{shape[0]}x{shape[1]}-{DTYPE_NAMES[dtype]}"


@pytest.mark.parametrize(
    "shape,dtype",
    [(s, d) for d in DTYPES for s in SHAPES],
    ids=[_case_id(s, d) for d in DTYPES for s in SHAPES],
)
def test_kernels_match_reference(shape: tuple[int, int], dtype: torch.dtype) -> None:
    failures = []
    for result in run_case(shape, dtype):
        if result.kind != ASSERTED or result.passed:
            continue
        detail = f"{result.case}: max_rel={result.max_rel_err:.3e}"
        if result.rtol is not None:
            detail += f" (rtol={result.rtol:g}, atol={result.atol:g})"
        if result.note:
            detail += f" [{result.note}]"
        failures.append(detail)
    assert not failures, "\n".join(failures)
