"""Correctness comparison between Triton kernels and PyTorch references.

Two kinds of check are produced.

ASSERTED checks have a tolerance and can fail the build:

  rmsnorm / softmax / fused
      Each kernel against its same-precision PyTorch reference. Tolerances
      come from harness.config and are derived from dtype epsilon.

  fusion_not_less_accurate
      The fused kernel must be at least as close to an fp64 ground truth as
      the unfused two-kernel sequence is. This replaces a naive "fused and
      unfused agree" check, which is the wrong question in low precision:
      the two implementations disagree precisely because the unfused one is
      worse, and asserting agreement would either fail spuriously or force
      the tolerance to be widened until it hid real errors.

REPORTED checks carry no tolerance and never fail. They exist so the numbers
appear in the results table:

  fused_vs_unfused        how far apart the two pipelines land
  fused_accuracy_fp64     fused kernel error against fp64 ground truth
  unfused_accuracy_fp64   unfused sequence error against the same ground truth
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from harness.config import DTYPE_NAMES, EPS, TOLERANCES, make_inputs
from kernels.fused_rmsnorm_softmax import fused_rmsnorm_softmax
from kernels.rmsnorm import rmsnorm as rmsnorm_kernel
from kernels.softmax import softmax as softmax_kernel
from reference import torch_reference as ref

# Relative error is meaningless where the reference is essentially zero, so
# those elements are excluded from the relative statistic. They remain covered
# by the absolute check.
_REL_FLOOR = 1e-30

ASSERTED = "assert"
REPORTED = "report"


@dataclass(frozen=True)
class Result:
    case: str
    shape: tuple[int, int]
    dtype: str
    max_abs_err: float
    max_rel_err: float
    kind: str
    passed: bool
    rtol: float | None = None
    atol: float | None = None
    note: str = ""

    @property
    def verdict(self) -> str:
        if self.kind == REPORTED:
            return "report"
        return "PASS" if self.passed else "FAIL"


def _errors(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    """Max absolute error, max relative error, and the elementwise absolute error."""
    a = got.to(torch.float64)
    b = want.to(torch.float64)
    abs_err = (a - b).abs()
    max_abs = abs_err.max().item()

    significant = b.abs() > _REL_FLOOR
    if significant.any():
        max_rel = (abs_err[significant] / b[significant].abs()).max().item()
    else:
        max_rel = 0.0
    return max_abs, max_rel, abs_err


def compare(
    case: str,
    got: torch.Tensor,
    want: torch.Tensor,
    shape: tuple[int, int],
    dtype: torch.dtype,
) -> Result:
    """Asserted check against a same-precision reference."""
    tol = TOLERANCES[dtype]
    max_abs, max_rel, abs_err = _errors(got, want)
    passed = bool((abs_err <= tol.atol + tol.rtol * want.to(torch.float64).abs()).all())
    return Result(
        case=case,
        shape=shape,
        dtype=DTYPE_NAMES[dtype],
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        kind=ASSERTED,
        passed=passed,
        rtol=tol.rtol,
        atol=tol.atol,
    )


def report(
    case: str,
    got: torch.Tensor,
    want: torch.Tensor,
    shape: tuple[int, int],
    dtype: torch.dtype,
    note: str = "",
) -> Result:
    """Measurement with no pass/fail semantics."""
    max_abs, max_rel, _ = _errors(got, want)
    return Result(
        case=case,
        shape=shape,
        dtype=DTYPE_NAMES[dtype],
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        kind=REPORTED,
        passed=True,
        note=note,
    )


def run_case(shape: tuple[int, int], dtype: torch.dtype) -> list[Result]:
    """Every correctness check for one (shape, dtype) combination."""
    x, weight = make_inputs(shape, dtype)
    results: list[Result] = []

    fused_out = fused_rmsnorm_softmax(x, weight, EPS)
    # The genuine two-kernel sequence: the intermediate really does round-trip
    # through memory in the storage dtype.
    unfused_out = softmax_kernel(rmsnorm_kernel(x, weight, EPS))
    truth = ref.ground_truth(x, weight, EPS)

    # --- asserted: each kernel against its same-precision reference ---
    results.append(
        compare("rmsnorm", rmsnorm_kernel(x, weight, EPS), ref.rmsnorm(x, weight, EPS), shape, dtype)
    )
    results.append(compare("softmax", softmax_kernel(x), ref.softmax(x), shape, dtype))
    results.append(compare("fused", fused_out, ref.fused_pipeline(x, weight, EPS), shape, dtype))

    # --- reported: how the two pipelines relate ---
    _, fused_rel, _ = _errors(fused_out, truth)
    _, unfused_rel, _ = _errors(unfused_out, truth)

    results.append(
        report(
            "fused_vs_unfused",
            fused_out,
            unfused_out,
            shape,
            dtype,
            note="divergence, not an error; driven by intermediate rounding",
        )
    )
    results.append(
        report("fused_accuracy_fp64", fused_out, truth, shape, dtype, note="lower is better")
    )
    results.append(
        report("unfused_accuracy_fp64", unfused_out, truth, shape, dtype, note="lower is better")
    )

    # --- asserted: fusion must not cost accuracy ---
    # Exact equality is allowed (fp32, where the intermediate rounding is
    # negligible and both paths agree). Any slack beyond that would be an
    # excuse, so there is none.
    results.append(
        Result(
            case="fusion_not_less_accurate",
            shape=shape,
            dtype=DTYPE_NAMES[dtype],
            max_abs_err=float("nan"),
            max_rel_err=fused_rel,
            kind=ASSERTED,
            passed=bool(fused_rel <= unfused_rel),
            note=f"fused {fused_rel:.3e} vs unfused {unfused_rel:.3e} against fp64",
        )
    )

    return results


def run_all(shapes, dtypes) -> list[Result]:
    results: list[Result] = []
    for dtype in dtypes:
        for shape in shapes:
            results.extend(run_case(shape, dtype))
    return results
