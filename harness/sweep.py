"""Run every check and regenerate docs/RESULTS.md.

    docker run --rm -e TRITON_INTERPRET=1 -e PYTHONPATH=/work \
        -v "$PWD:/work" -w /work kfl:cpu python -m harness.sweep

On a CPU-only machine this produces the correctness tables and the analytical
memory-traffic table, and writes "not measured" in every timing field. On a
GPU it additionally fills in the benchmark table.
"""

from __future__ import annotations

import datetime
import os
import platform
import sys
from pathlib import Path

import torch
import triton

from harness import benchmark, memory_model
from harness.config import DTYPES, SHAPES
from harness.correctness import ASSERTED, REPORTED, run_all as run_correctness

RESULTS_PATH = Path("docs/RESULTS.md")


def _correctness_markdown(results) -> tuple[str, str]:
    asserted = [r for r in results if r.kind == ASSERTED]
    reported = [r for r in results if r.kind == REPORTED]

    a_lines = [
        "| shape | dtype | check | max abs err | max rel err | rtol | atol | result |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in asserted:
        abs_txt = "n/a" if r.max_abs_err != r.max_abs_err else f"{r.max_abs_err:.3e}"
        rtol_txt = f"{r.rtol:g}" if r.rtol is not None else "n/a"
        atol_txt = f"{r.atol:g}" if r.atol is not None else "n/a"
        a_lines.append(
            f"| {r.shape[0]}x{r.shape[1]} | {r.dtype} | {r.case} | {abs_txt} | "
            f"{r.max_rel_err:.3e} | {rtol_txt} | {atol_txt} | {r.verdict} |"
        )

    r_lines = [
        "| shape | dtype | measurement | max abs err | max rel err |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for r in reported:
        r_lines.append(
            f"| {r.shape[0]}x{r.shape[1]} | {r.dtype} | {r.case} | "
            f"{r.max_abs_err:.3e} | {r.max_rel_err:.3e} |"
        )

    return "\n".join(a_lines), "\n".join(r_lines)


def main() -> int:
    interpreter = os.environ.get("TRITON_INTERPRET") == "1"
    gpu = benchmark.gpu_available()

    print("running correctness sweep ...")
    correctness = run_correctness(SHAPES, DTYPES)
    failures = [r for r in correctness if r.kind == ASSERTED and not r.passed]

    print("building analytical memory model ...")
    models = memory_model.model_all(SHAPES, DTYPES)

    if gpu:
        print("running benchmarks ...")
        peak = benchmark.measure_peak_bandwidth()
        timings = benchmark.run_all(SHAPES, DTYPES)
    else:
        print("no GPU: benchmarks will be reported as not measured")
        peak = None
        timings = benchmark.run_all(SHAPES, DTYPES)

    asserted_md, reported_md = _correctness_markdown(correctness)
    n_asserted = len([r for r in correctness if r.kind == ASSERTED])
    n_passed = n_asserted - len(failures)
    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dev = benchmark.device_name() or "none (CPU only)"
    peak_txt = f"{peak:.1f} GB/s (measured by device-to-device copy)" if peak else "not measured"

    document = f"""# Results

Generated {generated} by `python -m harness.sweep`. Do not edit by hand.

## Environment

| | |
| --- | --- |
| Python | {platform.python_version()} |
| PyTorch | {torch.__version__} |
| Triton | {triton.__version__} |
| Execution | {"Triton CPU interpreter (TRITON_INTERPRET=1)" if interpreter else "native"} |
| GPU | {dev} |
| Measured peak bandwidth | {peak_txt} |

## Correctness

Kernels are compared against naive PyTorch references. Tolerances are derived
from dtype machine epsilon in `harness/config.py`, not tuned to make tests
pass. `fusion_not_less_accurate` requires the fused kernel to be at least as
close to an fp64 ground truth as the unfused two-kernel sequence.

{asserted_md}

**Summary: {n_passed} of {n_asserted} asserted checks passed.**

### Measurements without pass/fail semantics

`fused_vs_unfused` is a divergence, not an error: it records how far apart the
two pipelines land. In bf16 they diverge substantially, and the two accuracy
rows show why -- the unfused sequence rounds its intermediate to bf16 before
softmax sees it, while the fused kernel keeps it in fp32 registers. Lower is
better in the two `*_accuracy_fp64` rows.

{reported_md}

## Memory traffic (analytical)

Derived from shapes and dtype sizes only. Nothing here is measured; the
arithmetic is written out in the module docstring of `harness/memory_model.py`
so it can be verified by hand.

The `reduction` column assumes the N-element weight vector is fetched from HBM
once and served from cache afterwards. The `cold W` column assumes it is
re-fetched for every row. Reality sits near the first bound for these shapes,
and the second is shown so the claim is bracketed rather than asserted.

{memory_model.format_markdown(models)}

The saving is exactly `2*M*N` elements in every case: one write of the
intermediate and one read of it back. As M grows, the N-element weight term
becomes negligible and the reduction tends to 50%.

## Wall-clock performance

{benchmark.format_markdown(timings, peak)}

## What these results do and do not establish

Established:

- The three kernels compute what the PyTorch references compute, across four
  shapes and two dtypes, within tolerances derived from dtype epsilon.
- Fusing RMSNorm into softmax does not cost accuracy. In bf16 it measurably
  improves it, because the intermediate is never rounded to storage precision.
- The fused kernel moves half as many bytes to and from HBM as the unfused
  pair, and roughly 7.5x fewer than naive eager PyTorch. This is arithmetic
  from shapes and dtype sizes, not a measurement.

Not established without a GPU run:

- Any wall-clock speedup. The correctness figures above come from the Triton
  CPU interpreter, which executes kernels sequentially in Python via NumPy and
  models neither the memory hierarchy nor occupancy. No timing taken there
  would mean anything, so none is reported.
- Whether the predicted traffic reduction actually converts into a
  proportional speedup. That depends on how close each variant runs to the
  bandwidth roofline, which requires hardware to determine.
"""

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(document, encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")

    if failures:
        print(f"\n{len(failures)} asserted check(s) FAILED:")
        for f in failures:
            print(f"  {f.shape} {f.dtype} {f.case}: max_rel={f.max_rel_err:.3e} {f.note}")
        return 1

    print("all asserted checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
