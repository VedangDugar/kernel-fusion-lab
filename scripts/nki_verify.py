"""Validate the NKI fused kernel against the same references as the Triton one.

Runs on the NKI CPU simulator. No Trainium hardware is involved, and no
timing is produced or implied -- the simulator establishes numerics only.

    docker run --rm -e PYTHONPATH=/work -v "$PWD:/work" -w /work kfl:nki \
        python scripts/nki_verify.py
"""

from __future__ import annotations

import numpy as np
import nki

from kernels_nki.fused_rmsnorm_softmax import fused_rmsnorm_softmax

SHAPES = [(128, 512), (256, 1024), (512, 2048), (1024, 4096)]
EPS = 1e-6


def reference(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Naive NumPy ground truth, computed in float64."""
    x64 = x.astype(np.float64)
    w64 = w.astype(np.float64)
    mean_square = (x64 ** 2).mean(axis=-1, keepdims=True)
    h = x64 / np.sqrt(mean_square + eps) * w64
    shifted = h - h.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def main() -> int:
    rng = np.random.default_rng(0)
    header = f"{'shape':>12} {'max abs err':>13} {'max rel err':>13}  result"
    print(header)
    print("-" * len(header))

    failures = 0
    for shape in SHAPES:
        rows, cols = shape
        x = rng.standard_normal(shape, dtype=np.float32)
        # (1, N), not (N,): every NKI tile has a partition axis.
        w = rng.standard_normal((1, cols), dtype=np.float32)

        got = np.asarray(nki.simulate(fused_rmsnorm_softmax)(x, w, EPS), dtype=np.float64)
        want = reference(x, w, EPS)

        abs_err = np.abs(got - want)
        max_abs = abs_err.max()
        max_rel = (abs_err / np.abs(want)).max()

        # fp32 accumulation over up to 4096 terms; the Triton harness uses the
        # same 1e-5 relative bound for fp32 and this holds it to the same one.
        ok = max_rel <= 1e-5
        failures += not ok
        print(f"{str(shape):>12} {max_abs:>13.3e} {max_rel:>13.3e}  {'PASS' if ok else 'FAIL'}")

    print()
    if failures:
        print(f"{failures} shape(s) FAILED")
        return 1
    print("all shapes passed on the NKI CPU simulator")
    print("note: numerics only. No Trainium hardware was used and no timing is claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
