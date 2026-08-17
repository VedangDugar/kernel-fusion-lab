"""Smoke test for the NKI toolchain, mirroring scripts/hello_kernel.py.

Proves that a trivial NKI kernel compiles and runs on the CPU simulator
before any real kernel is attempted.
"""

import numpy as np
import nki
import nki.language as nl


@nki.jit
def add_kernel(a, b):
    # Outputs are declared as HBM tensors by the kernel itself and returned.
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    ta = nl.load(a)   # HBM -> SBUF
    tb = nl.load(b)
    # NkiTensor does not overload Python arithmetic operators; every operation
    # is an explicit nl.* call that maps to a hardware instruction.
    nl.store(out, nl.add(ta, tb))  # SBUF -> HBM
    return out


def main() -> int:
    print("nki", nki.__version__)
    print("pmax (SBUF partitions):", nl.tile_size.pmax)

    rng = np.random.default_rng(0)
    # Partition dim must not exceed pmax = 128.
    a = rng.standard_normal((128, 512), dtype=np.float32)
    b = rng.standard_normal((128, 512), dtype=np.float32)

    got = nki.simulate(add_kernel)(a, b)
    want = a + b
    err = np.abs(got - want).max()

    print("max abs err:", err)
    if err != 0.0:
        print("FAIL")
        return 1
    print("PASS: NKI kernel ran on the CPU simulator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
