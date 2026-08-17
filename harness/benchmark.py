"""Wall-clock benchmarks. GPU only, by design.

This module refuses to produce a timing number unless it is running on real
hardware. Under the Triton interpreter every kernel is executed sequentially
in Python on top of NumPy, so a measurement there would describe the
interpreter and not the kernel. When no GPU is present, every field comes
back as `None` and the reporting layer prints "not measured".

There is no estimation path. If it was not timed, it is not reported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from harness.config import DTYPE_NAMES, EPS, make_inputs
from harness.memory_model import model

# Median plus 20th/80th percentile. A single mean is misleading on a shared or
# thermally throttled GPU, which is exactly what a free Colab instance is.
QUANTILES = [0.5, 0.2, 0.8]


def gpu_available() -> bool:
    """True only when kernels will actually compile and run on a device."""
    if os.environ.get("TRITON_INTERPRET") == "1":
        return False
    return torch.cuda.is_available()


def device_name() -> str | None:
    if not gpu_available():
        return None
    return torch.cuda.get_device_name(0)


@dataclass(frozen=True)
class Timing:
    shape: tuple[int, int]
    dtype: str
    provider: str
    ms_median: float | None
    ms_p20: float | None
    ms_p80: float | None
    achieved_gbps: float | None

    @property
    def measured(self) -> bool:
        return self.ms_median is not None


def _not_measured(shape, dtype, provider) -> Timing:
    return Timing(shape, DTYPE_NAMES[dtype], provider, None, None, None, None)


def measure_peak_bandwidth(device: str = "cuda", n_bytes: int = 512 * 1024 * 1024) -> float | None:
    """Achievable device bandwidth from a large device-to-device copy.

    This is the denominator for any "percent of peak" claim. Using the
    datasheet figure instead would overstate the roofline, because no real
    kernel reaches the datasheet number.
    """
    if not gpu_available():
        return None
    import triton

    n_elements = n_bytes // 4
    src = torch.empty(n_elements, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)

    ms = triton.testing.do_bench(lambda: dst.copy_(src), return_mode="median")
    # A copy touches each byte twice: once read, once written.
    moved = 2 * n_elements * 4
    return moved * 1e-9 / (ms * 1e-3)


def _providers(x, weight):
    """Callables to time, keyed by name."""
    from kernels.fused_rmsnorm_softmax import fused_rmsnorm_softmax
    from kernels.rmsnorm import rmsnorm as rmsnorm_kernel
    from kernels.softmax import softmax as softmax_kernel
    from reference import torch_reference as ref

    providers = {
        "triton_fused": lambda: fused_rmsnorm_softmax(x, weight, EPS),
        "triton_unfused": lambda: softmax_kernel(rmsnorm_kernel(x, weight, EPS)),
        "torch_eager": lambda: ref.unfused_pipeline(x, weight, EPS),
    }

    # torch.compile fuses these operations itself and emits Triton, so it is
    # the honest hard baseline. Compilation is done once, outside the timed
    # region, via a warm-up call in run_case.
    try:
        compiled = torch.compile(ref.fused_pipeline, dynamic=False)
        providers["torch_compile"] = lambda: compiled(x, weight, EPS)
    except Exception:
        pass

    return providers


def run_case(shape: tuple[int, int], dtype: torch.dtype) -> list[Timing]:
    if not gpu_available():
        return [
            _not_measured(shape, dtype, p)
            for p in ("triton_fused", "triton_unfused", "torch_eager", "torch_compile")
        ]

    import triton

    x, weight = make_inputs(shape, dtype, device="cuda")
    traffic = model(shape, dtype)

    results: list[Timing] = []
    for name, fn in _providers(x, weight).items():
        try:
            fn()  # warm up: triggers JIT and torch.compile tracing
            torch.cuda.synchronize()
            median, p20, p80 = triton.testing.do_bench(fn, quantiles=QUANTILES)
        except Exception:
            results.append(_not_measured(shape, dtype, name))
            continue

        # Bandwidth is computed against the *analytical* traffic for that
        # variant, so the number answers "how close to the memory-traffic
        # bound did this get", not "how many bytes did the driver move".
        if name == "triton_fused":
            bytes_moved = traffic.fused_bytes
        elif name == "triton_unfused":
            bytes_moved = traffic.unfused_bytes
        else:
            bytes_moved = traffic.eager_bytes

        results.append(
            Timing(
                shape=shape,
                dtype=DTYPE_NAMES[dtype],
                provider=name,
                ms_median=median,
                ms_p20=p20,
                ms_p80=p80,
                achieved_gbps=bytes_moved * 1e-9 / (median * 1e-3),
            )
        )
    return results


def run_all(shapes, dtypes) -> list[Timing]:
    results: list[Timing] = []
    for dtype in dtypes:
        for shape in shapes:
            results.extend(run_case(shape, dtype))
    return results


def format_markdown(timings: list[Timing], peak_gbps: float | None) -> str:
    if not any(t.measured for t in timings):
        return (
            "**not measured** -- no GPU was available when this report was "
            "generated. Wall-clock timing is only collected on real hardware; "
            "see `notebooks/colab_benchmark.ipynb`."
        )

    lines = [
        "| shape | dtype | provider | median (ms) | p20-p80 (ms) | achieved GB/s | % of peak |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for t in timings:
        if not t.measured:
            lines.append(
                f"| {t.shape[0]}x{t.shape[1]} | {t.dtype} | {t.provider} | "
                "not measured | not measured | not measured | not measured |"
            )
            continue
        pct = f"{100.0 * t.achieved_gbps / peak_gbps:.1f}%" if peak_gbps else "not measured"
        lines.append(
            f"| {t.shape[0]}x{t.shape[1]} | {t.dtype} | {t.provider} | "
            f"{t.ms_median:.4f} | {t.ms_p20:.4f}-{t.ms_p80:.4f} | "
            f"{t.achieved_gbps:.1f} | {pct} |"
        )
    return "\n".join(lines)
