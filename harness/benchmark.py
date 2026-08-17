"""Wall-clock benchmarks. GPU only, by design.

This module refuses to produce a timing number unless it is running on real
hardware. Under the Triton interpreter every kernel is executed sequentially
in Python on top of NumPy, so a measurement there would describe the
interpreter and not the kernel. When no GPU is present, every field comes
back as `None` and the reporting layer prints "not measured".

There is no estimation path. If it was not timed, it is not reported.

A note on derived bandwidth
---------------------------
Bandwidth is bytes moved divided by time, and the numerator has to come from
somewhere. For the two Triton variants it comes from `harness.memory_model`,
which is exact: the kernels are ours, their loads and stores are visible in
the source, and the analytical count is what they actually issue.

That reasoning does not extend to the PyTorch providers. `torch.compile`
decides for itself how much to fuse, so attributing the naive eager byte count
to its runtime produces a fictitious bandwidth -- in an earlier version of this
file that yielded figures above 600% of measured peak, which is a good
reminder that a derived number is only as good as its assumptions. Bandwidth
is therefore reported for the Triton kernels only, and left blank elsewhere
rather than fabricated.
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

# Providers whose HBM traffic we know exactly, because we wrote the kernels.
_TRAFFIC_KNOWN = {"triton_fused", "triton_unfused"}


def gpu_available() -> bool:
    """True only when kernels will actually compile and run on a device."""
    if os.environ.get("TRITON_INTERPRET") == "1":
        return False
    return torch.cuda.is_available()


def device_name() -> str | None:
    if not gpu_available():
        return None
    return torch.cuda.get_device_name(0)


def l2_cache_bytes() -> int | None:
    """L2 capacity, which turns out to explain most of this project's results."""
    if not gpu_available():
        return None
    return getattr(torch.cuda.get_device_properties(0), "L2_cache_size", None)


def bf16_natively_supported() -> bool | None:
    """bf16 tensor-core support begins with Ampere (compute capability 8.0).

    On older hardware TorchInductor declines to compile bf16 and silently
    falls back, which makes `torch_compile` stop being a fused baseline. That
    has to be labelled or the comparison is misleading.
    """
    if not gpu_available():
        return None
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


@dataclass(frozen=True)
class Timing:
    shape: tuple[int, int]
    dtype: str
    provider: str
    ms_median: float | None
    ms_p20: float | None
    ms_p80: float | None
    achieved_gbps: float | None
    note: str = ""

    @property
    def measured(self) -> bool:
        return self.ms_median is not None


def _not_measured(shape, dtype, provider) -> Timing:
    return Timing(shape, DTYPE_NAMES[dtype], provider, None, None, None, None)


def measure_peak_bandwidth(device: str = "cuda", n_bytes: int = 512 * 1024 * 1024) -> float | None:
    """Achievable device bandwidth from a large device-to-device copy.

    This is the denominator for any "percent of peak" claim. Using the
    datasheet figure instead would overstate the roofline, because no real
    kernel reaches the datasheet number. The buffer is deliberately far larger
    than any L2 so the copy is genuinely HBM-bound.
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
    # the honest hard baseline. Compilation happens in the warm-up call.
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
    bf16_ok = bf16_natively_supported()

    results: list[Timing] = []
    for name, fn in _providers(x, weight).items():
        try:
            fn()  # warm up: triggers JIT and torch.compile tracing
            torch.cuda.synchronize()
            median, p20, p80 = triton.testing.do_bench(fn, quantiles=QUANTILES)
        except Exception:
            results.append(_not_measured(shape, dtype, name))
            continue

        if name in _TRAFFIC_KNOWN:
            bytes_moved = traffic.fused_bytes if name == "triton_fused" else traffic.unfused_bytes
            gbps = bytes_moved * 1e-9 / (median * 1e-3)
        else:
            # We do not know how much these actually move, so we do not claim to.
            gbps = None

        note = ""
        if name == "torch_compile" and dtype is torch.bfloat16 and bf16_ok is False:
            note = "Inductor skipped bf16 on this GPU; falls back to eager, not a fused baseline"

        results.append(
            Timing(
                shape=shape,
                dtype=DTYPE_NAMES[dtype],
                provider=name,
                ms_median=median,
                ms_p20=p20,
                ms_p80=p80,
                achieved_gbps=gbps,
                note=note,
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
        "| shape | dtype | provider | median (ms) | p20-p80 (ms) | GB/s (model-implied) | % of measured peak |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    notes: list[str] = []
    for t in timings:
        if not t.measured:
            lines.append(
                f"| {t.shape[0]}x{t.shape[1]} | {t.dtype} | {t.provider} | "
                "not measured | not measured | not measured | not measured |"
            )
            continue

        if t.achieved_gbps is None:
            gbps_txt, pct = "traffic not known", "traffic not known"
        else:
            gbps_txt = f"{t.achieved_gbps:.1f}"
            pct = f"{100.0 * t.achieved_gbps / peak_gbps:.1f}%" if peak_gbps else "not measured"

        marker = ""
        if t.note:
            notes.append(f"{t.shape[0]}x{t.shape[1]} {t.dtype} {t.provider}: {t.note}")
            marker = " [^]"

        lines.append(
            f"| {t.shape[0]}x{t.shape[1]} | {t.dtype} | {t.provider}{marker} | "
            f"{t.ms_median:.4f} | {t.ms_p20:.4f}-{t.ms_p80:.4f} | {gbps_txt} | {pct} |"
        )

    if notes:
        lines.append("")
        lines.append("Marked rows:")
        lines.append("")
        for n in sorted(set(notes)):
            lines.append(f"- {n}")

    return "\n".join(lines)


def format_prediction_markdown(
    timings: list[Timing], shapes, dtypes, l2_bytes: int | None
) -> str:
    """Predicted traffic reduction against the measured speedup.

    This is the table the whole project exists to produce. The prediction is
    `unfused_bytes / fused_bytes`, which tends to 2.00x. The measurement is the
    ratio of median runtimes. Where they disagree, the disagreement is the
    interesting part, not an embarrassment.
    """
    measured = {(t.shape, t.dtype, t.provider): t for t in timings}
    if not any(t.measured for t in timings):
        return "**not measured** -- requires a GPU run."

    l2_txt = f"{l2_bytes / (1024 * 1024):.0f} MiB" if l2_bytes else "unknown"

    lines = [
        f"L2 cache on this device: {l2_txt}. The intermediate column is the size of the "
        "tensor that fusion avoids round-tripping; compare it against L2.",
        "",
        "| shape | dtype | intermediate | fits in L2 | predicted | measured | measured/predicted |",
        "| --- | --- | ---: | :--: | ---: | ---: | ---: |",
    ]

    for dtype in dtypes:
        name = DTYPE_NAMES[dtype]
        for shape in shapes:
            fused = measured.get((shape, name, "triton_fused"))
            unfused = measured.get((shape, name, "triton_unfused"))
            if not (fused and unfused and fused.measured and unfused.measured):
                continue

            traffic = model(shape, dtype)
            predicted = traffic.unfused_bytes / traffic.fused_bytes
            actual = unfused.ms_median / fused.ms_median

            intermediate = shape[0] * shape[1] * traffic.element_bytes
            fits = "yes" if l2_bytes and intermediate <= l2_bytes else "no"
            if l2_bytes is None:
                fits = "?"

            lines.append(
                f"| {shape[0]}x{shape[1]} | {name} | "
                f"{intermediate / (1024 * 1024):.2f} MiB | {fits} | "
                f"{predicted:.2f}x | {actual:.2f}x | {100.0 * actual / predicted:.0f}% |"
            )

    return "\n".join(lines)
