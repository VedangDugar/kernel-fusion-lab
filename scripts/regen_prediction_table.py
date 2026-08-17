"""Rebuild the predicted-versus-measured table from a recorded GPU run.

The prediction table depends only on the fused and unfused median runtimes, so
it can be regenerated off-GPU from numbers captured in an earlier run. This
exists so a reporting fix does not require burning another Colab session.
"""

import torch

from harness.benchmark import Timing, format_prediction_markdown
from harness.config import DTYPES, SHAPES

T4_L2_BYTES = 4 * 1024 * 1024

# medians in ms, from the Tesla T4 run recorded in docs/RESULTS.md
RECORDED = {
    ((128, 512), "float32"): (0.0092, 0.0138),
    ((256, 1024), "float32"): (0.0131, 0.0160),
    ((512, 2048), "float32"): (0.0410, 0.0752),
    ((1024, 4096), "float32"): (0.1507, 0.2955),
    ((128, 512), "bfloat16"): (0.0060, 0.0079),
    ((256, 1024), "bfloat16"): (0.0090, 0.0119),
    ((512, 2048), "bfloat16"): (0.0226, 0.0301),
    ((1024, 4096), "bfloat16"): (0.0779, 0.1490),
}


def main() -> None:
    timings = []
    for (shape, name), (fused_ms, unfused_ms) in RECORDED.items():
        for provider, ms in (("triton_fused", fused_ms), ("triton_unfused", unfused_ms)):
            timings.append(Timing(shape, name, provider, ms, ms, ms, None))

    print(format_prediction_markdown(timings, SHAPES, DTYPES, T4_L2_BYTES))


if __name__ == "__main__":
    main()
