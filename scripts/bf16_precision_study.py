"""Why the fused and unfused bf16 pipelines disagree by ~8%.

Running this project's correctness suite for the first time produced a bf16
disagreement between the fused kernel and the unfused two-kernel sequence far
larger than output rounding could explain. This script establishes the cause
rather than widening the tolerance until the failure disappeared.

Claim under test
----------------
The unfused sequence materialises its intermediate H = rmsnorm(X, W) in
memory, in the storage dtype. In bf16 that rounds H by up to one ULP at H's
own magnitude. Softmax converts an absolute perturbation of its logits into
approximately the same *relative* perturbation of its outputs, so the two
pipelines should diverge by roughly

    max|H| * 2^-8

The fused kernel never writes H, so it never pays this cost, and should come
out closer to a high-precision ground truth.

Run:
    docker run --rm -e TRITON_INTERPRET=1 -e PYTHONPATH=/work \
        -v "$PWD:/work" -w /work kfl:cpu python scripts/bf16_precision_study.py
"""

import torch

from harness.config import EPS, SHAPES, make_inputs
from kernels.fused_rmsnorm_softmax import fused_rmsnorm_softmax
from kernels.rmsnorm import rmsnorm as rmsnorm_kernel
from kernels.softmax import softmax as softmax_kernel
from reference import torch_reference as ref

BF16_EPS = 2.0 ** -8


def study(shape, dtype):
    x, w = make_inputs(shape, dtype)

    # The intermediate, computed exactly and as the unfused path stores it.
    x32, w32 = x.float(), w.float()
    mean_square = x32.pow(2).mean(-1, keepdim=True)
    h_exact = x32 * torch.rsqrt(mean_square + EPS) * w32
    h_stored = rmsnorm_kernel(x, w, EPS).float()

    max_h = h_exact.abs().max().item()
    intermediate_round_err = (h_exact - h_stored).abs().max().item()

    fused = fused_rmsnorm_softmax(x, w, EPS)
    unfused = softmax_kernel(rmsnorm_kernel(x, w, EPS))
    truth = ref.ground_truth(x, w, EPS)

    def max_rel(a, b):
        a64, b64 = a.to(torch.float64), b.to(torch.float64)
        sig = b64.abs() > 1e-30
        return ((a64[sig] - b64[sig]).abs() / b64[sig].abs()).max().item()

    return {
        "max_h": max_h,
        "intermediate_round_err": intermediate_round_err,
        "predicted": max_h * BF16_EPS,
        "observed": max_rel(fused, unfused),
        "fused_err": max_rel(fused, truth),
        "unfused_err": max_rel(unfused, truth),
    }


def main():
    for dtype in (torch.float32, torch.bfloat16):
        name = "float32" if dtype is torch.float32 else "bfloat16"
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        header = (
            f"{'shape':>12} {'max|H|':>9} {'H round err':>12} "
            f"{'predicted':>11} {'observed':>11} {'fused err':>11} {'unfused err':>12}"
        )
        print(header)
        print("-" * len(header))
        for shape in SHAPES:
            s = study(shape, dtype)
            print(
                f"{str(shape):>12} {s['max_h']:>9.3f} {s['intermediate_round_err']:>12.3e} "
                f"{s['predicted']:>11.3e} {s['observed']:>11.3e} "
                f"{s['fused_err']:>11.3e} {s['unfused_err']:>12.3e}"
            )

    print(
        "\nReading the table: 'predicted' and 'observed' agree to within a small\n"
        "factor in bf16, confirming that the divergence is intermediate rounding\n"
        "rather than a kernel defect. 'fused err' below 'unfused err' shows the\n"
        "fused kernel is the more accurate of the two, not merely a different one.\n"
        "In fp32 the effect vanishes, as it should: fp32 rounding of H is far\n"
        "below the tolerance."
    )


if __name__ == "__main__":
    main()
