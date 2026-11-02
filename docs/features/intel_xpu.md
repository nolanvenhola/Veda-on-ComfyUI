# Experimental Intel XPU backend

The fork isolates Intel work from the existing CUDA, ROCm and Apple paths.
No Intel kernel imports another backend or changes the CUDA kernel source.
Weights, predictor selection and tile-selection semantics are unchanged.

## Two measured candidates

`xpu-gather` gathers only selected K/V blocks and calls PyTorch XPU SDPA.
It is a correctness baseline, not an assumed performance improvement.
It bounds batches using K/V bytes plus conservative fp32 score workspace;
global query tiles remain dense, and padding/empty rows are masked.

`xpu-triton-int8` ports the independent INT8 Q/K sparse loop to Intel Triton.
It removes CUDA capability checks, TMA and CUDA allocation hooks. Compiler
and GPU numerical tests are required before making it the selected path.
INT8 numerical tolerance is 5%, inherited from Veda's quantized arithmetic
contract, and never applies to tile/mask/index semantics.

## Environment

Initial target: Windows 11, Arc Pro B70 32 GB, PyTorch 2.15.0+xpu,
Intel-enabled Triton 3.9.0. Use an isolated tooling environment; do not
replace the installed PyTorch or install generic CUDA Triton dependencies.
The fused candidate requires MSVC x64 Build Tools, Windows SDK, and matching
SYCL/Level Zero headers and libraries. Load the compiler environment only
for the experiment process. A missing compiler is an explicit failure.

## Acceptance and benchmark

Run `pytest tests/unit`, `ruff check .`, then `pytest tests/gpu` on Intel.
Run `python tools/bench_attention.py --device xpu:0 --latent-t 37 --repeat 3`
and `--latent-t 102`, with `--backend xpu-triton-int8` for the fused candidate.
XPU measurements synchronize the device; warmup is excluded. The synthetic
benchmark includes Veda layout, prediction, selection and gather/scatter,
not model loading, denoising MLPs, VAE decoding or audio/video encoding.
Random predictor weights measure cost only, never output quality.

On November 2, 2026, the gather candidate passed backend correctness on
both B70s. Its five-second-shaped problem (38,228 tokens) took 651.8 ms
against 522.5 ms dense, and its fourteen-second-shaped problem (104,484
tokens) took 4522.7 ms against 4136.9 ms dense (two measured repeats).
These results do not justify enabling it in production.

Final acceptance requires matched dense/Veda videos using the correct
T2VA/FL2VA or R2VA predictor, same prompt/seed/steps/resolution, and human
review of motion, identity and audio. Neither benchmark establishes that.

## Selected implementation: XPU FlexAttention

`xpu-flex` reuses Veda's learned block choices with PyTorch's Intel
Inductor/FlexAttention sparse kernel. Unlike the gathered baseline, selected
blocks are consumed in the compiled attention operation. It uses bf16/fp16
and the existing 2% numerical tolerance, with an explicit per-call XPU device
guard. It is the only automatic Intel candidate; there is no fallback chain.
The other two candidates remain explicit benchmark controls only.

On November 2, 2026, synchronized warm measurements on one idle B70:

| Synthetic workload | Dense SDPA | Veda XPU Flex | Attention speedup |
|---|---:|---:|---:|
| 38,228 tokens, about 5 seconds at 1 MP | 525.3 ms | 221.2 ms | 2.38x |
| 104,484 tokens, about 14 seconds at 1 MP | 4116.3 ms | 1027.3 ms | 4.01x |

Three repeats; 56 heads, head dimension 128, 90% configured video sparsity.
Actual computed attention fractions are 15.9% and 13.6%, because text/audio
global rows remain dense. First sparse calls, including compilation, took
6.5 and 12.3 seconds respectively. This is not end-to-end video speed.

The direct Intel Triton INT8 port passed its initial self-test but measured
1190.2/6695.0 ms at these shapes, so it is not the automatic Intel path.
Cross-device launch tests require `torch.xpu.device(q.device)`; without
that guard the Intel launcher can crash when switching devices in-process.
Use `warp_size`, not CUDA-oriented `threads_per_warp`, with Intel Triton 3.9.

PyTorch's compiler must run from an x64 MSVC developer environment on this
Windows build. Production ComfyUI workers have not been changed by these
standalone tests. Keep the plugin opt-in until matched video review.

Acceptance run: 162 tests passed, 3 skipped; Ruff passed. GPU acceptance covered both B70 devices, bf16/fp16, mask semantics, padding, empty rows and cross-device launches. The skipped checks were CUDA-only arithmetic comparisons and an unrelated optional path. Human video quality review is still pending.

The first real H3 attempt exposed a fixed-shape compiler-cache problem:
head-group sizes change by layer. `torch.compile(dynamic=True,
fullgraph=True)` is required. The backend pads each head batch to eight heads because Intel
FlexAttention still specialises heads under dynamic compilation; fullgraph makes cache/graph failures explicit instead of
silently falling back to an unusably slow eager path. A GPU regression
varies eleven head counts under a four-graph recompilation limit.

Padded heads have zero selected blocks, are discarded after attention, and never modify real-head masks or output values. Dynamic compilation handles sequence geometry; fullgraph forbids a silent eager fallback.
