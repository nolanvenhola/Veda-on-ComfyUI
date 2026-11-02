"""Experimental fused INT8 sparse attention for Intel Triton."""

from __future__ import annotations

import functools

import torch

from . import base
from ..core import selection


@functools.cache
def _kernel():
    from ..kernels.xpu import sparse_int8
    return sparse_int8


class XpuTritonBackend(base.Backend):
    """Intel DPAS candidate, validated by the standard backend self-test."""

    name = 'xpu-triton-int8'
    display = 'Intel XPU Triton INT8 (experimental)'
    dtypes = (torch.bfloat16, torch.float16)
    tolerance = 0.05

    @torch.no_grad()
    def attend(self, q, k, v, block_mask, layout):
        index, count = selection.tile_index_list(block_mask & layout.kv_ok)
        with torch.xpu.device(q.device):
            return _kernel().attend(q, k, v, index, count,
                                    layout.valid_count)

    def warmup_note(self) -> str:
        return 'compiling Intel Triton kernels (first run only)'


def create(info) -> base.Backend:
    """Checks XPU identity without silently falling back to dense attention."""
    if info.kind != 'xpu' or not torch.xpu.is_available():
        raise base.BackendUnavailable('Intel XPU runtime is required. '
                                     '\u9700\u8981 Intel XPU '
                                     '\u8fd0\u884c\u65f6\u3002')
    try:
        _kernel()
    except ImportError as error:
        raise base.MissingDependency(
            'Intel-compatible Triton is required in the experimental '
            'environment. / \u9700\u8981 Intel Triton\u3002') from error
    return XpuTritonBackend()
