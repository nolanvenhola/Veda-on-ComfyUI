"""Intel experiment using PyTorch's compiled block-sparse FlexAttention."""

from __future__ import annotations

import functools

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention

from . import base
from ..core import tiling


@functools.cache
def _compiled():
    return torch.compile(flex_attention, dynamic=False)


class XpuFlexBackend(base.Backend):
    """Uses Veda block choices with Intel Inductor's sparse kernel."""

    name = 'xpu-flex'
    display = 'Intel XPU FlexAttention (experimental)'
    dtypes = (torch.bfloat16, torch.float16)
    tolerance = 0.02

    @torch.no_grad()
    def attend(self, q, k, v, block_mask, layout):
        selected = block_mask & layout.kv_ok
        counts = selected.sum(-1, dtype=torch.int32)[None].contiguous()
        order = torch.argsort((~selected).to(torch.int8), dim=-1,
                              stable=True)[None].to(torch.int32)
        valid = layout.slot_valid.bool()

        def mask_mod(batch, head, query, key):
            return valid[key]

        sparse = BlockMask.from_kv_blocks(
            counts, order, BLOCK_SIZE=tiling.TILE_SIZE, mask_mod=mask_mod,
            seq_lengths=(q.shape[0], k.shape[0]), compute_q_blocks=False)
        with torch.xpu.device(q.device):
            result = _compiled()(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                v.transpose(0, 1)[None], block_mask=sparse)
        return result[0].transpose(0, 1).contiguous()


def create(info) -> base.Backend:
    """Creates a compiler-dependent XPU backend with no fallback."""
    if info.kind != 'xpu' or not torch.xpu.is_available():
        raise base.BackendUnavailable('Intel PyTorch XPU is required. '
                                     '\u9700\u8981 Intel PyTorch '
                                     'XPU\u3002')
    return XpuFlexBackend()
