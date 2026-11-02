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
    return torch.compile(flex_attention, dynamic=True, fullgraph=True)


class XpuFlexBackend(base.Backend):
    """Uses Veda block choices with Intel Inductor's sparse kernel."""

    name = 'xpu-flex'
    display = 'Intel XPU FlexAttention (experimental)'
    dtypes = (torch.bfloat16, torch.float16)
    tolerance = 0.02

    @torch.no_grad()
    def attend(self, q, k, v, block_mask, layout):
        # Intel FlexAttention specialises the head dimension even when
        # dynamic=True. H3 changes group sizes per layer: normalise to
        # eight heads so a run cannot exhaust the compiler cache.
        batch_heads = 8
        valid = layout.slot_valid.bool()

        def mask_mod(batch, head, query, key):
            return valid[key]

        pieces = []
        for first in range(0, q.shape[1], batch_heads):
            last = min(q.shape[1], first + batch_heads)
            real_heads = last - first
            selected = block_mask[first:last] & layout.kv_ok
            counts = selected.sum(-1, dtype=torch.int32)
            order = torch.argsort((~selected).to(torch.int8), dim=-1,
                                  stable=True).to(torch.int32)
            tensors = [x[:, first:last].transpose(0, 1)[None].contiguous()
                       for x in (q, k, v)]
            if real_heads < batch_heads:
                missing = batch_heads - real_heads
                counts = torch.cat((counts, counts.new_zeros(
                    missing, counts.shape[1])), dim=0)
                order = torch.cat((order, order.new_zeros(
                    missing, order.shape[1], order.shape[2])), dim=0)
                tensors = [torch.cat((x, x.new_zeros(
                    1, missing, x.shape[2], x.shape[3])), dim=1)
                           for x in tensors]
            sparse = BlockMask.from_kv_blocks(
                counts[None].contiguous(), order[None].contiguous(),
                BLOCK_SIZE=tiling.TILE_SIZE, mask_mod=mask_mod,
                seq_lengths=(q.shape[0], k.shape[0]), compute_q_blocks=False)
            with torch.xpu.device(q.device):
                result = _compiled()(*tensors, block_mask=sparse)
            pieces.append(result[0, :real_heads].transpose(0, 1))
        return torch.cat(pieces, dim=1).contiguous()


def create(info) -> base.Backend:
    """Creates a compiler-dependent XPU backend with no fallback."""
    if info.kind != 'xpu' or not torch.xpu.is_available():
        raise base.BackendUnavailable('Intel PyTorch XPU is required. '
                                     '\u9700\u8981 Intel PyTorch '
                                     'XPU\u3002')
    return XpuFlexBackend()
