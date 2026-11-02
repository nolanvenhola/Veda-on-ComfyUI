"""Experimental Intel XPU gather + PyTorch SDPA backend.

Only selected key/value blocks are materialised. The memory budget includes
both gathered tensors and a conservative fp32 SDPA score workspace. Unlike
a full-length boolean attention mask, this reduces the attention problem.
No compiler, CUDA shim, host tensor round-trip, or dependency change is used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from . import base
from ..core import tiling

DEFAULT_CHUNK_BYTES = 128 * 2**20


class XpuGatherBackend(base.Backend):
    """Bounded-memory block-sparse attention using device-native SDPA."""

    name = 'xpu-gather'
    display = 'Intel XPU gather + SDPA (experimental)'
    dtypes = (torch.bfloat16, torch.float16, torch.float32)
    tolerance = 0.02

    def __init__(self, chunk_bytes: int = DEFAULT_CHUNK_BYTES):
        if chunk_bytes <= 0:
            raise ValueError('chunk_bytes must be positive')
        self.chunk_bytes = chunk_bytes

    @torch.no_grad()
    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               block_mask: torch.Tensor,
               layout: tiling.TileLayout) -> torch.Tensor:
        tile = tiling.TILE_SIZE
        tiles, heads, dim = layout.n_tiles, q.shape[1], q.shape[2]
        # Head-major indexing keeps each gathered block contiguous. No data
        # crosses to the CPU; only scalar allocation bounds are read back.
        qt, kt, vt = (x.reshape(tiles, tile, heads, dim)
                      .permute(2, 0, 1, 3).contiguous() for x in (q, k, v))
        k_flat = kt.reshape(heads * tiles, tile, dim)
        v_flat = vt.reshape(heads * tiles, tile, dim)
        valid = layout.slot_valid.reshape(tiles, tile).bool()
        out = torch.zeros_like(qt)
        # Global query rows are dense by definition; separating them avoids
        # padding every sparse video row up to the full sequence length.
        for first, last in ((0, layout.n_video_tiles),
                            (layout.n_video_tiles, tiles)):
            if first == last:
                continue
            selected = (block_mask[:, first:last] & layout.kv_ok)
            counts = selected.sum(-1)
            kept = int(counts.max().item())
            if kept == 0:
                continue
            order = torch.argsort((~selected).to(torch.int8), dim=-1,
                                  stable=True)[..., :kept]
            per_group = kept * tile * (2 * dim * q.element_size()
                                       + tile * 4 + 1)
            groups = max(1, self.chunk_bytes // per_group)
            row_count = last - first
            flat_order = order.reshape(-1, kept)
            flat_counts = counts.reshape(-1)
            for begin in range(0, heads * row_count, groups):
                end = min(heads * row_count, begin + groups)
                group = torch.arange(begin, end, device=q.device)
                head = group // row_count
                row = group % row_count + first
                picked = flat_order[begin:end]
                index = (picked + head[:, None] * tiles).reshape(-1)
                keys, values = (x.index_select(0, index).reshape(
                    end - begin, 1, kept * tile, dim)
                    for x in (k_flat, v_flat))
                key_ok = valid[picked] & (torch.arange(
                    kept, device=q.device)[None, :, None]
                    < flat_counts[begin:end, None, None])
                query = qt[head, row].unsqueeze(1)
                result = functional.scaled_dot_product_attention(
                    query, keys, values,
                    attn_mask=key_ok.reshape(end - begin, 1, 1, -1),
                    dropout_p=0.0, is_causal=False)
                out[head, row] = result.squeeze(1)
        return out.permute(1, 2, 0, 3).reshape_as(q)


def create(info) -> base.Backend:
    """Creates the experimental backend only for an available Intel XPU."""
    if info.kind != 'xpu' or not torch.xpu.is_available():
        raise base.BackendUnavailable(
            'Intel XPU attention needs a working PyTorch XPU runtime. '
            '\u8bf7\u5b89\u88c5 PyTorch XPU \u8fd0\u884c\u65f6\u3002')
    return XpuGatherBackend()
