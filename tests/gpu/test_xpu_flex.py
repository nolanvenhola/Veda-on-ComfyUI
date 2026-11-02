"""Intel FlexAttention correctness on all visible GPUs and both dtypes."""

import pytest
import torch

from veda_comfy.backends import base
from veda_comfy.backends.xpu_flex import XpuFlexBackend
from veda_comfy.core import reference


@pytest.mark.skipif(not torch.xpu.is_available(), reason='needs Intel XPU')
@pytest.mark.parametrize('index', range(torch.xpu.device_count()))
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_intel_flex_mask_padding_and_empty_rows(index, dtype):
    device = torch.device('xpu', index)
    backend = XpuFlexBackend()
    base.self_test(backend, device, dtype)
    q, k, v, mask, layout = base.selftest_problem(device, dtype)
    mask[:, 0] = False
    got = backend.attend(q, k, v, mask, layout)
    want = reference.block_sparse_attention(q, k, v, mask, layout)
    live = layout.slot_valid.bool()
    torch.testing.assert_close(got[live].float(), want[live].float(),
                               atol=0.03, rtol=0.02)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[:128]) == 0


@pytest.mark.skipif(not torch.xpu.is_available(), reason='needs Intel XPU')
def test_varying_h3_head_groups_do_not_exhaust_compile_cache():
    # The real predictor partitions different head counts on every layer.
    # Static compilation looked fast on a fixed shape but fell back in H3.
    import torch._dynamo
    from veda_comfy.backends import xpu_flex
    torch._dynamo.reset()
    xpu_flex._compiled.cache_clear()
    device = torch.device('xpu', 0)
    q, k, v, mask, layout = base.selftest_problem(device, torch.bfloat16)
    backend = XpuFlexBackend()
    try:
        with torch._dynamo.config.patch(recompile_limit=4):
            for heads in (1, 2, 3, 4, 6, 7, 8, 9, 11, 14, 16):
                tensors = [x[:, :1].expand(-1, heads, -1).contiguous()
                           for x in (q, k, v)]
                selected = mask[:1].expand(heads, -1, -1).contiguous()
                got = backend.attend(*tensors, selected, layout)
                want = reference.block_sparse_attention(
                    *tensors, selected, layout)
                live = layout.slot_valid.bool()
                torch.testing.assert_close(
                    got[live].float(), want[live].float(),
                    atol=0.03, rtol=0.02)
    finally:
        torch._dynamo.reset()
        xpu_flex._compiled.cache_clear()
