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
