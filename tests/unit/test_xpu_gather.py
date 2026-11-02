"""The XPU algorithm also runs on CPU for exact mask regression checks."""

import dataclasses

import pytest
import torch

from veda_comfy import backends
from veda_comfy.backends import base
from veda_comfy.backends.xpu_gather import XpuGatherBackend
from veda_comfy.core import reference


@pytest.mark.parametrize('budget', [1, 128 * 2**20])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_sparse_padding_globals_and_chunk_boundaries(budget, dtype):
    backend = XpuGatherBackend(budget)
    base.self_test(backend, torch.device('cpu'), dtype)
    q, k, v, mask, layout = base.selftest_problem(torch.device('cpu'), dtype)
    # Empty query rows must return finite zeros rather than NaNs.
    mask[:, 0] = False
    actual = backend.attend(q, k, v, mask, layout)
    expected = reference.block_sparse_attention(q, k, v, mask, layout)
    live = layout.slot_valid.bool()
    torch.testing.assert_close(actual[live].float(), expected[live].float(),
                               atol=0.03, rtol=0.02)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:128]) == 0


def test_xpu_registered_without_changing_other_devices():
    from veda_comfy import hardware
    info = hardware.DeviceInfo('xpu', 0, 'Intel Arc', None, 'xpu',
                               'intel-arc', 'windows', 'amd64',
                               32 * 2**30, False, None)
    assert backends.candidates(info) == ['xpu-flex']
    assert backends.candidates(dataclasses.replace(info, kind='cpu')) == []


def test_invalid_chunk_budget():
    with pytest.raises(ValueError, match='positive'):
        XpuGatherBackend(0)
