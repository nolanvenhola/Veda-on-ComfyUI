"""Explicit Intel compiler/kernel acceptance, separate from CUDA tests."""

import pytest
import torch

from veda_comfy.backends import base
from veda_comfy.backends.xpu_triton import XpuTritonBackend


@pytest.mark.skipif(not torch.xpu.is_available(), reason='needs Intel XPU')
@pytest.mark.parametrize('index', range(torch.xpu.device_count()))
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_intel_triton_selftest(index, dtype):
    base.self_test(XpuTritonBackend(), torch.device('xpu', index), dtype)
