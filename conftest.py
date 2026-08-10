"""Test-suite-wide pytest fixtures.

The suite builds a bge-m3 in-memory index in a per-file fixture for every search
test file. Previously those fixtures were ``scope="session"``, which keeps every
model alive until the whole pytest session ends — on a 16 GiB Apple Silicon Mac
the cumulative MPS GPU allocations (~2 GiB × 12 files) exceeded the default
20 GiB high-watermark and raised ``RuntimeError: MPS backend out of memory`` deep
into the run. The fixtures are now module-scoped (torn down at each file's end);
this fixture additionally flushes the PyTorch MPS cache and runs GC after every
test so freed model memory is returned to the pool promptly.
"""

import gc

import pytest


@pytest.fixture(autouse=True)
def _flush_mps_cache():
    """Flush PyTorch MPS cached allocations and collect garbage after each test."""
    yield
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()
