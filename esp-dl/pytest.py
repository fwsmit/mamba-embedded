import pytest


@pytest.mark.timeout(120)
def test_mamba_inference(dut):

    dut.expect_exact("=== PROGRAM COMPLETE ===", timeout=120)
