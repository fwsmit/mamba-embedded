import pytest


@pytest.mark.timeout(120)
def test_cat_detect(dut):

    dut.expect_exact("=== PROGRAM COMPLETE ===", timeout=120)
