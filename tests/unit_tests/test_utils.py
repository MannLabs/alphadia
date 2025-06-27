#!python -m unittest tests.test_utils
"""This module provides unit tests for alphadia.cli."""

import pytest

from alphadia.utils import (
    get_torch_device,
)


@pytest.mark.parametrize("use_gpu", [True, False])
def test_get_torch_device(use_gpu):
    # given

    # when
    device = get_torch_device(use_gpu)

    # then
    assert device in ["gpu", "mps", "cpu"]
