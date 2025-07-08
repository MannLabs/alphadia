import pytest
import torch

from alphadia.fdr.utils import manage_torch_threads


@manage_torch_threads(max_threads=2)
def sample_decorated_function():
    return torch.get_num_threads()


@manage_torch_threads(max_threads=2)
def failing_decorated_function():
    raise ValueError("Test exception")


def test_manage_torch_threads():
    # Given
    original_threads = torch.get_num_threads()
    torch.set_num_threads(4)  # Set to higher number to test reduction

    # When
    threads_during_execution = sample_decorated_function()
    threads_after_execution = torch.get_num_threads()

    # Then
    assert threads_during_execution == 2  # Should be limited to 2 during execution
    assert threads_after_execution == 4  # Should be restored to original value

    # Cleanup
    torch.set_num_threads(original_threads)


def test_manage_torch_threads_with_exception():
    # Given
    original_threads = torch.get_num_threads()
    torch.set_num_threads(4)  # Set to higher number to test reduction

    # When/Then
    with pytest.raises(ValueError, match="Test exception"):
        failing_decorated_function()

    # Then
    assert (
        torch.get_num_threads() == 4
    )  # Verify threads were restored even after exception

    # Cleanup
    torch.set_num_threads(original_threads)
