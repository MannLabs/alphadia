import numpy as np
import pandas as pd
import pytest
import torch

from alphadia.fdr.utils import (
    SCORE_TO_NEXT_CANDIDATE,
    add_score_to_next_candidate,
    manage_torch_threads,
)


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


def test_add_score_to_next_candidate_compares_best_with_runner_up_and_others_with_best():
    # given
    df = pd.DataFrame(
        {"precursor_idx": [7, 3, 7, 7, 3], "score": [5.0, 2.0, 9.0, 1.0, 6.0]}
    )

    # when
    add_score_to_next_candidate(df)

    # then
    np.testing.assert_allclose(
        df[SCORE_TO_NEXT_CANDIDATE].to_numpy(), [-4.0, -4.0, 4.0, -8.0, 4.0]
    )


def test_add_score_to_next_candidate_single_candidate_uses_lowest_score_of_frame():
    # given
    df = pd.DataFrame({"precursor_idx": [1, 2, 2], "score": [10.0, 4.0, 3.0]})

    # when
    add_score_to_next_candidate(df)

    # then
    np.testing.assert_allclose(df[SCORE_TO_NEXT_CANDIDATE].to_numpy(), [7.0, 1.0, -1.0])


def test_add_score_to_next_candidate_tied_best_candidates_get_zero():
    # given
    df = pd.DataFrame({"precursor_idx": [4, 4, 4], "score": [6.0, 6.0, 2.0]})

    # when
    add_score_to_next_candidate(df)

    # then
    np.testing.assert_allclose(df[SCORE_TO_NEXT_CANDIDATE].to_numpy(), [0.0, 0.0, -4.0])


def test_add_score_to_next_candidate_empty_frame():
    # given
    df = pd.DataFrame({"precursor_idx": np.array([], dtype=int), "score": []})

    # when
    add_score_to_next_candidate(df)

    # then
    assert SCORE_TO_NEXT_CANDIDATE in df
    assert len(df) == 0
