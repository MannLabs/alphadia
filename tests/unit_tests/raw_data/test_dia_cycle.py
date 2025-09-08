from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphadia.exceptions import NotValidDiaDataError
from alphadia.raw_data.dia_cycle import (
    _get_cycle_length,
    _get_cycle_start,
    _is_valid_cycle,
    _normed_auto_correlation,
    determine_dia_cycle,
)


def test_normed_auto_correlation():
    # given
    x = np.array([1, 2, 3, 4, 5])

    # when
    result = _normed_auto_correlation(x)

    # then
    expected = np.array([1.0, 0.4, -0.1, -0.4, -0.4])
    np.testing.assert_almost_equal(result, expected, decimal=10)


def test_normed_auto_correlation_constant():
    # given
    x = np.array([2, 2, 2, 2, 2])

    # when
    result = _normed_auto_correlation(x)

    # then
    assert np.isnan(result).all()


def test_normed_auto_correlation_periodic():
    # given
    x = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])

    # when
    result = _normed_auto_correlation(x)

    # then
    expected = np.array(
        [
            1.0,
            -0.3333333333,
            -0.5,
            0.6666666667,
            -0.1666666667,
            -0.3333333333,
            0.3333333333,
            0.0,
            -0.1666666667,
        ]
    )
    np.testing.assert_almost_equal(result, expected, decimal=10)


def test_cycle():
    # given
    rand_cycle_start = np.random.randint(0, 100)
    rand_cycle_length = np.random.randint(5, 10)
    rand_num_cycles = np.random.randint(10, 50)

    cycle = np.zeros(rand_cycle_start)
    cycle = np.append(cycle, np.tile(np.arange(rand_cycle_length), rand_num_cycles))

    # when
    cycle_length = _get_cycle_length(cycle)
    cycle_start = _get_cycle_start(cycle, cycle_length)
    cycle_valid = _is_valid_cycle(cycle, cycle_length, cycle_start)

    # then
    assert cycle_valid
    assert cycle_length == rand_cycle_length
    assert cycle_start == rand_cycle_start


def test_get_cycle_length():
    # given
    cycle = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

    # when / then
    assert _get_cycle_length(cycle) == 5


def test_get_cycle_length_error():
    # given
    cycle = np.array([1, 1, 1])

    # when / then
    assert _get_cycle_length(cycle) == -1


def test_get_cycle_start():
    # given
    cycle = np.array([0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    cycle_length = 5

    # when
    assert _get_cycle_start(cycle, cycle_length) == 3


def test_get_cycle_start_error():
    # given
    cycle = np.array([0, 0, 0])
    cycle_length = 5

    # when
    assert _get_cycle_start(cycle, cycle_length) == -1


def test_is_valid_cycle_valid():
    # given
    cycle = np.array([0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    cycle_length = 5
    cycle_start = 3

    # when
    valid = _is_valid_cycle(cycle, cycle_length, cycle_start)

    # then
    assert valid


def test_is_valid_cycle_invalid():
    # given
    cycle = np.array([0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 9, 4, 5, 1, 2, 3, 4, 5])
    cycle_length = 5
    cycle_start = 3

    # when
    valid = _is_valid_cycle(cycle, cycle_length, cycle_start)

    # then
    assert not valid


def test_determine_dia_cycle():
    # given
    num_spectra = 100
    cycle_length = 5
    cycle_start = 10

    isolation_lower = np.zeros(num_spectra)
    isolation_upper = np.zeros(num_spectra)

    # Create repeating pattern
    pattern_lower = np.array([400, 500, 600, 700, 800])
    pattern_upper = np.array([500, 600, 700, 800, 900])

    for i in range(cycle_start, num_spectra):
        cycle_pos = (i - cycle_start) % cycle_length
        isolation_lower[i] = pattern_lower[cycle_pos]
        isolation_upper[i] = pattern_upper[cycle_pos]

    rt = np.linspace(0, 10, num_spectra)
    df = pd.DataFrame(
        {
            "isolation_lower_mz": isolation_lower,
            "isolation_upper_mz": isolation_upper,
            "rt": rt,
        }
    )

    # when
    cycle, detected_start, detected_length = determine_dia_cycle(df)

    # then
    assert detected_start == cycle_start
    assert detected_length == cycle_length
    assert cycle.shape == (1, cycle_length, 1, 2)
    np.testing.assert_array_equal(cycle[0, :, 0, 0], pattern_lower)
    np.testing.assert_array_equal(cycle[0, :, 0, 1], pattern_upper)


def test_determine_dia_cycle_invalid_cycle():
    # given
    num_spectra = 100
    cycle_length = 5
    cycle_start = 10

    isolation_lower = np.zeros(num_spectra)
    isolation_upper = np.zeros(num_spectra)

    # Create repeating pattern that breaks after a while
    pattern_lower = np.array([400, 500, 600, 700, 800])
    pattern_upper = np.array([500, 600, 700, 800, 900])

    for i in range(cycle_start, num_spectra):
        cycle_pos = (i - cycle_start) % cycle_length
        isolation_lower[i] = pattern_lower[cycle_pos]
        isolation_upper[i] = pattern_upper[cycle_pos]

    # Break the pattern
    isolation_lower[50] = 999

    rt = np.linspace(0, 10, num_spectra)
    df = pd.DataFrame(
        {
            "isolation_lower_mz": isolation_lower,
            "isolation_upper_mz": isolation_upper,
            "rt": rt,
        }
    )

    # when / then
    with pytest.raises(NotValidDiaDataError, match="detected, but is not consistent"):
        determine_dia_cycle(df)


@patch("alphadia.raw_data.dia_cycle._get_cycle_length")
def test_determine_dia_cycle_invalid_length(mock_get_cycle_length):
    # given
    mock_get_cycle_length.return_value = -1

    # when / then
    with pytest.raises(
        NotValidDiaDataError, match="Failed to determine length of DIA cycle"
    ):
        determine_dia_cycle(MagicMock())


@patch("alphadia.raw_data.dia_cycle._get_cycle_length")
@patch("alphadia.raw_data.dia_cycle._get_cycle_start")
def test_determine_dia_cycle_invalid_start(mock_get_cycle_start, mock_get_cycle_length):
    # given
    mock_get_cycle_length.return_value = 1
    mock_get_cycle_start.return_value = -1

    # when / then
    with pytest.raises(
        NotValidDiaDataError, match="Failed to determine start of DIA cycle"
    ):
        determine_dia_cycle(MagicMock())


@patch("alphadia.raw_data.dia_cycle._get_cycle_length")
@patch("alphadia.raw_data.dia_cycle._get_cycle_start")
@patch("alphadia.raw_data.dia_cycle._is_valid_cycle")
def test_determine_dia_cycle_invalid_cycle_(
    mock_is_valid_cycle, mock_get_cycle_start, mock_get_cycle_length
):
    # given
    mock_get_cycle_length.return_value = 1
    mock_get_cycle_start.return_value = 1
    mock_is_valid_cycle.return_value = False

    df = pd.DataFrame(
        {
            "rt": [1, 2, 3],
            "isolation_lower_mz": [1, 2, 3],
            "isolation_upper_mz": [1, 2, 3],
        }
    )
    # when / then
    with pytest.raises(
        NotValidDiaDataError,
        match="Cycle with start 2.00 min and length 1 detected, but is not consistent.",
    ):
        determine_dia_cycle(df)
