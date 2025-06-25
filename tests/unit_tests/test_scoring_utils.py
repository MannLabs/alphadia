import numpy as np

from alphadia.plexscoring.scoring_utils import (
    correlation_coefficient,
    median_axis,
    normalize_profiles,
)


def test_correlation_coefficient():
    # Given: A base array and test cases for different correlation scenarios
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    ys = np.array(
        [
            [1, 2, 3, 4, 5],  # Perfect positive correlation (r = 1.0)
            [-1, -2, -3, -4, -5],  # Perfect negative correlation (r = -1.0)
            [1, -1, 1, -1, 1],  # No correlation (r ≈ 0.0)
            [2, 2, 2, 2, 2],  # Zero variance case (r = 0.0)
        ],
        dtype=np.float32,
    )

    # When: Calculating correlation coefficients
    result = correlation_coefficient(x, ys)

    # Then: Results should match expected correlation values
    expected = np.array([1.0, -1.0, 0.0, 0.0])
    assert np.allclose(result, expected, rtol=1e-5)


def test_normalize_profiles():
    # Given: An array with 7 points (center at index 3) and different intensity patterns
    intensity_slice = np.array(
        [
            [1, 1, 1, 2, 4, 4, 4],  # Normal case with varying intensities
            [1, 2, 0, 0, 0, 6, 7],  # Case with zero values in center region
            [0, 0, 0, 0, 0, 0, 0],  # All zero values case
        ],
        dtype=np.float32,
    )

    # When: Normalizing profiles with no center dilation
    result = normalize_profiles(intensity_slice, center_dilations=0)

    # Then: Profiles should be normalized correctly
    expected = np.array(
        [
            [
                0.5,
                0.5,
                0.5,
                1.0,
                2.0,
                2.0,
                2.0,
            ],  # Each value divided by center value (2)
            [0, 0, 0, 0, 0, 0, 0],  # Zero center region case
            [0, 0, 0, 0, 0, 0, 0],  # All zeros case
        ]
    )
    assert np.allclose(result, expected, rtol=1e-5)


def test_normalize_profiles_dilation():
    # Given: An array with 7 points and different intensity patterns
    intensity_slice = np.array(
        [
            [3, 3, 8, 1, 0, 3, 3],  # Normal case with peak at index 2
            [1, 2, 0, 0, 0, 6, 7],  # Case with zero values in center region
        ],
        dtype=np.float32,
    )

    # When: Normalizing profiles with center dilation of 1
    result = normalize_profiles(intensity_slice, center_dilations=1)

    # Then: Profiles should be normalized using expanded center region
    expected = np.array(
        [
            [
                1,
                1,
                8 / 3,
                1 / 3,
                0,
                1.0,
                1.0,
            ],  # Each value divided by center region mean (3)
            [0, 0, 0, 0, 0, 0, 0],  # Zero center region case
        ]
    )
    assert np.allclose(result, expected, rtol=1e-5)


def test_median_axis_dim0():
    # Given: A 3x4 array of sequential numbers
    array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)

    # When: Calculating median along axis 0 (columns)
    result_axis0 = median_axis(array, axis=0)

    # Then: Should return median of each column
    expected_axis0 = np.array([5, 6, 7, 8])  # Median of each column
    assert np.allclose(result_axis0, expected_axis0)


def test_median_axis_dim1():
    # Given: A 3x4 array of sequential numbers
    array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)

    # When: Calculating median along axis 1 (rows)
    result_axis1 = median_axis(array, axis=1)

    # Then: Should return median of each row
    expected_axis1 = np.array([2.5, 6.5, 10.5])  # Median of each row
    assert np.allclose(result_axis1, expected_axis1)
