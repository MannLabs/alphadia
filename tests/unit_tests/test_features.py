import numpy as np
import pytest

from alphadia.features import center_envelope_1d


@pytest.mark.parametrize(
    "input_array, expected_output",
    [
        (
            np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=np.float64),
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=np.float64,
            ),
        ),
        (
            np.array([[100, 10, 1, 1, 1, 10, 100]], dtype=np.float64),
            np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=np.float64),
        ),
        (
            np.array([[100, 0, 0, 1, 0, 0, 100]], dtype=np.float64),
            np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float64),
        ),
        (
            np.array([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float64),
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=np.float64,
            ),
        ),
        (
            np.array([[100, 10, 1, 1, 1, 1, 10, 100]], dtype=np.float64),
            np.array([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float64),
        ),
        (
            np.array([[100, 0, 0, 1, 1, 0, 0, 100]], dtype=np.float64),
            np.array([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=np.float64),
        ),
    ],
)
def test_center_envelope_1d_simple(input_array, expected_output):
    # given

    # when
    center_envelope_1d(input_array)

    # then
    np.testing.assert_array_almost_equal(input_array, expected_output)


def test_center_envelope_1d_multiple_rows():
    # given
    shape = (10, 11)

    input_array = np.random.rand(*shape)
    output_array = input_array.copy()
    # when
    center_envelope_1d(input_array)

    # then
    assert output_array.shape == input_array.shape
    assert np.all(input_array[:, 0] <= output_array[:, 0])
    assert np.all(input_array[:, -1] <= output_array[:, -1])


def test_center_envelope_1d_empty_array():
    # given
    x = np.array([[]], dtype=np.float64)

    # when
    center_envelope_1d(x)

    # then
    assert x.shape == (1, 0)
