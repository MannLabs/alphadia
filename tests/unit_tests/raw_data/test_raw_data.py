import numpy as np
import pytest

from alphadia.raw_data import alpharaw_wrapper, bruker


def test_transpose():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    tof_indices = np.array([0, 3, 2, 4, 1, 2, 4])
    push_ptr = np.array([0, 2, 4, 5, 7])
    n_tof_indices = 7

    push_indices, tof_indptr, intensity_values = bruker._transpose(
        tof_indices, push_ptr, n_tof_indices, values
    )

    _push_indices = np.array([0, 2, 1, 3, 0, 1, 3])
    _tof_indptr = np.array([0, 1, 2, 4, 5, 7, 7, 7])
    _intensity_values = np.array([1.0, 5.0, 3.0, 6.0, 2.0, 4.0, 7.0])

    assert np.allclose(push_indices, _push_indices)
    assert np.allclose(tof_indptr, _tof_indptr)
    assert np.allclose(intensity_values, _intensity_values)


@pytest.fixture
def mock_alpha_raw_jit():
    # Create mock data for AlphaRawJIT
    cycle = np.zeros((1, 5, 1, 2), dtype=np.float64)
    cycle[0, :, 0, 0] = [100.0, 200.0, 300.0, 400.0, 500.0]
    cycle[0, :, 0, 1] = [200.0, 300.0, 400.0, 500.0, 600.0]

    rt_values = np.arange(0, 100, 1).astype(np.float32)
    mobility_values = np.array([0.0, 0.0], dtype=np.float32)
    zeroth_frame = 0

    max_mz_value = 1000.0
    min_mz_value = 100.0

    quad_max_mz_value = 500.0
    quad_min_mz_value = 100.0

    precursor_cycle_max_index = 19

    # 0, 10, 20, ..., 990 with length 100
    peak_start_idx_list = np.arange(0, 1000, 10, dtype=np.int64)
    # 1, 2, 3, ..., 1001 with length 100
    peak_stop_idx_list = peak_start_idx_list + 1

    mz_values = (
        np.random.rand(1000) * (max_mz_value - min_mz_value) + min_mz_value
    ).astype(np.float32)
    intensity_values = np.random.rand(1000).astype(np.float32)

    scan_max_index = 0

    frame_max_index = 99

    # Instantiate AlphaRawJIT
    alpha_raw_jit = alpharaw_wrapper.AlphaRawJIT(
        cycle=cycle,
        rt_values=rt_values,
        mobility_values=mobility_values,
        zeroth_frame=zeroth_frame,
        max_mz_value=max_mz_value,
        min_mz_value=min_mz_value,
        quad_max_mz_value=quad_max_mz_value,
        quad_min_mz_value=quad_min_mz_value,
        precursor_cycle_max_index=precursor_cycle_max_index,
        peak_start_idx_list=peak_start_idx_list,
        peak_stop_idx_list=peak_stop_idx_list,
        mz_values=mz_values,
        intensity_values=intensity_values,
        scan_max_index=scan_max_index,
        frame_max_index=frame_max_index,
    )
    return alpha_raw_jit


def test_get_frame_indices(mock_alpha_raw_jit):
    # given
    optimize_size = 1
    min_size = 1
    rt_values = np.array([10.0, 20.0], dtype=np.float32)
    expected_indices = np.array([[10, 20, 1]], dtype=np.int64)

    # when
    frame_indices = mock_alpha_raw_jit._get_frame_indices(
        rt_values, optimize_size, min_size
    )

    # then
    assert np.array_equal(frame_indices, expected_indices)


def test_get_frame_indices_optimization_right(mock_alpha_raw_jit):
    # given
    optimize_size = 4
    min_size = 1
    rt_values = np.array([10.0, 20.0], dtype=np.float32)
    expected_indices = np.array([[10, 30, 1]], dtype=np.int64)

    # when
    frame_indices = mock_alpha_raw_jit._get_frame_indices(
        rt_values, optimize_size, min_size
    )

    # then
    assert np.array_equal(frame_indices, expected_indices)


def test_get_frame_indices_optimization_right_min_size(mock_alpha_raw_jit):
    # given
    optimize_size = 4
    min_size = 8
    rt_values = np.array([10.0, 20.0], dtype=np.float32)
    expected_indices = np.array([[10, 50, 1]], dtype=np.int64)

    # when
    frame_indices = mock_alpha_raw_jit._get_frame_indices(
        rt_values, optimize_size, min_size
    )

    # then
    assert np.array_equal(frame_indices, expected_indices)


def test_get_frame_indices_optimization_left(mock_alpha_raw_jit):
    # given
    optimize_size = 4
    min_size = 1
    rt_values = np.array([90.0, 95.0], dtype=np.float32)
    expected_indices = np.array([[75, 95, 1]], dtype=np.int64)

    # when
    frame_indices = mock_alpha_raw_jit._get_frame_indices(
        rt_values, optimize_size, min_size
    )

    # then
    assert np.array_equal(frame_indices, expected_indices)

    # test optimization and min size left


def test_get_frame_indices_optimization_left_min_size(mock_alpha_raw_jit):
    # given
    optimize_size = 4
    min_size = 8
    rt_values = np.array([90.0, 95.0], dtype=np.float32)
    expected_indices = np.array([[55, 95, 1]], dtype=np.int64)

    # when
    frame_indices = mock_alpha_raw_jit._get_frame_indices(
        rt_values, optimize_size, min_size
    )

    # then
    assert np.array_equal(frame_indices, expected_indices)


def test_get_frame_indices_optimization_left_min_size_overflow(mock_alpha_raw_jit):
    # given
    optimize_size = 4
    min_size = 1000
    rt_values = np.array([90.0, 95.0], dtype=np.float32)
    expected_indices = np.array([[5, 95, 1]], dtype=np.int64)

    # when
    frame_indices = mock_alpha_raw_jit._get_frame_indices(
        rt_values, optimize_size, min_size
    )

    # then
    assert np.array_equal(frame_indices, expected_indices)
