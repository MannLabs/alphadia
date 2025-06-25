import numpy as np
import pytest

from alphadia.data import alpharaw_wrapper, bruker


def test_transpose():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    tof_indices = np.array([0, 3, 2, 4, 1, 2, 4])
    push_ptr = np.array([0, 2, 4, 5, 7])
    n_tof_indices = 7

    push_indices, tof_indptr, intensity_values = bruker.transpose(
        tof_indices, push_ptr, n_tof_indices, values
    )

    _push_indices = np.array([0, 2, 1, 3, 0, 1, 3])
    _tof_indptr = np.array([0, 1, 2, 4, 5, 7, 7, 7])
    _intensity_values = np.array([1.0, 5.0, 3.0, 6.0, 2.0, 4.0, 7.0])

    assert np.allclose(push_indices, _push_indices)
    assert np.allclose(tof_indptr, _tof_indptr)
    assert np.allclose(intensity_values, _intensity_values)


@pytest.mark.slow()
def test_raw_data():  # TODO this is never executed as TEST_DATA_DIR (cf. conftest.py) is never set
    if pytest.test_data is None:
        pytest.skip("No test data found")
        return

    for name, file_list in pytest.test_data.items():
        for file in file_list:
            if name == "bruker":
                jit_data = bruker.TimsTOFTranspose(file).jitclass()
            elif name == "thermo":
                jit_data = alpharaw_wrapper.Thermo(file).jitclass()
            else:
                continue

        run_test_on_raw(name, jit_data)


def run_test_on_raw(name, jit_data):
    print("Testing", name)
    fuzz_get_frame_indices(jit_data)
    fuzz_get_frame_indices_tolerance(jit_data)
    fuzz_get_scan_indices(jit_data)
    fuzz_get_scan_indices_tolerance(jit_data)
    fuzz_get_dense(jit_data)


def fuzz_get_frame_indices(jit_data):
    for _ in range(1):
        start_index = np.random.randint(0, jit_data.rt_values.shape[0], size=1)[0]
        stop_index = np.random.randint(
            start_index, jit_data.rt_values.shape[0], size=1
        )[0]

        rt = jit_data.rt_values[[start_index, stop_index]]
        frame_indices = jit_data.get_frame_indices(rt, 16)

        # make sure the shape matches
        assert frame_indices.shape == (1, 3)

        start_stop = frame_indices[0, [0, 1]]

        assert np.all(start_stop >= 0)
        assert np.all(start_stop < jit_data.rt_values.shape[0])
        assert np.all(start_stop[1] >= start_stop[0])

        cycle_offset = (start_stop - jit_data.zeroth_frame) % jit_data.cycle.shape[1]

        # make sure that the start and stop are at the start of a cycle
        assert np.all(cycle_offset == 0)

        cycle_indices = (start_stop - jit_data.zeroth_frame) // jit_data.cycle.shape[1]
        cycle_len = cycle_indices[1] - cycle_indices[0]

        # make sure the cycle length is a multiple of 16
        assert cycle_len % 16 == 0


def fuzz_get_frame_indices_tolerance(jit_data):
    for _ in range(1000):
        rt = jit_data.rt_values[
            np.random.randint(0, jit_data.rt_values.shape[0], size=1)
        ][0]
        rt_tolerance = np.random.uniform(0, 300, size=1)[0]

        frame_indices = jit_data.get_frame_indices_tolerance(rt, rt_tolerance, 16)
        # make sure the shape matches
        assert frame_indices.shape == (1, 3)

        start_stop = frame_indices[0, [0, 1]]

        assert np.all(start_stop >= 0)
        assert np.all(start_stop < jit_data.rt_values.shape[0])
        assert np.all(start_stop[1] >= start_stop[0])

        cycle_offset = (start_stop - jit_data.zeroth_frame) % jit_data.cycle.shape[1]

        # make sure that the start and stop are at the start of a cycle
        assert np.all(cycle_offset == 0)

        cycle_indices = (start_stop - jit_data.zeroth_frame) // jit_data.cycle.shape[1]
        cycle_len = cycle_indices[1] - cycle_indices[0]

        # make sure the cycle length is a multiple of 16
        assert cycle_len % 16 == 0


def fuzz_get_scan_indices(jit_data):
    for _ in range(1000):
        start_index = np.random.randint(0, jit_data.mobility_values.shape[0], size=1)[0]
        stop_index = np.random.randint(
            start_index, jit_data.mobility_values.shape[0], size=1
        )[0]

        mobility = jit_data.mobility_values[[start_index, stop_index]]
        scan_indices = jit_data.get_scan_indices(mobility, 16)

        # make sure the shape matches
        assert scan_indices.shape == (1, 3)

        start_stop = scan_indices[0, [0, 1]]

        assert np.all(start_stop >= 0)
        assert np.all(start_stop <= jit_data.mobility_values.shape[0])
        assert np.all(start_stop[1] >= start_stop[0])

        scan_len = start_stop[1] - start_stop[0]

        # make sure the cycle length is a multiple of 16
        assert scan_len % 16 == 0 or scan_len == 2


def fuzz_get_scan_indices_tolerance(jit_data):
    for _ in range(1000):
        mobility = jit_data.mobility_values[
            np.random.randint(0, jit_data.mobility_values.shape[0], size=1)
        ][0]
        mobility_tolerance = np.random.uniform(0, 0.1, size=1)[0]

        scan_indices = jit_data.get_scan_indices_tolerance(
            mobility, mobility_tolerance, 16
        )

        # make sure the shape matches
        assert scan_indices.shape == (1, 3)

        scan_indices = scan_indices[0, [0, 1]]

        assert np.all(scan_indices >= 0)

        if not np.all(scan_indices <= jit_data.mobility_values.shape[0]):
            print(mobility, mobility_tolerance)
            print(scan_indices)
        assert np.all(scan_indices[1] >= scan_indices[0])

        scan_len = scan_indices[1] - scan_indices[0]

        # make sure the cycle length is a multiple of 16
        assert scan_len % 16 == 0 or scan_len == 2


def fuzz_get_dense(jit_data):
    for _ in range(100):
        min_mz = jit_data.mz_values.min()
        max_mz = jit_data.mz_values.max()
        use_q = np.random.randint(0, 2, dtype=bool)

        if use_q:
            _precursor = np.random.uniform(
                jit_data.quad_min_mz_value, jit_data.quad_max_mz_value, size=1
            )[0]
            precursor = np.array([[_precursor, _precursor]], dtype=np.float32)

        else:
            precursor = np.array([[-1.0, -1.0]], dtype=np.float32)

        mz = np.sort(np.random.uniform(min_mz, max_mz, size=10).astype(np.float32))
        mz_tolerance = np.random.uniform(0, 200, size=1)[0]

        rt = jit_data.rt_values[
            np.random.randint(0, jit_data.rt_values.shape[0], size=1)
        ][0]
        rt_tolerance = np.random.uniform(0, 300, size=1)[0]

        mobility = jit_data.mobility_values[
            np.random.randint(0, jit_data.mobility_values.shape[0], size=1)
        ][0]
        mobility_tolerance = np.random.uniform(0, 0.1, size=1)[0]

        scan_limits = jit_data.get_scan_indices_tolerance(mobility, mobility_tolerance)

        frame_limits = jit_data.get_frame_indices_tolerance(rt, rt_tolerance)

        dense, precursor_index = jit_data.get_dense(
            frame_limits,
            scan_limits,
            mz,
            mz_tolerance,
            precursor,
        )
        assert dense.ndim == 5
        assert dense.dtype == np.float32
        assert precursor_index.ndim == 1
        assert precursor_index.dtype == np.int64


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
    frame_indices = mock_alpha_raw_jit.get_frame_indices(
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
    frame_indices = mock_alpha_raw_jit.get_frame_indices(
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
    frame_indices = mock_alpha_raw_jit.get_frame_indices(
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
    frame_indices = mock_alpha_raw_jit.get_frame_indices(
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
    frame_indices = mock_alpha_raw_jit.get_frame_indices(
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
    frame_indices = mock_alpha_raw_jit.get_frame_indices(
        rt_values, optimize_size, min_size
    )

    # then
    assert np.array_equal(frame_indices, expected_indices)
