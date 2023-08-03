from alphadia.extraction.data import (
    transpose
)
import pytest
import numpy as np

def test_transpose():
    values = np.array([1., 2., 3., 4., 5., 6., 7.])
    tof_indices = np.array([0, 3, 2, 4 ,1, 2, 4])
    push_ptr = np.array([0, 2, 4, 5, 7])

    push_indices, tof_indptr, intensity_values = transpose(tof_indices, push_ptr, values)

    _push_indices = np.array([0, 2, 1, 3, 0, 1, 3])
    _tof_indptr = np.array([0, 1, 2, 4, 5, 7])
    _intensity_values = np.array([1., 5., 3., 6., 2., 4., 7.])

    assert np.allclose(push_indices, _push_indices)
    assert np.allclose(tof_indptr, _tof_indptr)
    assert np.allclose(intensity_values, _intensity_values)

def test_jitclass():
    if pytest.test_data is None:
        pytest.skip("No test data found")
    else:
        jit_data = pytest.test_data.jitclass()

def test_fuzz_get_frame_indices():
    if pytest.test_data is None:
        pytest.skip("No test data found")
    else:
        jit_data = pytest.test_data.jitclass()

        for i in range(1000):

            start_index = np.random.randint(0, jit_data.rt_values.shape[0], size=1)[0]
            stop_index = np.random.randint(start_index, jit_data.rt_values.shape[0], size=1)[0]

            rt = jit_data.rt_values[[start_index, stop_index]]
            frame_indices = jit_data.get_frame_indices(rt, 16)

            # make sure the shape matches
            assert frame_indices.shape == (2,)

            assert np.all(frame_indices >= 0)
            assert np.all(frame_indices < jit_data.rt_values.shape[0])
            assert np.all(frame_indices[1] >= frame_indices[0])

            cycle_offset = (frame_indices-jit_data.zeroth_frame) % jit_data.cycle.shape[1]

            # make sure that the start and stop are at the start of a cycle
            assert np.all(cycle_offset == 0)

            cycle_indices = (frame_indices-jit_data.zeroth_frame) // jit_data.cycle.shape[1]
            cycle_len = cycle_indices[1] - cycle_indices[0]

            # make sure the cycle length is a multiple of 16
            assert cycle_len % 16 == 0

def test_fuzz_get_frame_indices_tolerance():
    if pytest.test_data is None:
        pytest.skip("No test data found")
    else:
        jit_data = pytest.test_data.jitclass()

        for i in range(1000):
            rt = jit_data.rt_values[np.random.randint(0, jit_data.rt_values.shape[0], size=1)][0]
            rt_tolerance = np.random.uniform(0, 300, size=1)[0]

            frame_indices = jit_data.get_frame_indices_tolerance(rt, rt_tolerance, 16)

            # make sure the shape matches
            assert frame_indices.shape == (1, 3)

            frame_indices = frame_indices[0,[0,1]]

            assert np.all(frame_indices >= 0)
            assert np.all(frame_indices < jit_data.rt_values.shape[0])
            assert np.all(frame_indices[1] >= frame_indices[0])

            cycle_offset = (frame_indices-jit_data.zeroth_frame) % jit_data.cycle.shape[1]

            # make sure that the start and stop are at the start of a cycle
            assert np.all(cycle_offset == 0)

            cycle_indices = (frame_indices-jit_data.zeroth_frame) // jit_data.cycle.shape[1]
            cycle_len = cycle_indices[1] - cycle_indices[0]

            # make sure the cycle length is a multiple of 16
            assert cycle_len % 16 == 0

def test_fuzz_get_scan_indices():
    if pytest.test_data is None:
        pytest.skip("No test data found")
    else:
        jit_data = pytest.test_data.jitclass()

        for i in range(1000):

            start_index = np.random.randint(0, jit_data.mobility_values.shape[0], size=1)[0]
            stop_index = np.random.randint(start_index, jit_data.mobility_values.shape[0], size=1)[0]

            mobility = jit_data.mobility_values[[start_index, stop_index]]
            scan_indices = jit_data.get_scan_indices(mobility, 16)

            # make sure the shape matches
            assert scan_indices.shape == (2,)

            assert np.all(scan_indices >= 0)
            assert np.all(scan_indices <= jit_data.mobility_values.shape[0])
            assert np.all(scan_indices[1] >= scan_indices[0])

            scan_len = scan_indices[1] - scan_indices[0]

            # make sure the cycle length is a multiple of 16
            assert scan_len % 16 == 0


def test_fuzz_get_scan_indices_tolerance():
    if pytest.test_data is None:
        pytest.skip("No test data found")
    else:
        jit_data = pytest.test_data.jitclass()

        for i in range(1000):
            mobility = jit_data.mobility_values[np.random.randint(0, jit_data.mobility_values.shape[0], size=1)][0]
            mobility_tolerance = np.random.uniform(0, 0.1, size=1)[0]

            scan_indices = jit_data.get_scan_indices_tolerance(mobility, mobility_tolerance, 16)

            # make sure the shape matches
            assert scan_indices.shape == (1, 3)

            scan_indices = scan_indices[0,[0,1]]

            assert np.all(scan_indices >= 0)

            if not np.all(scan_indices <= jit_data.mobility_values.shape[0]):
                print(mobility, mobility_tolerance)
                print(scan_indices)
            assert np.all(scan_indices[1] >= scan_indices[0])

            scan_len = scan_indices[1] - scan_indices[0]

            # make sure the cycle length is a multiple of 16
            assert scan_len % 16 == 0

def test_fuzz_get_dense():
    if pytest.test_data is None:
        pytest.skip("No test data found")
    else:
        jit_data = pytest.test_data.jitclass()

        for i in range(100):

            min_mz = jit_data.mz_values.min()
            max_mz = jit_data.mz_values.max()
            use_q = np.random.randint(0, 2, dtype=bool)
            
            if use_q:
                _precursor = np.random.uniform(jit_data.quad_min_mz_value, jit_data.quad_max_mz_value, size=1)[0]
                precursor = np.array([[_precursor, _precursor]], dtype=np.float32)

            else:
                precursor = np.array([[-1.,-1.]], dtype=np.float32)

            mz = np.sort(np.random.uniform(min_mz, max_mz, size=10).astype(np.float32))
            mz_tolerance = np.random.uniform(0, 200, size=1)[0]

            rt = jit_data.rt_values[np.random.randint(0, jit_data.rt_values.shape[0], size=1)][0]
            rt_tolerance = np.random.uniform(0, 300, size=1)[0]

            mobility = jit_data.mobility_values[np.random.randint(0, jit_data.mobility_values.shape[0], size=1)][0]
            mobility_tolerance = np.random.uniform(0, 0.1, size=1)[0]

            scan_limits = jit_data.get_scan_indices_tolerance(
                mobility,
                mobility_tolerance
            )

            frame_limits = jit_data.get_frame_indices_tolerance(
                rt,
                rt_tolerance
            )

            dense, precursor_index = jit_data.get_dense(
                frame_limits,
                scan_limits,
                mz,
                mz_tolerance,
                precursor,
                True,
            )

            assert len(dense.shape) == 5
