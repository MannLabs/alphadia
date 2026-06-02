import h5py
import numpy as np
import pandas as pd
import pytest

from alphadia.exceptions import NotValidDiaDataError
from alphadia.raw_data import alpharaw_wrapper, bruker
from alphadia.workflow.managers.raw_file_manager import _is_alphatims_hdf


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


def _make_timstof_hdf_dict():
    """Build a minimal alphatims-style TimsTOF attribute dict.

    The transpose inputs reuse the example from `test_transpose` so the expected
    transposed output is known.
    """
    intensity_values = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    tof_indices = np.array([0, 3, 2, 4, 1, 2, 4], dtype=np.uint32)
    push_indptr = np.array([0, 2, 4, 5, 7], dtype=np.int64)
    mz_values = np.arange(7, dtype=np.float64)  # n_tof_indices == 7

    return {
        "_accumulation_times": np.ones(4, dtype=np.float64),
        "_cycle": np.zeros((1, 5, 1, 2), dtype=np.float64),
        "_dia_mz_cycle": np.zeros((5, 2), dtype=np.float64),
        "_dia_precursor_cycle": np.zeros(5, dtype=np.int64),
        "_frame_max_index": 4,
        "_intensity_corrections": np.ones(4, dtype=np.float64),
        "_intensity_max_value": 7.0,
        "_intensity_min_value": 1.0,
        "_intensity_values": intensity_values,
        "_max_accumulation_time": 1.0,
        "_mobility_max_value": 1.3,
        "_mobility_min_value": 0.6,
        "_mobility_values": np.linspace(1.3, 0.6, 5),
        "_mz_values": mz_values,
        "_precursor_indices": np.zeros(5, dtype=np.int64),
        "_precursor_max_index": 1,
        "_push_indptr": push_indptr,
        "_quad_indptr": np.array([0, 5], dtype=np.int64),
        "_quad_max_mz_value": 600.0,
        "_quad_min_mz_value": 100.0,
        "_quad_mz_values": np.zeros((1, 2), dtype=np.float64),
        "_raw_quad_indptr": np.array([0, 5], dtype=np.int64),
        "_rt_values": np.linspace(0, 10, 5),
        "_scan_max_index": 5,
        "_tof_indices": tof_indices,
        "_tof_max_index": 7,
        "_use_calibrated_mz_values_as_default": 0,
        "_zeroth_frame": True,
        # non-required structures exercising the DataFrame and nested-dict branches
        "_frames": pd.DataFrame({"Time": np.linspace(0, 10, 5), "Id": np.arange(5)}),
        "_meta_data": {"SampleName": "test"},
    }


def _write_group_from_dict(group, data):
    """Write a dict into an HDF group the way alphatims serializes a TimsTOF object."""
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            sub = group.create_group(key)
            sub.attrs["is_pd_dataframe"] = True
            for col in value.columns:
                sub.create_dataset(col, data=value[col].to_numpy())
        elif isinstance(value, dict):
            _write_group_from_dict(group.create_group(key), value)
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        else:
            group.attrs[key] = value


def _write_timstof_hdf(path, data, group_name=bruker.ALPHATIMS_HDF_GROUP):
    with h5py.File(path, "w") as hdf_file:
        _write_group_from_dict(hdf_file.create_group(group_name), data)


def test_timstof_transpose_load_from_hdf(tmp_path):
    # given
    data = _make_timstof_hdf_dict()
    hdf_path = str(tmp_path / "sample.hdf")
    _write_timstof_hdf(hdf_path, data)

    # when
    dia_data = bruker.TimsTOFTranspose(hdf_path)

    # then
    assert dia_data.has_mobility is True
    assert dia_data.has_ms1 is True
    # transposed output (see test_transpose)
    assert np.array_equal(dia_data._push_indices, np.array([0, 2, 1, 3, 0, 1, 3]))
    assert np.array_equal(dia_data._tof_indptr, np.array([0, 1, 2, 4, 5, 7, 7, 7]))
    assert np.allclose(dia_data._intensity_values, np.array([1, 5, 3, 6, 2, 4, 7]))
    # untouched attributes round-trip from the HDF
    assert np.allclose(dia_data._rt_values, data["_rt_values"])
    assert dia_data._frames["Time"].tolist() == data["_frames"]["Time"].tolist()
    assert dia_data._meta_data["SampleName"] == "test"


def test_timstof_transpose_hdf_missing_attribute_raises(tmp_path):
    # given
    data = _make_timstof_hdf_dict()
    del data["_cycle"]
    hdf_path = str(tmp_path / "incomplete.hdf")
    _write_timstof_hdf(hdf_path, data)

    # when / then
    with pytest.raises(NotValidDiaDataError, match="_cycle"):
        bruker.TimsTOFTranspose(hdf_path)


def test_is_alphatims_hdf(tmp_path):
    # given
    alphatims_path = str(tmp_path / "alphatims.hdf")
    with h5py.File(alphatims_path, "w") as hdf_file:
        hdf_file.create_group(bruker.ALPHATIMS_HDF_GROUP)

    alpharaw_path = str(tmp_path / "alpharaw.hdf")
    with h5py.File(alpharaw_path, "w") as hdf_file:
        hdf_file.create_group("ms_data")

    # when / then
    assert _is_alphatims_hdf(alphatims_path) is True
    assert _is_alphatims_hdf(alpharaw_path) is False
