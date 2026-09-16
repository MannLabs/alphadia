import numpy as np
import pandas as pd
import pytest

from alphadia.outputtransform.quantification.quant_builder import (
    _ion_hash,
    precursor_idx_from_ion,
    prepare_df,
)

LOSS_TYPE_SHIFT = 56


def _int64(values):
    return np.array(values, dtype=np.int64)


def test_ion_hash_rejects_uint64_precursor_idx():
    # a uint64 precursor_idx used to be silently promoted to float64
    with pytest.raises(TypeError, match="No matching definition"):
        _ion_hash(
            np.array([1], dtype=np.uint64),
            _int64([0]),
            _int64([0]),
            _int64([0]),
            _int64([0]),
        )


def test_ion_hash_exact_above_2_53():
    # loss_type sets bit 56, above float64's 53-bit mantissa; the low bits
    # from precursor_idx must survive
    precursor_idx = _int64([1, 2])
    zeros = _int64([0, 0])
    loss_type = _int64([1, 1])

    result = _ion_hash(precursor_idx, zeros, zeros, zeros, loss_type)

    assert result.dtype == np.int64
    assert result.tolist() == [(1 << LOSS_TYPE_SHIFT) + 1, (1 << LOSS_TYPE_SHIFT) + 2]


def test_prepare_df_ion_column_is_int64_for_uint64_precursor_idx():
    fragment_df = pd.DataFrame(
        {
            "precursor_idx": np.array([1, 2], dtype=np.uint64),
            "number": np.array([0, 0], dtype=np.uint8),
            "type": np.array([0, 0], dtype=np.uint8),
            "charge": np.array([0, 0], dtype=np.uint8),
            "loss_type": np.array([1, 1], dtype=np.uint8),
            "intensity": [10.0, 20.0],
        }
    )
    psm_df = pd.DataFrame({"precursor_idx": [1, 2]})

    result = prepare_df(fragment_df, psm_df, columns=["intensity"])

    assert result["ion"].dtype == np.int64
    assert result["ion"].tolist() == [
        (1 << LOSS_TYPE_SHIFT) + 1,
        (1 << LOSS_TYPE_SHIFT) + 2,
    ]


def test_precursor_idx_from_ion_roundtrip():
    # all upper fields set, precursor_idx up to the 32-bit limit
    precursor_idx = _int64([0, 1, 123456789, 2**32 - 1])
    number = _int64([1, 255, 7, 255])
    type_ = _int64([98, 121, 255, 98])
    charge = _int64([1, 2, 3, 255])
    loss_type = _int64([0, 1, 127, 127])

    ion = _ion_hash(precursor_idx, number, type_, charge, loss_type)
    result = precursor_idx_from_ion(ion)

    assert result.dtype == np.uint32
    assert result.tolist() == precursor_idx.tolist()
