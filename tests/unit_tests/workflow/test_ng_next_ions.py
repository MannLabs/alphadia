"""Unit tests for handing the next ions to the rust library."""

import numpy as np
import pandas as pd

from alphadia.libtransform.next_ions import NEXT_ION_COLUMNS
from alphadia.workflow.peptidecentric.ng.ng_mapper import set_next_ions


class _RecordingSpeclib:
    arrays = None

    def set_next_ions(self, *arrays):
        self.arrays = arrays


def _precursor_df() -> pd.DataFrame:
    df = pd.DataFrame({"precursor_idx": [4, 2]})
    next_mz = np.zeros((2, len(NEXT_ION_COLUMNS)), dtype=np.float32)
    next_mz[0, :2] = [300.0, 400.0]
    next_mz[1, 0] = 500.0
    return pd.concat([df, pd.DataFrame(next_mz, columns=NEXT_ION_COLUMNS)], axis=1)


def test_set_next_ions_passes_every_precursor_with_its_slots():
    # given
    speclib = _RecordingSpeclib()

    # when
    set_next_ions(speclib, _precursor_df())

    # then
    idx, mz = speclib.arrays
    assert idx.dtype == np.uint64
    assert idx.tolist() == [4, 2]
    assert mz.dtype == np.float32
    assert mz.shape == (2, len(NEXT_ION_COLUMNS))
    np.testing.assert_allclose(mz[0, :3], [300.0, 400.0, 0.0])


def test_set_next_ions_skips_libraries_without_next_ions():
    # given
    speclib = _RecordingSpeclib()

    # when
    set_next_ions(speclib, pd.DataFrame({"precursor_idx": [0, 1]}))

    # then
    assert speclib.arrays is None
