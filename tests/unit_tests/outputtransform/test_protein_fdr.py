import numpy as np
import pandas as pd

from alphadia.outputtransform.protein_fdr import (
    NO_EVIDENCE_QVAL,
    transfer_library_pg_qval,
)


def test_transfer_library_pg_qval():
    """Test target groups with a confident precursor take the library q-value, all other rows get none."""
    # given
    psm_df = pd.DataFrame(
        {
            "pg": ["A", "A", "B", "C", "A"],
            "decoy": [0, 0, 0, 0, 1],
            "qval": [0.001, 0.5, 0.5, 0.001, 0.001],
        }
    )
    library_pg_qval = pd.DataFrame({"pg": ["A", "B"], "pg_qval": [0.002, 0.003]})

    # when
    result = transfer_library_pg_qval(psm_df, library_pg_qval)

    # then: A has a confident target, B only a weak one, C is not in the library, the A decoy is a decoy
    np.testing.assert_array_equal(
        result["pg_qval"].values,
        [0.002, 0.002, NO_EVIDENCE_QVAL, NO_EVIDENCE_QVAL, NO_EVIDENCE_QVAL],
    )
