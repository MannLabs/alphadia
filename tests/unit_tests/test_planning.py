import tempfile
import pytest
import os
from alphadia import planning
from alphabase.constants import _const


@pytest.mark.slow
def test_fasta_digest():
    # digest & predict new library
    common_contaminants = os.path.join(_const.CONST_FILE_FOLDER, "contaminants.fasta")
    tempdir = tempfile.gettempdir()
    plan = planning.Plan(
        tempdir,
        fasta_path_list=[common_contaminants],
        config={"library_prediction": {"predict": True}},
    )

    assert len(plan.spectral_library.precursor_df) > 0
    assert len(plan.spectral_library.fragment_df) > 0

    speclib_path = os.path.join(tempdir, "speclib.hdf")
    assert os.path.exists(speclib_path)

    # predict existing library
    plan = planning.Plan(
        tempdir,
        library_path=speclib_path,
        config={"library_prediction": {"predict": True}},
    )
    assert len(plan.spectral_library.precursor_df) > 0
    assert len(plan.spectral_library.fragment_df) > 0

    # load existing library without predict
    plan = planning.Plan(
        tempdir,
        library_path=speclib_path,
        config={"library_prediction": {"predict": False}},
    )
    assert len(plan.spectral_library.precursor_df) > 0
    assert len(plan.spectral_library.fragment_df) > 0
