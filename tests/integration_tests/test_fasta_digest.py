import os
import tempfile

from alphabase.constants import _const

from alphadia import search_step


def test_fasta_digest():
    # digest & predict new library
    common_contaminants = os.path.join(_const.CONST_FILE_FOLDER, "contaminants.fasta")
    tempdir = tempfile.gettempdir()
    step = search_step.SearchStep(
        tempdir,
        config={"library_prediction": {"enabled": True}},
        cli_config={"fasta_paths": [common_contaminants]},
    )
    step.load_library()

    assert len(step.spectral_library.precursor_df) > 0
    assert len(step.spectral_library.fragment_df) > 0

    speclib_path = os.path.join(tempdir, "speclib.hdf")
    assert os.path.exists(speclib_path)

    # predict existing library
    step = search_step.SearchStep(
        tempdir,
        config={"library_prediction": {"enabled": True}},
        cli_config={"library_path": speclib_path},
    )
    step.load_library()

    assert len(step.spectral_library.precursor_df) > 0
    assert len(step.spectral_library.fragment_df) > 0

    # load existing library without predict
    step = search_step.SearchStep(
        tempdir,
        config={"library_prediction": {"enabled": False}},
        cli_config={"library_path": speclib_path},
    )
    step.load_library()

    assert len(step.spectral_library.precursor_df) > 0
    assert len(step.spectral_library.fragment_df) > 0
