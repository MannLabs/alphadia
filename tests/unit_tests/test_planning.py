import os
import tempfile

import pytest
from alphabase.constants import _const
from alphabase.constants.modification import MOD_DF

from alphadia import planning
from alphadia.test_data_downloader import DataShareDownloader


@pytest.mark.slow()
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


@pytest.mark.slow()
def test_library_loading():
    temp_directory = tempfile.gettempdir()

    test_cases = [
        {
            "name": "alphadia_speclib",
            "url": "https://datashare.biochem.mpg.de/s/NLZ0Y6qNfwMlGs0",
        },
        {
            "name": "diann_speclib",
            "url": "https://datashare.biochem.mpg.de/s/DF12ObSdZnBnqUV",
        },
        {
            "name": "msfragger_speclib",
            "url": "https://datashare.biochem.mpg.de/s/Cka1utORt3r5A4a",
        },
    ]

    for test_dict in test_cases:
        print("Testing {}".format(test_dict["name"]))

        test_data_location = DataShareDownloader(
            test_dict["url"], temp_directory
        ).download()
        plan = planning.Plan(temp_directory, library_path=test_data_location)
        assert len(plan.spectral_library.precursor_df) > 0
        assert len(plan.spectral_library.fragment_df) > 0


def test_custom_modifications():
    temp_directory = tempfile.gettempdir()

    config = {
        "custom_modififcations": {
            "ThisModDoesNotExists@K": {
                "composition": "H(10)",
            },
        }
    }

    plan = planning.Plan(temp_directory, [], config=config)  # noqa F841
    assert "ThisModDoesNotExists@K" in MOD_DF["mod_name"].values
