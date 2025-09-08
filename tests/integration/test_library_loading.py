import tempfile

import pytest
from alphabase.tools.data_downloader import DataShareDownloader

from alphadia import search_step


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
        step = search_step.SearchStep(
            temp_directory, {"library_path": test_data_location}
        )
        assert len(step.spectral_library.precursor_df) > 0
        assert len(step.spectral_library.fragment_df) > 0
