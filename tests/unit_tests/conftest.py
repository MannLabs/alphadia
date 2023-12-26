import pytest
import os
import re
from alphadia import data

import matplotlib


def pytest_configure(config):
    test_data_path = os.environ.get("TEST_DATA_DIR", None)

    pytest.test_data = {}

    if test_data_path is None:
        return

    # get all folders in the test_data_path
    raw_folders = [
        item
        for item in os.listdir(test_data_path)
        if os.path.isdir(os.path.join(test_data_path, item))
    ]

    for raw_folder in raw_folders:
        raw_files = [
            os.path.join(test_data_path, raw_folder, item)
            for item in os.listdir(os.path.join(test_data_path, raw_folder))
            if bool(re.search("(.d|.raw|.hdf)$", item))
        ]
        pytest.test_data[raw_folder] = raw_files

    # important to supress matplotlib output
    matplotlib.use("Agg")
