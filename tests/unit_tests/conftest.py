import pytest
import os
from alphadia.extraction import data

def pytest_configure(config):
    test_data_path = os.environ.get('TEST_DATA_BRUKER', None)
    print("test_data_path: ", test_data_path)
    if test_data_path is not None:
        pytest.test_data = data.TimsTOFTranspose(test_data_path)
    else:
        pytest.test_data = None