"""Prepare test data for the end-to-end tests.

Reads the test case from the yaml file, downloads the required files to the target path and created/adapts the config.
"""

import os.path
import sys
from collections import defaultdict

import yaml

from alphadia.test_data_downloader import DataShareDownloader

OUTPUT_DIR_NAME = "output"

TEST_CASES_FILE_NAME = "e2e_test_cases.yaml"

DEFAULT_CONFIG_FILE_NAME = "config.yaml"

CONFIG_SOURCE_PATH = "../../alphadia/constants/default.yaml"


class YamlKeys:
    """String constants for the yaml keys."""

    TEST_CASES = "test_cases"
    NAME = "name"
    CONFIG = "config"
    LIBRARY = "library_path"
    FASTA = "fasta_paths"
    RAW_DATA = "raw_paths"
    SOURCE_URL = "source_url"


def _download_all_files(test_case: dict, target_path: str) -> dict:
    """Download all files in the test case."""

    downloaded_files = defaultdict(list)
    for item in [YamlKeys.LIBRARY, YamlKeys.FASTA, YamlKeys.RAW_DATA]:
        if item not in test_case:
            continue

        for item_data in test_case[item]:
            file_name = DataShareDownloader(
                item_data[YamlKeys.SOURCE_URL], target_path
            ).download()
            downloaded_files[item].append(file_name)

    return downloaded_files


def _create_config_file(
    target_path: str, downloaded_files: dict, extra_config: dict
) -> None:
    """Create the config file from paths to the input files and optional extra_config."""

    config_to_write = {
        "raw_paths": downloaded_files[YamlKeys.RAW_DATA],
        "output_directory": os.path.join(target_path, OUTPUT_DIR_NAME),
    } | extra_config

    if YamlKeys.LIBRARY in downloaded_files:
        config_to_write = config_to_write | {
            "library_path": downloaded_files[YamlKeys.LIBRARY][0]
        }

    if YamlKeys.FASTA in downloaded_files:
        config_to_write = config_to_write | {
            "fasta_paths": downloaded_files[YamlKeys.FASTA]
        }

    config_target_path = os.path.join(target_path, DEFAULT_CONFIG_FILE_NAME)
    with open(config_target_path, "w") as file:
        yaml.safe_dump(config_to_write, file)


def get_test_case(test_case_name: str) -> dict:
    """Get the test case from the yaml file."""
    with open(TEST_CASES_FILE_NAME) as file:
        test_cases = yaml.safe_load(file)

    return [
        c for c in test_cases[YamlKeys.TEST_CASES] if c[YamlKeys.NAME] == test_case_name
    ][0]


if __name__ == "__main__":
    test_case_name = sys.argv[1]
    target_path = test_case_name

    os.makedirs(os.path.join(target_path, OUTPUT_DIR_NAME), exist_ok=True)

    test_case = get_test_case(test_case_name)

    downloaded_files = _download_all_files(test_case, target_path)

    try:
        extra_config = test_case[YamlKeys.CONFIG]
    except (KeyError, TypeError):
        extra_config = {}

    _create_config_file(target_path, downloaded_files, extra_config)
