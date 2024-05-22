"""Prepare test data for the end-to-end tests.

Reads the test case from the yaml file, downloads the required files to the target path and created/adapts the config.
"""

import os.path
import sys
from collections import defaultdict

import yaml
import requests

from alphadia.testing import DataShareDownloader

OUTPUT_DIR_NAME = "output"

TEST_CASES_FILE_NAME = "e2e_test_cases.yaml"

DEFAULT_CONFIG_FILE_NAME = "config.yaml"

CONFIG_SOURCE_PATH = "../../alphadia/constants/default.yaml"


def _download_file(url: str, target_path: str) -> None:
    """Download a file from the given `url` to the `target_path`."""
    # could potentially reuse testing.py:download_datashare()
    print(f"downloading {url} to {target_path}")

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print("Failed to download the file")
        return

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Download complete: {target_path}")


def _download_all_files(test_case: dict, target_path: str) -> dict:
    """Download all files in the test case."""

    downloaded_files = defaultdict(list)
    for item in ["library", "fasta", "raw_data"]:
        if item not in test_case:
            continue

        for item_data in test_case[item]:
            file_name = DataShareDownloader(
                item_data["source_url"], target_path
            ).download()
            downloaded_files[item].append(file_name)

    return downloaded_files


def _create_config_file(
    target_path: str, downloaded_files: dict, extra_config: dict
) -> None:
    """Create the config file from paths to the input files and optional extra_config."""

    config_to_write = {
        "raw_path_list": downloaded_files["raw_data"],
        "output_directory": os.path.join(target_path, OUTPUT_DIR_NAME),
    } | extra_config

    if "library" in downloaded_files:
        config_to_write = config_to_write | {"library": downloaded_files["library"][0]}

    if "fasta" in downloaded_files:
        config_to_write = config_to_write | {"fasta_list": downloaded_files["fasta"]}

    config_target_path = os.path.join(target_path, DEFAULT_CONFIG_FILE_NAME)
    yaml.safe_dump(config_to_write, open(config_target_path, "w"))


def get_test_case(test_case_name: str) -> dict:
    """Get the test case from the yaml file."""
    with open(TEST_CASES_FILE_NAME, "r") as file:
        test_cases = yaml.safe_load(file)

    return [c for c in test_cases["test_cases"] if c["name"] == test_case_name][0]


if __name__ == "__main__":
    test_case_name = sys.argv[1]
    target_path = test_case_name

    os.makedirs(os.path.join(target_path, OUTPUT_DIR_NAME), exist_ok=True)

    test_case = get_test_case(test_case_name)

    downloaded_files = _download_all_files(test_case, target_path)

    try:
        extra_config = test_case["config"]
    except (KeyError, TypeError):
        extra_config = {}

    _create_config_file(target_path, downloaded_files, extra_config)
