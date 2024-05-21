"""Prepare test data for the end-to-end tests.

Reads the test case from the yaml file, downloads the required files to the target path and created/adapts the config.
"""

import os.path
import sys

import yaml
import requests

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


def _download_all_files(test_case: dict, target_path: str) -> None:
    """Download all files in the test case."""
    for item in ["library", "raw_data"]:
        for item_data in test_case[item]:
            target = os.path.join(target_path, item_data["target_name"])
            if os.path.exists(target):
                # TODO use cached version only after passed md5 check
                print(f"using cached version of {target}")
                continue

            _download_file(item_data["source_url"], target)


def _create_config_file(
    target_path: str, library: str, raw_files: list[str], extra_config: dict
) -> None:
    """Create the config file from paths to the input files and optional extra_config."""
    config_to_write = {
        "library": os.path.join(target_path, library),
        "raw_path_list": [
            os.path.join(target_path, raw_file) for raw_file in raw_files
        ],
        "output_directory": os.path.join(target_path, OUTPUT_DIR_NAME),
    } | extra_config

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

    library_name = test_case["library"][0]["target_name"]
    raw_file_names = [r["target_name"] for r in test_case["raw_data"]]
    try:
        extra_config = test_case["config"]
    except (KeyError, TypeError):
        extra_config = {}

    _create_config_file(target_path, library_name, raw_file_names, extra_config)

    _download_all_files(test_case, target_path)
