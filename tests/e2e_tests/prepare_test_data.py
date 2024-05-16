"""Prepare test data for the end-to-end tests.

Reads the test case from the yaml file, downloads the required files to the target path and created/adapts the config.
"""

import os.path
import sys

import yaml
import requests

TEST_CASES_FILE_NAME = "e2e_test_cases.yaml"

DEFAULT_CONFIG_FILE_NAME = "config.yaml"

CONFIG_SOURCE_PATH = "../../alphadia/constants/default.yaml"


def _download_file(url: str, target_name: str) -> None:
    """Download a file from the given url to the target path."""
    print(f"downloading {url} to {target_name}")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Download complete: {target_name}")
    else:
        print("Failed to download the file")


def _download_all_files(test_case: dict, target_path: str) -> None:
    """Download all files in the test case."""
    for item in ["library", "raw_data"]:
        for item_data in test_case[item]:
            target = target_path + item_data["target_name"]
            if os.path.exists(target):
                # TODO use cached version only after passed md5 check
                print(f"using cached version of {target}")
                continue

            _download_file(item_data["source_url"], target)


def _add_paths_to_config_file(
    config_name: str, target_path: str, library: str, raw_files: list[str]
) -> None:
    """Add paths to the config file."""
    config_to_write = {
        "library": target_path + library,
        "raw_path_list": [target_path + r for r in raw_files],
        "output_directory": target_path + "output",
    }

    # append to the config file or create a new one
    yaml.safe_dump(config_to_write, open(target_path + config_name, "a"))


def _get_test_case(test_case_name: str) -> dict:
    """Get the test case from the yaml file."""
    with open(TEST_CASES_FILE_NAME, "r") as file:
        test_cases = yaml.safe_load(file)

    return [c for c in test_cases["test_cases"] if c["name"] == test_case_name][0]


if __name__ == "__main__":
    test_case_name = sys.argv[1]  # "basic_e2e"
    target_path = test_case_name + "/"

    os.makedirs(target_path + "output", exist_ok=True)

    test_case = _get_test_case(test_case_name)

    if test_case["config"] is not None:
        _download_file(
            test_case["config"]["source_url"], target_path + DEFAULT_CONFIG_FILE_NAME
        )

    library_name = test_case["library"][0]["target_name"]
    raw_file_names = [r["target_name"] for r in test_case["raw_data"]]

    _add_paths_to_config_file(
        DEFAULT_CONFIG_FILE_NAME, target_path, library_name, raw_file_names
    )

    _download_all_files(test_case, target_path)
