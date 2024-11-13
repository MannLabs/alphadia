# Discover project files by searching a project regex. Attempted to make this
# as fast as possible by running rglob/glob search to get all filepaths with
# correct endings into a dict, and then searching that dict with the actual
# project-specific regex. Searching the dict is significantly faster than
# stepping through all directories and subdirectories and running re.search.

import sys
from pathlib import Path

import pandas as pd
import regex as re


def match_files(
    project_regex: str,
    source_directories: list,
    search_recursively: bool = True,
    file_ending: str = "raw",
):
    # Convert search directories to paths and report
    _search_dirs = [Path(d) for d in source_directories]

    # Collect all files with correct endings into dict
    print(
        f"--> Collecting '.{file_ending}' files from {source_directories} \n--> search_recursively = {search_recursively}"
    )
    _file_dict = {}
    for _dir in _search_dirs:
        _dir_files = list(
            _dir.rglob(f"*.{file_ending}")
            if search_recursively
            else _dir.glob(f"*.{file_ending}")
        )
        for _file in _dir_files:
            # assign path to filename-key, for quick & unique searching
            _file_dict[str(_file)] = _file
    print(f"--> Collected {len(_file_dict)} '.{file_ending}' files")

    # search project regex against file dict keys and return all matching paths
    _matched_paths = []
    regex_pattern = re.compile(project_regex)
    print(f"--> Searching files matching '{project_regex}'")
    for _file, _path in _file_dict.items():
        if regex_pattern.search(_file):
            _matched_paths.append(_path)

    # report
    print(
        f"--> Discovered {len(_matched_paths)} matching filepaths for {project_regex}."
    )

    # suitable path dataframe
    out_frame = pd.DataFrame(
        columns=["project", "filepath"], index=range(len(_matched_paths))
    )
    out_frame["project"] = project_regex
    out_frame["filepath"] = _matched_paths

    return out_frame


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="Discovering project filenames",
        description="Search project files based on regex string and put them into a csv file for distributed processing",
    )
    parser.add_argument(
        "--project_regex", help="Regex string to match project files", default=".*"
    )
    parser.add_argument(
        "--source_directories", nargs="+", help="List of source directories"
    )
    parser.add_argument("--search_recursively", action="store_true")
    parser.add_argument("--file_ending", default="raw")
    parser.add_argument("--output_filename", default="file_list.csv")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    out_frame = match_files(
        args.project_regex,
        args.source_directories,
        args.search_recursively,
        args.file_ending,
    )

    output_path = (
        args.output_filename
        if os.path.isabs(args.output_filename)
        else os.path.join("./", args.output_filename)
    )
    out_frame.to_csv(output_path, index=False)
