import pandas as pd
from pathlib import Path
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
    print(f"--> Collecting '.{file_ending}' files from {source_directories} \n--> search_recursively = {search_recursively}")
    _file_dict = {}
    for _dir in _search_dirs:
        _dir_files = list(_dir.rglob(f"*.{file_ending}") if search_recursively else _dir.glob(f"*.{file_ending}"))
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
    print(f"--> Discovered {len(_matched_paths)} matching filepaths for {project_regex}.")

    # suitable path dataframe
    out_frame = pd.DataFrame(columns=['project','filepath'], index=range(len(_matched_paths)))
    out_frame['project'] = project_regex
    out_frame['filepath'] = _matched_paths

    return out_frame