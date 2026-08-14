# Link quantification folders of previously searched raw files into the
# quant directory of a distributed search, so that the first search can be
# skipped and the MBR library is built from the existing results.
# Symlinks are used to avoid duplicating potentially large quantification files.

import argparse
import os
import sys

QUANT_FOLDER_NAME = "quant"
REQUIRED_FILES = ["psm.parquet", "frag.parquet"]

# parse input parameters
parser = argparse.ArgumentParser(
    prog="DistributedAlphaDIAReuseQuant",
    description="Link quantification folders from previous AlphaDIA runs into a distributed search.",
)
parser.add_argument("--run_folders_filename")
parser.add_argument("--target_directory")
args = parser.parse_args()

# read the run folders, one per line, ignoring empty lines and comments
with open(args.run_folders_filename) as file:
    run_folders = [
        line.strip() for line in file if line.strip() and not line.startswith("#")
    ]

if not run_folders:
    print(
        f"No run folders found in {args.run_folders_filename}, exiting...",
        file=sys.stderr,
    )
    sys.exit(1)

# link every raw file's quantification folder, keeping track of its origin
origin_by_raw_name = {}
for run_folder in run_folders:
    quant_folder = os.path.join(os.path.abspath(run_folder), QUANT_FOLDER_NAME)

    if not os.path.isdir(quant_folder):
        print(
            f"No '{QUANT_FOLDER_NAME}' folder in {run_folder}, exiting...",
            file=sys.stderr,
        )
        sys.exit(1)

    for raw_name in sorted(os.listdir(quant_folder)):
        source_folder = os.path.join(quant_folder, raw_name)
        if not os.path.isdir(source_folder):
            continue

        # an incomplete folder would make AlphaDIA search the raw file again
        # and overwrite the original results through the symlink
        missing_files = [
            file_name
            for file_name in REQUIRED_FILES
            if not os.path.exists(os.path.join(source_folder, file_name))
        ]
        if missing_files:
            print(
                f"Missing {missing_files} in {source_folder}, exiting...",
                file=sys.stderr,
            )
            sys.exit(1)

        if raw_name in origin_by_raw_name:
            print(
                f"Raw file '{raw_name}' is present in both {origin_by_raw_name[raw_name]} and {quant_folder}, exiting...",
                file=sys.stderr,
            )
            sys.exit(1)

        os.symlink(source_folder, os.path.join(args.target_directory, raw_name))
        origin_by_raw_name[raw_name] = quant_folder

# the only return value needed is the number of linked raw files
print(len(origin_by_raw_name))
