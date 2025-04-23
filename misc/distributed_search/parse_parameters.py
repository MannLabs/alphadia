# Parse input parameters for distributed AlphaDIA search
# The objective is to pull as much logic from the outer and inner shell scripts
# into a python script for easier reading/debugging. This script splits filepaths
# into properly sized chunks, creates chunk folders, copies the spectral library
# to each chunk folder*, writes an adapted config.yaml file for each chunk
# (i.e. a config.yaml which contains the filepaths for that specific chunk
# for running inner.sh).
# *Temporary solution to avoid simultaneous reads of the same library file.

import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import yaml


# Add keys to config if they don't exist
def safe_add_key(config, parent_key, key, value):
    """Safely add keys and values to config file dictionary"""
    if parent_key is None:
        config[key] = value
    else:
        if parent_key not in config:
            config[parent_key] = {}
        config[parent_key][key] = value


# parse input parameters
parser = argparse.ArgumentParser(
    prog="DistributedAlphaDIAParams",
    description="Parse input parameters into config files for chunked cluster processing of files with AlphaDIA",
)
parser.add_argument("--input_directory")
parser.add_argument("--input_filename")
parser.add_argument("--library_path")
parser.add_argument("--config_filename")
parser.add_argument("--target_directory")
parser.add_argument("--nnodes")
parser.add_argument("--reuse_quant")
args = parser.parse_args()

# read the input filename
infile = pd.read_csv(
    os.path.join(args.input_directory, args.input_filename), skiprows=0
)

# read the config .yaml file
with open(os.path.join(args.input_directory, args.config_filename)) as file:
    config = yaml.safe_load(file) or {}

# set requantition, False for searches, True for MBR, LFQ
safe_add_key(config, "general", "reuse_quant", args.reuse_quant == "1")

# library must be predicted/annotated prior to chunking
safe_add_key(config, "library_prediction", "predict", False)

# remove any fasta if one is present in the config file
config.pop("fasta_paths", None)

# determine chunk size: division of infile rowcount and number of nodes
chunk_size = int(np.ceil(infile.shape[0] / int(args.nnodes)))

# determine maximum number of tasks, i.e. the number of chunks needed
max_tasks = int(np.ceil(infile.shape[0] / chunk_size))

# split the filepaths into chunks
all_filepaths = infile.iloc[:, 1].values
target_subdirectories = []
for i in range(0, max_tasks):
    # get current chunk indices
    start_idx = chunk_size * i
    end_idx = start_idx + chunk_size

    # copy original config for the current chunk
    current_config = copy.deepcopy(config)

    # save current chunk indices into chunk-yaml as raw files
    safe_add_key(
        current_config, None, "raw_paths", list(all_filepaths[start_idx:end_idx])
    )

    # create folder for current chunk in target directory. Don't create the folder if it already exists.
    chunk_folder = os.path.join(args.target_directory, "chunk_" + str(i))
    os.makedirs(chunk_folder, exist_ok=True)

    # retrieve library path from config or arguments, set new library path in config
    if os.path.exists(args.library_path) and os.path.basename(
        args.library_path
    ).endswith(".hdf"):
        lib_source = args.library_path
    else:
        print(
            "No valid library_path to a .hdf file provided and no valid library path to a .hdf file specified in config file, exiting...",
            file=sys.stderr,
        )
        sys.exit(1)

    # set library path in config
    safe_add_key(current_config, None, "library_path", lib_source)

    # set chunk folder as output_directory in the config
    safe_add_key(current_config, None, "output_directory", "./")

    # save the config with the current chunk's rawfiles belonging to the current chunk folder
    with open(os.path.join(chunk_folder, "config.yaml"), "w") as file:
        yaml.safe_dump(
            current_config, file, default_style=None, default_flow_style=False
        )

    # save the target subdirectory
    target_subdirectories.append(chunk_folder)

# the only return value needed is number of tasks to be distributed by the scheduler
print(max_tasks)
