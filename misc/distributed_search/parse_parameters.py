# Parse input parameters for distributed AlphaDIA search
# The objective is to pull as much logic from the outer and inner shell scripts 
# into a python script for easier reading/debugging. This script splits filepaths 
# into properly sized chunks, creates chunk folders, copies the spectral library 
# to each chunk folder*, writes an adapted config.yaml file for each chunk 
# (i.e. a config.yaml which contains the filepaths for that specific chunk 
# for running inner.sh).
# *Temporary solution to avoid simultaneous reads of the same library file.

import os
import sys
import shutil
import argparse
import pandas as pd
import yaml
import numpy as np

# parse input parameters
parser = argparse.ArgumentParser(
        prog = 'DistributedAlphaDIAParams',
        description = 'Parse input parameters into config files for chunked cluster processing of files with AlphaDIA')
parser.add_argument('--input_directory')
parser.add_argument('--input_filename')
parser.add_argument('--library_path')
parser.add_argument('--config_filename')
parser.add_argument('--target_directory')
parser.add_argument('--nnodes')
parser.add_argument('--reuse_quant')
args = parser.parse_args()

# read the input filename
infile = pd.read_csv(os.path.join(args.input_directory, args.input_filename), skiprows = 0)

# read the config .yaml file
with open(os.path.join(args.input_directory, args.config_filename), 'r') as file:
    config = yaml.safe_load(file)

# set requantition, False for searches, True for MBR, LFQ
config['general']['reuse_quant'] = True if args.reuse_quant == "1" else False

# library must be predicted/annotated prior to chunking
if 'library_prediction' not in config:
    config['library_prediction'] = {}
config['library_prediction']['predict'] = False

# remove any fasta if one is present in the config file
config.pop('fasta_list', None)

# determine chunk size: division of infile rowcount and number of nodes
chunk_size = int(np.ceil(infile.shape[0] / int(args.nnodes)))

# determine maximum number of tasks, i.e. the number of chunks needed
max_tasks = int(np.ceil(infile.shape[0] / chunk_size))

# split the filepaths into chunks
all_filepaths = infile.iloc[:,1].values
target_subdirectories = []
for i in range(0, max_tasks):
    
    # get current chunk indices
    start_idx = chunk_size * i
    end_idx = start_idx + chunk_size

    # save current chunk indices into yaml as raw files
    current_config = config
    current_config['raw_path_list'] = list(all_filepaths[start_idx:end_idx])

    # create folder for current chunk in target directory. Don't create the folder if it already exists.
    chunk_folder = os.path.join(args.target_directory, "chunk_" + str(i))
    if not os.path.exists(chunk_folder):
        os.makedirs(chunk_folder)

    # retrieve library path from config or arguments and copy to chunk folder, set new library path in config
    if os.path.exists(args.library_path) and os.path.basename(args.library_path).endswith('.hdf'):
        lib_source = args.library_path
    else:
        print("No valid library_path to a .hdf file provided and no valid library path to a .hdf file specified in config file, exiting...", file=sys.stderr)
        sys.exit(1)

    # copy library into chunk folder to avoid simultaneous reads of the same library
    shutil.copy(lib_source, chunk_folder)
    
    # set new library path in chunk config
    current_config['library'] = os.path.join(chunk_folder, os.path.basename(lib_source))

    # set chunk folder as output_directory in the config
    current_config['output_directory'] = "./"

    # save the current config into the target directory: this config contains all search parameters
    # and the rawfiles belonging to the current chunk.
    with open(os.path.join(chunk_folder, 'config.yaml'), 'w') as file:
        yaml.safe_dump(current_config, file, default_style=None, default_flow_style=False)

    # save the target subdirectory
    target_subdirectories.append(chunk_folder)

# return the list of target subdirectories
for target_subdirectory in target_subdirectories:
    print(target_subdirectory)
    
        


    
    
