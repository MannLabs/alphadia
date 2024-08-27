# Parse input parameters for distributed AlphaDIA search

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

# read the yaml file
with open(os.path.join(args.input_directory, args.config_filename), 'r') as file:
    config = yaml.safe_load(file)

# if reuse quant is set to 1, make no separate quant directory and set the corresponding arg in the config file
if args.reuse_quant == "1":
    config['general']['reuse_quant'] = True
else:
    config['general']['reuse_quant'] = False

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
    if args.library_path is not None:
        lib_source = args.library_path
    elif 'library' in current_config:
        lib_source = current_config['library']
    else:
        print("No library path provided in config or args, exiting...")
        sys.exit(1)
        
    shutil.copy(lib_source, chunk_folder)
    
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
    
        


    
    
