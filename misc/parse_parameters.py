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
parser.add_argument('--num_candidates')
parser.add_argument('--rt_tolerance')
parser.add_argument('--inference_strategy')
args = parser.parse_args()

# read the input filename
infile = pd.read_csv(os.path.join(args.input_directory, args.input_filename), skiprows = 0)

# read the config .yaml file
with open(os.path.join(args.input_directory, args.config_filename), 'r') as file:
    config = yaml.safe_load(file)

# if reuse quant is set to 1, make no separate quant directory and set the corresponding arg in the config file
if args.reuse_quant == "1":
    config['general']['reuse_quant'] = True
else:
    config['general']['reuse_quant'] = False

# always set prediction of library to false, this should happen prior to chunking
config['library_prediction']['predict'] = False

# throw out fasta at this point, library must be fully predicted prior to chunking
try:
    del config['fasta_list']
except:
    pass

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
    elif 'library' in current_config and os.path.exists(current_config['library']) and os.path.basename(current_config['library']).endswith('.hdf'):
        lib_source = current_config['library']
    else:
        print("No valid library_path to a .hdf file provided and no valid library path to a .hdf file specified in config file, exiting...")
        sys.exit(1)

    # copy library into chunk folder to avoid simultaneous reads of the same library
    shutil.copy(lib_source, chunk_folder)
    
    # set new library path in chunk config
    current_config['library'] = os.path.join(chunk_folder, os.path.basename(lib_source))

    # set chunk folder as output_directory in the config
    current_config['output_directory'] = "./"

    # set number of candidates
    current_num_candidates = current_config['search']['target_num_candidates'] if 'target_num_candidates' in current_config['search'] else 3
    current_config['search']['target_num_candidates'] = int(args.num_candidates) if args.num_candidates else current_num_candidates

    # set rt tolerance
    current_rt_tolerance = current_config['search']['target_rt_tolerance'] if 'target_rt_tolerance' in current_config['search'] else 200
    current_config['search']['target_rt_tolerance'] = float(args.rt_tolerance) if args.rt_tolerance else current_rt_tolerance

    # set inference strategy
    current_inf_strategy = current_config['fdr']['inference_strategy'] if 'inference_strategy' in current_config['fdr'] else 'heuristic'
    current_config['fdr']['inference_strategy'] = args.inference_strategy if args.inference_strategy else current_inf_strategy

    # save the current config into the target directory: this config contains all search parameters
    # and the rawfiles belonging to the current chunk.
    with open(os.path.join(chunk_folder, 'config.yaml'), 'w') as file:
        yaml.safe_dump(current_config, file, default_style=None, default_flow_style=False)

    # save the target subdirectory
    target_subdirectories.append(chunk_folder)

# return the list of target subdirectories
for target_subdirectory in target_subdirectories:
    print(target_subdirectory)
    
        


    
    
