# Modify config.yaml file for spectral library prediction by removing any 
# rawfiles that may be set, setting prediction to True and adding 
# library/fasta paths. Since spectral library prediction should take place 
# with the same settings as the first search, the config filename is hard-coded.

import os
import yaml
import argparse

parser = argparse.ArgumentParser(
    prog = 'DistributedAlphaDIALibrary',
    description = 'Append fasta file to config for library prediction.')
parser.add_argument('--input_directory')
parser.add_argument('--target_directory')
parser.add_argument('--fasta_path')
parser.add_argument('--library_path')
parser.add_argument('--config_filename')
args = parser.parse_args()

# read the config.yaml file from the input directory
with open(os.path.join(args.input_directory, args.config_filename), 'r') as file:
    config = yaml.safe_load(file)

# if library and fasta are set, predicting will result in repredicted & annotated library
# add fasta_list to config
config['fasta_list'] = [args.fasta_path] if args.fasta_path else []

# add library path to config
config['library'] = args.library_path if args.library_path else None

# set library prediction to True
if 'library_prediction' not in config:
    config['library_prediction'] = {}
config['library_prediction']['predict'] = True

# remove rawfiles for prediction step in case some are set
config.pop('raw_path_list', None)

# set output directory for predicted spectral library
config['output_directory'] = os.path.join(args.target_directory)

# write speclib_config.yaml to input directory for the library prediction
with open(os.path.join(config['output_directory'], 'speclib_config.yaml'), 'w') as file:
    yaml.safe_dump(config, file, default_style=None, default_flow_style=False)
