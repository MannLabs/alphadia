# modify config to include fasta for library prediction

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
args = parser.parse_args()

# read the config.yaml file from the input directory
with open(os.path.join(args.input_directory, "config.yaml"), 'r') as file:
    config = yaml.safe_load(file)

# if library and fasta are set, predicting will result in repredicted & annotated library
# add fasta_list to config
config['fasta_list'] = [args.fasta_path] if args.fasta_path else []

# add library path to config
config['library_path'] = args.library_path if args.library_path else None

# set library prediction to True
config['library_prediction']['predict'] = True

# remove rawfiles for prediction step in case some are set
config.pop('raw_path_list', None)

# set output directory for predicted spectral library
config['output_directory'] = os.path.join(args.target_directory, 'predicted_speclib/')

# write speclib_config.yaml to input directory for the library prediction
with open(os.path.join(config['output_directory'], 'speclib_config.yaml'), 'w') as file:
    yaml.safe_dump(config, file, default_style=None, default_flow_style=False)
