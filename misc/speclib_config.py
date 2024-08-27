# modify config to include fasta for library prediction

import os
import yaml
import argparse

parser = argparse.ArgumentParser(
    prog = 'DistributedAlphaDIALibrary',
    description = 'Append fasta file to config for library prediction.')
parser.add_argument('--input_directory')
parser.add_argument('--fasta_filename')
args = parser.parse_args()

# read the config.yaml file from the input directory
with open(os.path.join(args.input_directory, "config.yaml"), 'r') as file:
    config = yaml.safe_load(file)

# remove rawfiles for conversion step
del config['raw_path_list']

# add fasta_list & set prediction
config['output_directory'] = args.input_directory
config['fasta_list'] = args.fasta_filename
config['library_prediction']['predict'] = True

# write yaml to input directory
with open(os.path.join(args.input_directory, 'speclib_config.yaml'), 'w') as file:
    yaml.safe_dump(config, file, default_style=None, default_flow_style=False)



