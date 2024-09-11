# modify config to include fasta for library prediction

import os
import yaml
import argparse

parser = argparse.ArgumentParser(
    prog = 'DistributedAlphaDIALibrary',
    description = 'Append fasta file to config for library prediction.')
parser.add_argument('--input_directory')
parser.add_argument('--target_directory')
args = parser.parse_args()

# read the config.yaml file from the input directory
with open(os.path.join(args.input_directory, "config.yaml"), 'r') as file:
    config = yaml.safe_load(file)

# remove rawfiles for conversion step
try: 
    del config['raw_path_list']
except:
    pass

# add fasta_list & set prediction
config['output_directory'] = os.path.join(args.target_directory, 'predicted_speclib/')
config['library_prediction']['predict'] = True

# write speclib_config.yaml to input directory for the library prediction
with open(os.path.join(config['output_directory'], 'speclib_config.yaml'), 'w') as file:
    yaml.safe_dump(config, file, default_style=None, default_flow_style=False)
