#!/usr/bin/env bash

# Main script, sets source and destination folders, which csv file to read
# in order to get rawfile paths. Walks through the steps of a distributed
# search, waiting for each one to be finished before initializing
# the next one. For searches, the "inner.sh" script is called. For predicting
# speclibs, alphadia is run directly without rawfiles.

#SBATCH --job-name=dist_AD
#SBATCH --time=21-00:00:00
#SBATCH --output=./logs/%j-%x-slurm.out

# Set behavior when errors are encountered
# add -x for debugging
set -u -e

# Default search parameters
nnodes=1
ntasks_per_node=1
cpus=32
mem='250G'

# Default search flags
predict_library=1
first_search=1
mbr_library=1
second_search=1
lfq=1

# Default search configuration file
search_config="search.config"

while [[ "$#" -gt 0 ]]; do
	case $1 in
		# Search parameters
		--search_config) search_config="$2"; shift ;;
		# SLURM parameters
		--nnodes) nnodes="$2"; shift ;;
		--ntasks_per_node) ntasks_per_node="$2"; shift ;;
		--cpus) cpus="$2"; shift ;;
		--mem) mem="$2"; shift ;;
		# Search flags: if a flag is present, set to true
		--predict_library) predict_library="$2"; shift ;;
		--first_search) first_search="$2"; shift ;;
		--mbr_library) mbr_library="$2"; shift ;;
		--second_search) second_search="$2"; shift ;;
		--lfq) lfq="$2"; shift ;;
		*) echo "Unknown parameter $1"; shift ;;
	esac
	shift
done

# Source search configuration
if [[ -n "${search_config}" ]]; then
	source ./${search_config}
	echo "Using search config: ${search_config}"
else
	echo "No search config provided, exiting"
	exit 1
fi

# report set parameters
echo "Search config: ${search_config}"
echo "SLURM parameters: nnodes=${nnodes}, ntasks_per_node=${ntasks_per_node}, cpus=${cpus}, mem=${mem}"
echo "Search flags: predict_library=${predict_library}, first_search=${first_search}, mbr_library=${mbr_library}, second_search=${second_search}, lfq=${lfq}"

# Derive output directory name from CSV filename (without .csv extension)
csv_basename=$(basename "${input_filename}" .csv)
output_directory="${target_directory}/ad_search_${csv_basename}"

# Create output directory (ok if exists) but fail if not empty
mkdir -p "${output_directory}"
if [ "$(ls -A ${output_directory})" ]; then
	echo "Output directory ${output_directory} is not empty, exiting"
	exit 1
fi

# create logs directory if it does not exist
mkdir -p ./logs

# use output_directory instead of target_directory for all subsequent paths
target_directory="${output_directory}"

predicted_library_directory="${target_directory}/1_predicted_speclib"
mkdir -p ${predicted_library_directory}

first_search_directory="${target_directory}/2_first_search"
mkdir -p ${first_search_directory}

mbr_library_directory="${target_directory}/3_mbr_library"
mkdir -p ${mbr_library_directory}

mbr_progress_directory="${target_directory}/3_mbr_library/chunk_0/quant"
mkdir -p ${mbr_progress_directory}

second_search_directory="${target_directory}/4_second_search"
mkdir -p ${second_search_directory}

lfq_directory="${target_directory}/5_lfq"
mkdir -p ${lfq_directory}

lfq_progress_directory="${target_directory}/5_lfq/chunk_0/quant"
mkdir -p ${lfq_progress_directory}

### PREDICT LIBRARY ###

if [[ "$predict_library" -eq 1 ]]; then

	# generate config without rawfiles and with fasta
	python ./speclib_config.py \
	--input_directory "${input_directory}" \
	--target_directory "${predicted_library_directory}" \
	--fasta_path "${fasta_path}" \
	--library_path "${library_path}" \
	--config_filename "${first_search_config_filename}" \

	# log current directory and navigate to predicted speclib directory
	home_directory=$(pwd)
	cd "${predicted_library_directory}"

	# call alphadia to predict spectral library as one task
	echo "Predicting spectral library with AlphaDIA..."
	sbatch --array="0-0%1" \
	--wait \
	--nodes=1 \
	--ntasks-per-node=${ntasks_per_node} \
	--cpus-per-task=${cpus} \
	--mem=${mem} \
	--output="${home_directory}/logs/%j-%x-speclib-slurm.out" \
	--export=ALL,N_CPUS=$cpus --wrap="alphadia --config=speclib_config.yaml"

	# navigate back to home directory
	cd "${home_directory}"

	# if prediction took place, let the new speclib.hdf be the library path
	library_path="${predicted_library_directory}/speclib.hdf"
else
	echo "Skipping library prediction"
fi

### FIRST SEARCH ###

if [[ "$first_search" -eq 1 ]]; then

	# generate subdirectories for chunked first search
	num_tasks=$(python ./parse_parameters.py \
	--input_directory "${input_directory}" \
	--input_filename "${input_filename}" \
	--config_filename "${first_search_config_filename}" \
	--target_directory "${first_search_directory}" \
	--nnodes ${nnodes} \
	--reuse_quant 0 \
	--library_path ${library_path})

	# create slurm array for first search
	slurm_array="0-$(($num_tasks-1))%${nnodes}"
	echo "First search array: ${slurm_array}"

	# slurm passes the array index to the inner script, we add the target directory
	echo "Performing first search in ${num_tasks} chunks..."
	sbatch --array=${slurm_array} \
	--wait \
	--nodes=1 \
	--ntasks-per-node=${ntasks_per_node} \
	--cpus-per-task=${cpus} \
	--mem=${mem} \
	--export=ALL,N_CPUS=$cpus,target_directory=${first_search_directory},quant_dir=${mbr_progress_directory} ./inner.sh
else
	echo "Skipping first search"
fi

### MBR LIBRARY building ### --> simply reuse inner.sh with one chunk containing all files

if [[ "$mbr_library" -eq 1 ]]; then

	# set mbr library directory to the quant files from the first search
	num_tasks=$(python ./parse_parameters.py \
	--input_directory "${input_directory}" \
	--input_filename "${input_filename}" \
	--config_filename "${second_search_config_filename}" \
	--target_directory "${mbr_library_directory}" \
	--nnodes 1 \
	--reuse_quant 1 \
	--library_path ${library_path})

	# create slurm array with one subdir, which is the quant files from the first search
	slurm_array="0-$(($num_tasks-1))%${nnodes}"
	echo "MBR library array: ${slurm_array}"

	# we force the single array to select the correct chunk and run the library building search
	echo "Performing MBR library building in ${num_tasks} chunks on all quant files from first search..."
	sbatch --array=${slurm_array} \
	--wait \
	--nodes=1 \
	--ntasks-per-node=${ntasks_per_node} \
	--cpus-per-task=${cpus} \
	--mem=${mem} \
	--export=ALL,N_CPUS=$cpus,target_directory=${mbr_library_directory} ./inner.sh
else
	echo "Skipping MBR library building"
fi

### SECOND SEARCH ###

if [[ "$second_search" -eq 1 ]]; then

	num_tasks=$(python ./parse_parameters.py \
	--input_directory "${input_directory}" \
	--input_filename "${input_filename}" \
	--config_filename "${second_search_config_filename}" \
	--target_directory "${second_search_directory}" \
	--nnodes ${nnodes} \
	--reuse_quant 0 \
	--library_path "${mbr_library_directory}/chunk_0/speclib.mbr.hdf")

	# create slurm array for second search
	slurm_array="0-$(($num_tasks-1))%${nnodes}"
	echo "Second search array: ${slurm_array}"

	# slurm passes the array index to the inner script, we add the target directory
	echo "Performing second search in ${num_tasks} chunks..."
	sbatch --array=${slurm_array} \
	--wait \
	--nodes=1 \
	--ntasks-per-node=${ntasks_per_node} \
	--cpus-per-task=${cpus} \
	--mem=${mem} \
	--export=ALL,N_CPUS=$cpus,target_directory=${second_search_directory},quant_dir=${lfq_progress_directory} ./inner.sh
else
	echo "Skipping second search"
fi

### LFQ ###

if [[ "$lfq" -eq 1 ]]; then

	# set lfq directory to the quant files from the second search
	num_tasks=$(python ./parse_parameters.py \
	--input_directory "${input_directory}" \
	--input_filename "${input_filename}" \
	--config_filename "${second_search_config_filename}" \
	--target_directory "${lfq_directory}" \
	--nnodes 1 \
	--reuse_quant 1 \
	--library_path "${mbr_library_directory}/chunk_0/speclib.mbr.hdf")

	# create slurm array with one subdir, which is the quant files from the second search
	slurm_array="0-$(($num_tasks-1))%${nnodes}"
	echo "LFQ array: ${slurm_array}"

	# we force the single array to select the correct chunk and run the library building search
	echo "Performing LFQ in ${num_tasks} chunks on all quant files from second search..."
	sbatch --array=${slurm_array} \
	--wait \
	--nodes=1 \
	--ntasks-per-node=${ntasks_per_node} \
	--cpus-per-task=${cpus} \
	--mem=${mem} \
	--export=ALL,N_CPUS=$cpus,target_directory=${lfq_directory} ./inner.sh
else
	echo "Skipping LFQ"
fi
