#!/usr/bin/env bash

#SBATCH --job-name=dist_AD
#SBATCH --time=21-00:00:00
#SBATCH --output=./logs/%j-%x-slurm.out

# slurm parameters
nnodes=1
ntasks_per_node=1
cpus=32
mem='250G'
# Search parameters
input_directory="/fs/home/brennsteiner/alphadia/distributed_search_test/"
input_filename="file_list.csv"
target_directory="/fs/home/brennsteiner/alphadia/distributed_search_test/search/"
library_path="/fs/home/brennsteiner/alphadia/distributed_search_test/48_fraction_hela_PaSk_orbitrap_ms2.hdf"
fasta_path="/fs/home/brennsteiner/alphadia/distributed_search_test/uniprotkb_proteome_UP000005640_2023_11_12.fasta"
first_search_config_filename="config.yaml"
second_search_config_filename="config.yaml"
# Search flags
predict_library=1
first_search=1
mbr_library=1
second_search=1
lfq=1

while [[ "$#" -gt 0 ]]; do
	case $1 in
		# Search parameters
		--input_directory) input_directory="$2"; shift ;;
		--input_filename) input_filename="$2"; shift ;;
		--target_directory) target_directory="$2"; shift ;;
		--library_path) library_path="$2"; shift ;;
		--fasta_path) fasta_path="$2"; shift ;;
		--first_search_config_filename) first_search_config_filename="$2"; shift ;;
		--second_search_config_filename) second_search_config_filename="$2"; shift ;;
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

	# generate timestamp in YYYMMDDHHMM format
	timestamp=$(date + "%Y%m%d%H%M")
	timestamp=""	

	# create logs directory if it does not exist
	mkdir -p ./logs

	# remove and create target directory
	rm -rf ${target_directory}
	mkdir ${target_directory}

	predicted_library_directory="${target_directory}predicted_speclib${timestamp}/"
	mkdir -p ${predicted_library_directory}

	first_search_directory="${target_directory}first_search/"
	mkdir -p ${first_search_directory}

	mbr_library_directory="${target_directory}mbr_library/"
	mkdir -p ${mbr_library_directory}
	
	mbr_progress_directory="${target_directory}mbr_library/chunk_0/.progress/"
	mkdir -p ${mbr_progress_directory}

	second_search_directory="${target_directory}second_search/"
	mkdir -p ${second_search_directory}

	lfq_directory="${target_directory}lfq/"
	mkdir -p ${lfq_directory}

	lfq_progress_directory="${target_directory}lfq/chunk_0/.progress/"
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

	# call alphadia to predict spectral library
	echo "Predicting spectral library with AlphaDIA"
	sbatch --array=${slurm_array} \
	--wait --nodes=1 \
	--ntasks-per-node=${ntasks_per_node} \
	--cpus-per-task=${cpus} \
	--mem=${mem} \
	--output="${home_directory}/logs/%j-%x-speclib-slurm.out" \
	--export=ALL --wrap="alphadia --config=speclib_config.yaml"

	# navigate back to home directory
	cd "${home_directory}"

	# if prediction took place, let the new speclib.hdf be the library path
	library_path="${predicted_library_directory}speclib.hdf"
else
	echo "Skipping library prediction"
fi

### FIRST SEARCH ###

if [[ "$first_search" -eq 1 ]]; then
	echo "Performing first search"
	
	# generate subdirectories for chunked first search
	first_search_subdirectories=$(python ./parse_parameters.py \
	--input_directory "${input_directory}" \
	--input_filename "${input_filename}" \
	--config_filename "${first_search_config_filename}" \
	--target_directory "${first_search_directory}" \
	--nnodes ${nnodes} \
	--reuse_quant 0 \
	--library_path ${library_path})

	# create slurm array for first search
	IFS=$'\n' read -d '' -r -a subdir_array <<< "$first_search_subdirectories"
	subdir_array_length=${#subdir_array[@]}
	slurm_array="0-$(($subdir_array_length - 1))%${nnodes}" # throttle is nnodes by definition

	# slurm passes the array index to the inner script, we add the target directory
	echo "Performing first search in ${subdir_array_length} chunks..."
	sbatch --array=${slurm_array} \
	--wait --nodes=1 \
	--ntasks-per-node=${ntasks_per_node} \
	--cpus-per-task=${cpus} \
	--mem=${mem} \
	--export=ALL,target_directory=${first_search_directory},custom_quant_dir=${mbr_progress_directory} ./inner.sh
else
	echo "Skipping first search"
fi

### MBR LIBRARY building ### --> simply reuse inner.sh with one chunk containing all files

if [[ "$mbr_library" -eq 1 ]]; then
	echo "Performing MBR library building"

	# set mbr library directory to the quant files from the first search
	mbr_library_subdirectories=$(python ./parse_parameters.py \
	--input_directory "${input_directory}" \
	--input_filename "${input_filename}" \
	--config_filename "${second_search_config_filename}" \
	--target_directory "${mbr_library_directory}" \
	--nnodes 1 \
	--reuse_quant 1 \
	--library_path ${library_path})

	# create slurm array with one subdir, which is the quant files from the first search
	slurm_array="0-0%1"

	# we force the single array to select the correct chunk and run the library building search
	echo "Performing library building search on all quant files from first search"
	sbatch --array=${slurm_array} \
	--wait --nodes=1 \
	--ntasks-per-node=${ntasks_per_node} \
	--cpus-per-task=${cpus} \
	--mem=${mem} \
	--export=ALL,target_directory=${mbr_library_directory} ./inner.sh
else
	echo "Skipping MBR library building"
fi

### SECOND SEARCH ###

if [[ "$second_search" -eq 1 ]]; then
	echo "Performing second search"

	second_search_subdirectories=$(python ./parse_parameters.py \
	--input_directory "${input_directory}" \
	--input_filename "${input_filename}" \
	--config_filename "${second_search_config_filename}" \
	--target_directory "${second_search_directory}" \
	--nnodes ${nnodes} \
	--reuse_quant 0 \
	--library_path "${mbr_library_directory}chunk_0/speclib.mbr.hdf")

	# create slurm array for second search
	IFS=$'\n' read -d '' -r -a subdir_array <<< "$second_search_subdirectories"
	subdir_array_length=${#subdir_array[@]}
	slurm_array="0-$(($subdir_array_length - 1))%${nnodes}" # throttle is nnodes by definition

	# slurm passes the array index to the inner script, we add the target directory
	echo "Performing second search in ${subdir_array_length} chunks..."
	sbatch --array=${slurm_array} \
	--wait --nodes=1 \
	--ntasks-per-node=${ntasks_per_node} \
	--cpus-per-task=${cpus} \
	--mem=${mem} \
	--export=ALL,target_directory=${second_search_directory},custom_quant_dir=${lfq_progress_directory} ./inner.sh
else
	echo "Skipping second search"
fi

### LFQ ###

if [[ "$lfq" -eq 1 ]]; then
	echo "Performing LFQ"

	# set lfq directory to the quant files from the second search
	lfq_subdirectories=$(python ./parse_parameters.py \
	--input_directory "${input_directory}" \
	--input_filename "${input_filename}" \
	--config_filename "${second_search_config_filename}" \
	--target_directory "${lfq_directory}" \
	--nnodes 1 \
	--reuse_quant 1 \
	--library_path "${mbr_library_directory}chunk_0/speclib.mbr.hdf")

	# create slurm array with one subdir, which is the quant files from the second search
	slurm_array="0-0%1"

	# we force the single array to select the correct chunk and run the library building search
	echo "Performing LFQ on all quant files from second search"
	sbatch --array=${slurm_array} \
	--wait --nodes=1 \
	--ntasks-per-node=${ntasks_per_node} \
	--cpus-per-task=${cpus} \
	--mem=${mem} \
	--export=ALL,target_directory=${lfq_directory} ./inner.sh
else
	echo "Skipping LFQ"
fi

