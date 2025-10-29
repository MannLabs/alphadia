#!/usr/bin/env bash

# Run alphadia with custom quantification directory. The point here is to use
# the slurm_index to find the proper chunk folder.

#SBATCH --job-name=alphaDIA
#SBATCH --time=21-00:00:00
#SBATCH --output=./logs/%A_%a_%x-slurm.out

# navigate to chunk directory
slurm_index=${SLURM_ARRAY_TASK_ID}
chunk_directory="${target_directory}/chunk_${slurm_index}/"
cd $chunk_directory || exit

# config file fixed as config.yaml
config_filename="config.yaml"

N_THREADS=$N_CPUS

# run with or without custom quant_dir
if [ -z "${quant_dir}" ]; then
    alphadia --config ${config_filename} --config-dict "{\"general\": {\"thread_count\": $N_THREADS}}"
else
    alphadia --config ${config_filename} --config-dict "{\"general\": {\"thread_count\": $N_THREADS}}" --quant-dir ${quant_dir}
fi

echo "AlphaDIA completed successfully"
