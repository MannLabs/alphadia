#!/usr/bin/env bash
# A simple SLURM script to run AlphaDIA end2end tests on a SLURM cluster.
# Prerequisites:
# - conda environment with working AlphaDIA installation
# - input parameters (see below) set to desired values
# - (optional) SBATCH directives adapted to current use case
#SBATCH --job-name=alphadia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# input parameters:
CONDA_ENV=alphadia
BRANCH=main  # branch to take test cases from
TEST_CASE=basic

#
set -u -e
OUTPUT_FOLDER=output_${CONDA_ENV}_${TEST_CASE}_${SLURM_JOB_ID}
URL=https://raw.githubusercontent.com/MannLabs/alphadia/${BRANCH}/tests
export TQDM_MININTERVAL=10  # avoid lots of tqdm outputs

echo CONDA_ENV=$CONDA_ENV
echo BRANCH=$BRANCH
echo OUTPUT_FOLDER=$OUTPUT_FOLDER

echo CONDA_ENV ">>>>>>"
conda info
conda run -n $CONDA_ENV pip freeze
echo "<<<<<<"

echo MONO_VERSION ">>>>>>"
conda run -n $CONDA_ENV mono --version
echo "<<<<<<"

echo "Preparing test data.."

mkdir -p $OUTPUT_FOLDER && cd $OUTPUT_FOLDER

# note: if the locations of these files change, this script will need to be updated
wget $URL/run_e2e_tests.sh

mkdir -p e2e_tests && cd e2e_tests
wget $URL/e2e_tests/e2e_test_cases.yaml
wget $URL/e2e_tests/prepare_test_data.py
wget $URL/e2e_tests/calc_metrics.py
cd ..

chmod +x ./run_e2e_tests.sh

echo "Running alphadia.."
./run_e2e_tests.sh $TEST_CASE $CONDA_ENV
