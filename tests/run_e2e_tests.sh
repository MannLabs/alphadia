#!/bin/bash
set -e

eval "$(conda shell.bash hook)"


TEST_CASE_NAME=$1
conda activate alphadia

cd e2e_tests || exit




python prepare_test_data.py $TEST_CASE_NAME
ls *
ls */*

cat $TEST_CASE_NAME/config.yaml

echo which alphadia
which alphadia

alphadia --config $TEST_CASE_NAME/config.yaml

ls */*

conda deactivate
cd -
