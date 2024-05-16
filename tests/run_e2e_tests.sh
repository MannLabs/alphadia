#!/bin/bash

TEST_CASE_NAME=$1

cd e2e_tests || exit
conda activate alphadia


python prepare_test_data.py $TEST_CASE_NAME
ls *
ls */*

cat $TEST_CASE_NAME/config.yaml

alphadia --config $TEST_CASE_NAME/config.yaml

ls */*

conda deactivate
cd -
