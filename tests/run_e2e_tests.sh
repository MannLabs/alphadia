#!/bin/bash
set -e -u
TEST_CASE_NAME=$1
ENV_NAME=$2

cd e2e_tests

conda run -n $ENV_NAME python prepare_test_data.py $TEST_CASE_NAME
ls */*

cat $TEST_CASE_NAME/config.yaml

conda run -n $ENV_NAME alphadia --config $TEST_CASE_NAME/config.yaml
ls */*

conda run -n $ENV_NAME python calc_metrics.py $TEST_CASE_NAME

cd -
