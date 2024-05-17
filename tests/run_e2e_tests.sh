#!/bin/bash
set -e -u
TEST_CASE_NAME=$1

cd e2e_tests

python prepare_test_data.py $TEST_CASE_NAME
ls */*

cat $TEST_CASE_NAME/config.yaml

alphadia --config $TEST_CASE_NAME/config.yaml
ls */*

python calc_metrics.py $TEST_CASE_NAME

cd -
