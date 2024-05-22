#!/bin/bash
set -e -u
TEST_CASE_NAME=$1
ENV_NAME=$2
SHORT_SHA=${3:-sha_na}
BRANCH_NAME=${4:-branch_na}

cd e2e_tests

conda run -n $ENV_NAME python prepare_test_data.py $TEST_CASE_NAME
ls */*

cat $TEST_CASE_NAME/config.yaml

conda run -n $ENV_NAME --no-capture-output alphadia --config $TEST_CASE_NAME/config.yaml
ls */*

conda run -n $ENV_NAME --no-capture-output python calc_metrics.py $TEST_CASE_NAME $SHORT_SHA $BRANCH_NAME

cd -
