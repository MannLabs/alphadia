#!/bin/bash
set -e -u
TEST_CASE_NAME=$1
ENV_NAME=$2
NEPTUNE_UPLOAD=${3:-False}
SHORT_SHA=${4:-sha_na}
BRANCH_NAME=${5:-branch_na}

cd e2e_tests

conda run -n $ENV_NAME --no-capture-output python prepare_test_data.py $TEST_CASE_NAME
ls */*

cat $TEST_CASE_NAME/config.yaml

TIMESTAMP_START=$(date +%s)
conda run -n $ENV_NAME --no-capture-output alphadia --config $TEST_CASE_NAME/config.yaml
ls */*

RUN_TIME=$(($(date +%s) - $TIMESTAMP_START))

conda run -n $ENV_NAME --no-capture-output python calc_metrics.py $TEST_CASE_NAME $RUN_TIME $NEPTUNE_UPLOAD $SHORT_SHA $BRANCH_NAME

cd -
