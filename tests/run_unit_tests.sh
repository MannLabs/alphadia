#!/bin/bash

ENV_NAME=${1:-alphadia}
TEST_TYPE=${2:-fast}

case "$(echo $TEST_TYPE | tr '[:upper:]' '[:lower:]')" in
  "all"|"true")
    conda run -n $ENV_NAME --no-capture-output coverage run --source=../alphadia -m pytest
    ;;
  "slow")
    conda run -n $ENV_NAME --no-capture-output coverage run --source=../alphadia -m pytest -k 'slow'
    ;;
  "fast"|"false"|*)
    conda run -n $ENV_NAME --no-capture-output coverage run --source=../alphadia -m pytest -k 'not slow'
    ;;
esac

# coverage run --source=../alphadia -m pytest && coverage html && coverage-badge -f -o ../coverage.svg
# coverage run --source=../alphadia -m pytest -k 'not slow' && coverage html && coverage-badge -f -o ../coverage.svg
