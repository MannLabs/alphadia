#!/bin/bash

ENV_NAME=${1:-alphadia}
TEST_TYPE=${2:-all}

case "$(echo $TEST_TYPE | tr '[:upper:]' '[:lower:]')" in
  "all")
    conda run -n $ENV_NAME --no-capture-output coverage run --source=../alphadia -m pytest
    ;;
  "integration")
    conda run -n $ENV_NAME --no-capture-output coverage run --source=../alphadia -m pytest integration_tests
    ;;
  "unit")
    conda run -n $ENV_NAME --no-capture-output coverage run --source=../alphadia -m pytest unit_tests
    ;;
esac

# coverage run --source=../alphadia -m pytest && coverage html && coverage-badge -f -o ../coverage.svg
# coverage run --source=../alphadia -m pytest -k 'not slow' && coverage html && coverage-badge -f -o ../coverage.svg
