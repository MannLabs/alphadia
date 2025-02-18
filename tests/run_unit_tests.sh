#!/bin/bash

ENV_NAME=${1:-alphadia}
INCLUDE_SLOW_TESTS=${2:-false}

if [ "${INCLUDE_SLOW_TESTS,,}" = "true" ]; then
  conda run -n $ENV_NAME --no-capture-output coverage run --source=../alphadia -m pytest
else
  conda run -n $ENV_NAME --no-capture-output coverage run --source=../alphadia -m pytest -k 'not slow'
fi

# coverage run --source=../alphadia -m pytest && coverage html && coverage-badge -f -o ../coverage.svg
# coverage run --source=../alphadia -m pytest -k 'not slow' && coverage html && coverage-badge -f -o ../coverage.svg
