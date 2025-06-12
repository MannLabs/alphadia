#!/bin/bash

ENV_NAME=${1:-alphadia}

NBS=$(find ./nbs/tutorial_nbs -name "*.ipynb" | grep -v "finetuning.ipynb")  # exclude finetuning notebook for it takes too long
conda run -n $ENV_NAME --no-capture-output python -m pytest --nbmake $(echo $NBS)
