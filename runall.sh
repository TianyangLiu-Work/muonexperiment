#!/bin/bash

CONDA_ENV=/opt/anaconda3/envs/muonexperiment-torch
NOTEBOOK_WORKERS=5

find Notebooks -maxdepth 1 -name 'E*.ipynb' | sort | \
  xargs -n 1 -P "$NOTEBOOK_WORKERS" -I {} \
  "$CONDA_ENV/bin/jupyter" nbconvert \
    --to notebook \
    --execute "{}" \
    --inplace \
    --ExecutePreprocessor.timeout=-1

