#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

TEMPORAL_RELATION = "after"
EXPT_NAME = "r1_eval_cat" + "_" + TEMPORAL_RELATION

python expt_scripts/r1_eval_cat.py
