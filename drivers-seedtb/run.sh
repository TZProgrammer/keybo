#!/bin/bash
# Pin ALL FOUR thread vars BEFORE python starts (inert after xgboost import).
export OMP_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export MKL_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48
export PYTHONPATH=/local/home/zegertho/agent/workspaces/seedtb/wt/src
exec /local/home/zegertho/repos/keybo/.venv/bin/python "$@"
