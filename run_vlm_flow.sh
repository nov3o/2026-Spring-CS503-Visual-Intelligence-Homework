#!/usr/bin/env bash
# Usage: ./run_vlm_flow.sh {vlm | flow <config.yaml>}
set -e
N=$(nvidia-smi -L | wc -l)
case "$1" in
  vlm)  cd NanoVLM_Homework   && OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$N train.py ;;
  flow) cd NanoFM_Homeworks   && OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$N run_training.py --config "$2" ;;
  *)    echo "Usage: $0 {vlm | flow <config.yaml>}"; exit 1 ;;
esac
