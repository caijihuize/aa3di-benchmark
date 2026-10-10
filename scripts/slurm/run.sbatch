#!/usr/bin/env bash
#SBATCH --job-name=aa3di-benchmark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=aa3di-%j.out
set -euo pipefail
# Site partition/account and the active Python environment are supplied by you.
# Example: sbatch --partition=YOUR_GPU_PARTITION scripts/slurm/run.sbatch \
#   --config configs/paper.yaml --output runs/prostt5 infer \
#   --method prostt5_decode_fp16_bs --dataset stage1_test474 --direction aa2di
exec python -m aa3di_benchmark.cli "$@"
