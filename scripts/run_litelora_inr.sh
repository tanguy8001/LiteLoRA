#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -J litelora_inr
#SBATCH -o logs/litelora_inr/litelora_inr_%j.out
#SBATCH -e logs/litelora_inr/litelora_inr_%j.err
#SBATCH -p seas_gpu,gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=256G

#  sbatch scripts/run_litelora_inr.sh --seed 1993 --order_seed 1993
#  sbatch scripts/run_litelora_inr.sh --seed 1995 --order_seed 1995
#  sbatch scripts/run_litelora_inr.sh --seed 2000 --order_seed 2000

cd "$SLURM_SUBMIT_DIR"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
module load cuda/12.9.1-fasrc01
conda activate litelora
nvidia-smi

python3 main.py --config=./exps/litelora_inr.json "$@"
