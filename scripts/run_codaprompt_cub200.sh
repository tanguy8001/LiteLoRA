#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH -J cp_cub200
#SBATCH -o logs/cp_cub200/cp_cub200_%j.out
#SBATCH -e logs/cp_cub200/cp_cub200_%j.err
#SBATCH -p seas_gpu,gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=256G

#  sbatch scripts/run_codaprompt_cub200.sh --seed 1993 --order_seed 1993
#  sbatch scripts/run_codaprompt_cub200.sh --seed 1995 --order_seed 1995
#  sbatch scripts/run_codaprompt_cub200.sh --seed 2000 --order_seed 2000

cd "$SLURM_SUBMIT_DIR"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
module load cuda/12.9.1-fasrc01
conda activate litelora
nvidia-smi

python3 main.py --config=./exps/coda_prompt_cub200.json "$@"