#!/bin/bash
# Lambda sweep for litelora on CUB-200.
# Usage: bash scripts/sweep_cub.sh

CONFIG="./exps/litelora_cub200.json"
BASE_PATH="./results/cub_lambda_sweep/"
MASTER_RESULTS="results_lambda_cub_sweep.csv"

LAMBDAS=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7 1.0)
LOGITS=(0.5)
SEEDS=(1993)

mkdir -p logs/sweep_cub

for LOGIT in "${LOGITS[@]}"
do
    for ((i=0; i<${#LAMBDAS[@]}; i+=3))
    do
        CHUNK=("${LAMBDAS[@]:i:3}")
        JOB_NAME="CUB_G${LOGIT}_$((i/3))"
        echo "Submitting: $JOB_NAME with Lambdas: ${CHUNK[*]}"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=logs/sweep_cub/%j_$JOB_NAME.out
#SBATCH --error=logs/sweep_cub/%j_$JOB_NAME.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH -p seas_gpu,gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=256G

cd "$SLURM_SUBMIT_DIR"
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
module load cuda/12.9.1-fasrc01
conda activate litelora
nvidia-smi

for L in ${CHUNK[@]}
do
    echo ">>> Starting CUB sweep: Lambda=\$L, Logit=$LOGIT"
    python3 main.py --config $CONFIG \
                   --lambda_sparsity \$L \
                   --init_logit $LOGIT \
                   --init_alpha 1.0 \
                   --epochs 20 \
                   --seed ${SEEDS[@]} \
                   --order_seed 1993 \
                   --filepath $BASE_PATH \
                   --master_results $MASTER_RESULTS \
                   --prefix "cub_sweep"
done
EOT
    done
done
