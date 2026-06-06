#!/bin/bash
# Lambda sweep for litelora on CIFAR-100.
# Usage: bash scripts/sweep_c100.sh

CONFIG="./exps/litelora_c100.json"
BASE_PATH="./results/c100_lambda_sweep/"
MASTER_RESULTS="results_lambda_c100_sweep.csv"

LAMBDAS=(0.005 0.008 0.01 0.012 0.014 0.016 0.02 0.025)
LOGITS=(0.5)
SEEDS=(1993)

mkdir -p logs/sweep_c100

for LOGIT in "${LOGITS[@]}"
do
    for ((i=0; i<${#LAMBDAS[@]}; i+=3))
    do
        CHUNK=("${LAMBDAS[@]:i:3}")
        JOB_NAME="C100_G${LOGIT}_$((i/3))"
        echo "Submitting: $JOB_NAME with Lambdas: ${CHUNK[*]}"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=logs/sweep_c100/%j_$JOB_NAME.out
#SBATCH --error=logs/sweep_c100/%j_$JOB_NAME.err
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

for L in ${CHUNK[@]}
do
    echo ">>> Starting C100 sweep: Lambda=\$L, Logit=$LOGIT"
    python3 main.py --config $CONFIG \
                   --lambda_sparsity \$L \
                   --init_logit $LOGIT \
                   --init_alpha 1.0 \
                   --epochs 20 \
                   --seed ${SEEDS[@]} \
                   --order_seed 1993 \
                   --filepath $BASE_PATH \
                   --master_results $MASTER_RESULTS \
                   --prefix "c100_sweep"
done
EOT
    done
done
