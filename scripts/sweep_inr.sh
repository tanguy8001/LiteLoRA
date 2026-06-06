#!/bin/bash
#SBATCH -t 1:00:00
# Lambda sweep for litelora on ImageNet-R.
# Each lambda gets its own isolated filepath to prevent checkpoint conflicts.
# Usage: bash scripts/sweep_inr.sh

CONFIG="./exps/litelora_inr.json"
BASE_PATH="./results/inr_lambda_sweep"
MASTER_RESULTS="results_lambda_inr_sweep.csv"

# Round 3: filling the 0.02->0.05 cliff + tighter resolution around the lower edge
LAMBDAS=(0.021 0.023 0.025 0.028 0.030 0.035 0.040 0.045)
LOGITS=(0.5)
SEEDS=(1993)

mkdir -p logs/sweep_inr

for LOGIT in "${LOGITS[@]}"
do
    for ((i=0; i<${#LAMBDAS[@]}; i+=1))
    do
        L="${LAMBDAS[$i]}"
        JOB_NAME="INR_G${LOGIT}_L${L}"
        LAMBDA_PATH="${BASE_PATH}/L${L}/"
        echo "Submitting: $JOB_NAME -> $LAMBDA_PATH"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=logs/sweep_inr/%j_$JOB_NAME.out
#SBATCH --error=logs/sweep_inr/%j_$JOB_NAME.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -p seas_gpu,gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=256G

cd "$SLURM_SUBMIT_DIR"
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
module load cuda/12.9.1-fasrc01
conda activate litelora

echo ">>> Starting INR sweep: Lambda=$L, Logit=$LOGIT"
python3 main.py --config $CONFIG \
               --lambda_sparsity $L \
               --init_logit $LOGIT \
               --init_alpha 1.0 \
               --epochs 20 \
               --seed ${SEEDS[0]} \
               --order_seed 1993 \
               --filepath $LAMBDA_PATH \
               --master_results $MASTER_RESULTS \
               --prefix "inr_sweep"
EOT
    done
done
