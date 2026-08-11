#!/usr/bin/env bash
#SBATCH --job-name=loraprune_qwen1.5-0.5b-100k-eval-commonsense
#SBATCH --output=./logs/loraprune_qwen1.5-0.5b-100k-eval-commonsense-%j.log
#SBATCH --error=./logs/loraprune_qwen1.5-0.5b-100k-eval-commonsense-%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="./data/benchmarks/"

srun python eval_commonsense.py \
    --model_id "./models/Qwen_Qwen1.5-0.5B" \
    --adapter_id "./outputs_dir/qwen1.5-0.5b-base-100k/" \
    --batch_size 8 \
    --output_dir "./outputs_dir/evaluation/results/" \
    --granular_gqa False