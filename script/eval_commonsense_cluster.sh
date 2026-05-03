#!/usr/bin/env bash
#SBATCH --job-name=loraprune_qwen1.5-0.5b-mmlu
#SBATCH --output=./logs/loraprune_qwen1.5-0.5b-mmlu%j.log
#SBATCH --error=./logs/loraprune_qwen1.5-0.5b-mmlu%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="./data/benchmarks/"

srun python eval_commonsense.py \
    --model_id "./models/Qwen_Qwen1.5-0.5B" \
    # --adapter_id "./outputs_dir/qwen15-05b-c4-100k/" \
    --n_shot 5 \
    --batch_size 8 \
    --output_dir "../output_dir/evaluation/results/"
