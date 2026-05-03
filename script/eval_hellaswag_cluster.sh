#!/usr/bin/env bash
#SBATCH --job-name=loraprune_qwen1.5-0.5b-hellaswag
#SBATCH --output=./logs/loraprune_qwen1.5-0.5b-hellaswag%j.log
#SBATCH --error=./logs/loraprune_qwen1.5-0.5b-hellaswag%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="./data/benchmarks/"

srun python eval_hellaswag.py \
    --model_id "./models/Qwen_Qwen1.5-0.5B" \
    --adapter_id "./outputs_dir/qwen15-05b-c4-10k/" \
    --n_shot 0 \
    --batch_size 8
