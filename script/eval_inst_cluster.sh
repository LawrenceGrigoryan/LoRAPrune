#!/usr/bin/env bash
#SBATCH --job-name=loraprune_qwen1.5-0.5b-chat-eval-instruction
#SBATCH --output=./logs/loraprune_qwen1.5-0.5b-chat-eval-instruction-%j.log
#SBATCH --error=./logs/loraprune_qwen1.5-0.5b-chat-eval-instruction-%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="./data/benchmarks/"

srun python eval_instruction.py \
    --model_id "./models/Qwen_Qwen1.5-0.5B-Chat" \
    # --adapter_id "./outputs_dir/qwen15-05b-c4-80k/" \
    --batch_size 8 \
    --output_dir "../output_dir/evaluation/results/"
