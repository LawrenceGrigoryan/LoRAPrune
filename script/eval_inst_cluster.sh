#!/usr/bin/env bash
#SBATCH --job-name=loraprune_qwen3-0.6b-chat-100k-granular-eval-instruction
#SBATCH --output=./logs/loraprune_qwen3-0.6b-chat-100k-granular-eval-instruction-%j.log
#SBATCH --error=./logs/loraprune_qwen3-0.6b-chat-100k-granular-eval-instruction-%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="./data/benchmarks/"

srun python eval_instruction.py \
    --model_id "./models/Qwen_Qwen3-0.6B" \
    --lora_weights "./outputs_dir/qwen3-0.6b-chat-100k-granular" \
    --batch_size 8 \
    --output_dir "./outputs_dir/evaluation/results/" \
    --granular_gqa True