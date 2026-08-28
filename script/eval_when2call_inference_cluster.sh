#!/usr/bin/env bash
#SBATCH --job-name=loraprune_qwen3-0.6b-when2call-inference
#SBATCH --output=./logs/loraprune_qwen3-0.6b-when2call-inference-%j.log
#SBATCH --error=./logs/loraprune_qwen3-0.6b-when2call-inference-%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="./data/benchmarks/"

srun python ./eval_when2call_inference.py \
    --base_model "./models/Qwen_Qwen3-0.6B-Base" \
    --lora_weights "./outputs_dir/qwen3-0.6b-base-10k-granular" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --output_dir "./outputs_dir/evaluation/results/" \
    --granular_gqa True \
    --batch_size 8 \
    --max_new_tokens 256 \
    --seed 42
