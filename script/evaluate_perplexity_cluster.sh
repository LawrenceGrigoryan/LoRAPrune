#!/usr/bin/env bash
#SBATCH --job-name=loraprune_perpl_eval
#SBATCH --output=./logs/loraprune_perpl_eval_%j.log
#SBATCH --error=./logs/loraprune_perpl_eval_%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="./data/benchmarks/"

srun python ./eval_perplexity.py \
    --base_model "./models/Qwen_Qwen3-0.6B-Base" \
    --lora_weights "./outputs_dir/qwen3-0.6b-base-10k-granular" \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --granular_gqa True
