#!/usr/bin/env bash
#SBATCH --job-name=loraprune_llama32-1B-80k-eval-perpl
#SBATCH --output=./logs/loraprune_llama32-1B-80k-eval-perpl-%j.log
#SBATCH --error=./logs/loraprune_llama32-1B-80k-eval-perpl-%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="./data/benchmarks/"

srun python ./eval_perplexity.py \
    --base_model "./models/meta_llama_Llama-3.2-1B" \
    --lora_weights "./outputs_dir/llama3.2-1b-base-80k" \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --granular_gqa False
