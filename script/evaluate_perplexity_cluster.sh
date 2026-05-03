#!/usr/bin/env bash
#SBATCH --job-name=loraprune_test
#SBATCH --output=loraprune_test%j.log
#SBATCH --error=loraprune_test%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

srun python inference.py \
    --base_model "./models/meta-llama_Llama-3.2-1B" \
    --lora_weights 'outputs_dir' \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
