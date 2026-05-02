#!/usr/bin/env bash
#SBATCH --job-name=loraprune_qwen1.5-0.5b-mmlu
#SBATCH --output=loraprune_qwen1.5-0.5b-mmlu%j.log
#SBATCH --error=loraprune_qwen1.5-0.5b-mmlu%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

srun python eval_mmlu.py \
    --base_model "./models/Qwen_Qwen1.5-0.5B" \
    --lora_weights "./outputs_dir/qwen15-05b-c4-10k/" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \