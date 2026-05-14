#!/usr/bin/env bash
#SBATCH --job-name=loraprune_qwen1.5-0.5b-eval-tool
#SBATCH --output=./logs/loraprune_qwen1.5-0.5b-eval-tool-%j.log
#SBATCH --error=./logs/loraprune_qwen1.5-0.5b-eval-tool-%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

srun python eval_when2call.py \
    --base_model "./models/Qwen_Qwen1.5-0.5B-Chat" \
    # --lora_weights "./outputs_dir/qwen15_05b_chat_lamini_20k/" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --output_path "./outputs_dir/evaluation/results"