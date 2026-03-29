#!/usr/bin/env bash
#SBATCH --job-name=loraprune_llama_32_1b_c4_20k
#SBATCH --output=loraprune_test%j.log
#SBATCH --error=loraprune_test%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

srun python eval_when2call.py \
    --base_model "./models/Qwen_Qwen1.5-0.5B-Chat" \
    --data_path './data/benchmarks/nvidia___when2_call' \
    --lora_weights "./outputs_dir/qwen15_05b_chat_lamini_20k/" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --output_path "./evaluation/results/qwen15-05b-chat-lamini-20k/when2call_results.jsonl"