#!/usr/bin/env bash
#SBATCH --job-name=loraprune_qwen3-0.6b-chat-100k-granular
#SBATCH --output=./logs/loraprune_qwen3-0.6b-chat-100k-granular-%j.log
#SBATCH --error=./logs/loraprune_qwen3-0.6b-chat-100k-granular-%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

srun python prune.py \
    --base_model "./models/Qwen_Qwen3-0.6B" \
    --data_path './data/MBZUAI___la_mini-instruction' \
    --output_dir 'outputs_dir/qwen3-0.6b-chat-100k-granular' \
    --num_epochs 2 \
    --batch_size 128 \
    --micro_batch_size 1 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --train_set_size 100000 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj, k_proj, v_proj, o_proj, gate_proj,up_proj, down_proj]' \
    --train_on_inputs False \
    --group_by_length \
    --ratio 0.2 \
    --prune_metric 'lora' \
    --prune_freq 10 \
    --fp16 True \
    --adaptive_ema False \
    --granular_gqa True \
