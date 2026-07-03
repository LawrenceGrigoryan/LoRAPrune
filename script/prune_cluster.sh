#!/usr/bin/env bash
#SBATCH --job-name=loraprune_test
#SBATCH --output=./logs/loraprune_qwen15-05b-c4-20k_%j.log
#SBATCH --error=./logs/loraprune_qwen15-05b-c4-20k_%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

srun python prune.py \
    --base_model "./models/Qwen_Qwen1.5-0.5B" \
    --data_path './data/allenai___c4_120k' \
    --output_dir 'outputs_dir/qwen15-05b-c4-20k-ratio-0.5' \
    --num_epochs 2 \
    --batch_size 128 \
    --micro_batch_size 1 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --train_set_size 20000 \
    --val_set_size 1000 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj, k_proj, v_proj, o_proj, gate_proj,up_proj, down_proj]' \
    --train_on_inputs \
    --group_by_length \
    --ratio 0.5 \
    --prune_metric 'lora' \
    --prune_freq 10 \
    --fp16 false
