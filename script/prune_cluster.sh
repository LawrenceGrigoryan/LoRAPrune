#!/usr/bin/env bash
#SBATCH --job-name=loraprune_test
#SBATCH --output=loraprune_test%j.log
#SBATCH --error=loraprune_test%j.err
#SBATCH --mail-user=REPLACE_USER_NAME@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

srun python prune.py \
    --base_model "./models/meta-llama_Llama-3.2-1B" \
    --data_path './data/MBZUAI_LaMini-instruction' \
    --output_dir 'outputs_dir' \
    --batch_size 2 \
    --micro_batch_size 1 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --train_set_size 500000 \
    --val_set_size 1000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj, k_proj, v_proj, o_proj, gate_proj,up_proj, down_proj]' \
    --train_on_inputs \
    --group_by_length \
    --ratio 0.5 \
    --prune_metric 'lora' \
    --prune_freq 10 \
