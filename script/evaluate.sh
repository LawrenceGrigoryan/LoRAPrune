CUDA_VISIBLE_DEVICES=0 python ./eval_perplexity.py \
    --base_model "Qwen/Qwen1.5-0.5B" \
    --lora_weights 'outputs_dir/qwen1.5-0.5b-base-10k' \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"