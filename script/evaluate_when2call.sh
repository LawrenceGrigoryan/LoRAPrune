CUDA_VISIBLE_DEVICES=0 python ./eval_when2call.py \
    --base_model "Qwen/Qwen1.5-0.5B-Chat" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --output_dir "./outputs_dir/evaluation/when2call/" \
    --granular_gqa False \
    --num-samples 100