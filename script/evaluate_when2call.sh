python eval_when2call.py \
    --base_model "Qwen/Qwen1.5-0.5B-Chat" \
    # --lora_weights "./outputs_dir/qwen15_05b_chat_lamini_20k" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --output_path "./evaluation/results/qwen15-05b-chat/when2call_results.jsonl"