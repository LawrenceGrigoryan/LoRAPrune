#!/usr/bin/env bash
# End-to-end When2Call LLM-as-a-judge evaluation: generate, then judge.
#
# Needs a GPU *and* network access in the same place. On an offline cluster run
# eval_when2call_inference.py there and eval_when2call_judge.py afterwards.
# Requires OPENAI_API_KEY in .env

python ./eval_when2call_llm_as_judge.py \
    --base_model "./models/Qwen_Qwen1.5-0.5B-Chat" \
    --lora_weights "./outputs_dir/qwen1.5-0.5b-chat-10k" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --output_dir "./outputs_dir/evaluation/results/" \
    --granular_gqa True \
    --batch_size 8 \
    --max_new_tokens 256 \
    --do_sample True \
    --temperature 0.1 \
    --num_samples 5 \
    --seed 42 \
    --judge_model "gpt-4.1-mini" \
    --judge_temperature 0.0 \
    --max_workers 8
