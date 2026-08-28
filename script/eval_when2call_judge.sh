#!/usr/bin/env bash
# Runs off-cluster: the judge needs network access.
# Requires OPENAI_API_KEY in .env

python ./eval_when2call_judge.py \
    --responses_path "./outputs_dir/evaluation/results/Qwen_Qwen1.5-0.5B-Chat/when2call_responses.jsonl" \
    --judge_model "gpt-4.1-mini" \
    --temperature 0.0 \
    --max_workers 8
