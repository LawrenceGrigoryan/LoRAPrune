"""
End-to-end When2Call LLM-as-a-judge evaluation: generate, then judge.

A thin orchestrator over the two stages, so a single command produces the
metrics rather than requiring the responses file to be threaded by hand:

    eval_when2call_inference.run_when2call_inference  ->  when2call_responses.jsonl
    eval_when2call_judge.run_when2call_judge          ->  when2call_judge.jsonl
                                                          when2call_judge_summary.json

All three land in ``{output_dir}/{run_name}/``, where ``run_name`` is the
checkpoint directory name (or the base model's, when evaluating unpruned).

Note this stage needs a GPU *and* network access in the same place: generation
runs locally, the judge calls the OpenAI API. On an offline cluster, run the two
scripts separately instead — inference there, judging afterwards on a networked
machine pointed at the responses file.
"""

import json
import os
from typing import List

import fire
from dotenv import load_dotenv
from loguru import logger

from eval_when2call_inference import run_when2call_inference
from eval_when2call_judge import run_when2call_judge

load_dotenv()

RESPONSES_FILE = "when2call_responses.jsonl"
JUDGE_FILE = "when2call_judge.jsonl"
SUMMARY_FILE = "when2call_judge_summary.json"


def _run_dir(output_dir: str, base_model: str, lora_weights: str | None) -> str:
    """Mirror the inference script's output layout: one directory per run."""
    run_name = os.path.basename(os.path.normpath(lora_weights or base_model))
    return os.path.join(output_dir, run_name)


def run_when2call_pipeline(
    base_model: str = "",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_weights: str | None = None,
    output_dir: str = "./outputs_dir/evaluation/results/",
    granular_gqa: bool = False,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 0.1,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    num_samples: int | None = None,
    seed: int = 42,
    judge_model: str = "gpt-4.1-mini",
    judge_temperature: float = 0.0,
    max_workers: int = 8,
    limit: int | None = None,
) -> dict:
    """Generate When2Call responses for a model, then judge them, in one run.

    Args:
        base_model: HuggingFace ID or local path of the base causal LM.
        lora_weights: LoRAPrune checkpoint directory; omit to evaluate the base model.
        granular_gqa: Must match the value the checkpoint was pruned with.
        temperature: Sampling temperature for *generation*. The judge's own
            temperature is ``judge_temperature``.
        judge_model: OpenAI model to judge with.
        limit: Judge only the first N responses (cheap trial run). Generation is
            capped separately, by ``num_samples``.

    Returns:
        The judge summary dict, also written to ``when2call_judge_summary.json``.
    """
    run_dir = _run_dir(output_dir, base_model, lora_weights)
    responses_path = os.path.join(run_dir, RESPONSES_FILE)
    judge_path = os.path.join(run_dir, JUDGE_FILE)
    summary_path = os.path.join(run_dir, SUMMARY_FILE)

    logger.info(f"===== When2Call LLM-as-a-judge pipeline: {os.path.basename(run_dir)} =====")

    logger.info("Stage 1/2: generating responses")
    run_when2call_inference(
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_weights=lora_weights,
        output_dir=output_dir,
        granular_gqa=granular_gqa,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        num_samples=num_samples,
        seed=seed,
    )
    
    # remove old verdicts if exist, otherwise judge will skip judging the responses
    if os.path.exists(judge_path):
        os.remove(judge_path)
        logger.info(f"Responses regenerated; discarded previous verdicts at {judge_path}")

    logger.info("Stage 2/2: judging responses")
    run_when2call_judge(
        responses_path=responses_path,
        judge_model=judge_model,
        temperature=judge_temperature,
        max_workers=max_workers,
        limit=limit,
    )

    with open(summary_path) as f:
        summary = json.load(f)

    logger.info(f"===== Done: {os.path.basename(run_dir)} =====")
    logger.info(f"  responses: {responses_path}")
    logger.info(f"  verdicts:  {judge_path}")
    logger.info(f"  summary:   {summary_path}")
    return summary


if __name__ == "__main__":
    fire.Fire(run_when2call_pipeline)
