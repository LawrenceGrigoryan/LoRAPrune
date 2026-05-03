import json
import os
import fire
import torch
import numpy as np
from peft.utils.save_and_load import load_peft_weights
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM

from loraprune.peft_model import get_peft_model
from loraprune.utils import freeze, prune_from_checkpoint
from loraprune.lora import LoraConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def eval_commonsense(model_id: str, adapter_id: str = None, batch_size: int = 8, limit: int = None, output_dir: str = "./evaluation/") -> None:
    """Evaluate a (optionally LoRA-pruned) causal LM on commonsense reasoning benchmarks.

    Runs lm-evaluation-harness on MMLU (5-shot), HellaSwag (0-shot) and WinoGrande (0-shot),
    logging per-task accuracy and the macro-average across subtasks.

    Args:
        model_id: HuggingFace model ID or local path of the base causal LM.
        adapter_id: Path to a LoRAPrune checkpoint. When provided, LoRA adapter
            weights are loaded and pruning masks are applied before evaluation.
            When omitted, the base model is evaluated as-is.
        batch_size: Batch size passed to lm-evaluation-harness.
        limit: Cap the number of evaluation samples per task (useful for quick runs).
        output_dir: Root directory for saving results. Metrics are written to
            ``{output_dir}/{adapter_id}/commonsense.json``.
    """
    logger.info(
        f"Evaluation with params:\n"
        f"Base model: {model_id}\n"
        f"Adapter: {adapter_id}\n"
        f"Batch_size: {batch_size}\n"
        f"Limit: {limit}\n"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=False,
        device_map=device,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_id or model_id)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters before pruning: {total_params}")
    
    if adapter_id:
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj".split(","),
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        
        adapters_weights = load_peft_weights(adapter_id)
        for name, param in adapters_weights.items():
            if 'lora_mask' in name:
                adapters_weights[name] = param.reshape(-1)
        model.load_state_dict(adapters_weights, strict=False)
        model.to(device)

        freeze(model)
        prune_from_checkpoint(model)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters after pruning: {total_params}")
    else:
        logger.warning("No adapter provided, evaluating the base model")
    
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, device=device)

    logger.info(f"Evaluation on MMLU, 5-shot...")
    mmlu_res = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["mmlu"],
        num_fewshot=5,
        batch_size=batch_size,
        limit=limit,
    )
    logger.info(f"Evaluation on HellaSwag, 0-shot...")
    hellaswag_res = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size=batch_size,
        limit=limit,
    )
    logger.info(f"Evaluation on WinoGrande, 0-shot...")
    wino_res = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["winogrande"],
        num_fewshot=0,
        batch_size=batch_size,
        limit=limit,
    )

    named_results = [("mmlu", mmlu_res), ("hellaswag", hellaswag_res), ("winogrande", wino_res)]
    output = {}
    for name, result in named_results:
        task_accs = {}
        for task, metrics in result["results"].items():
            acc = metrics["acc,none"]
            task_accs[task] = acc
            logger.info(f"{task}: {acc:.4f}")
        avg = float(np.mean(list(task_accs.values())))
        logger.info(f"{name} avg: {avg:.4f}")
        output[name] = {**task_accs, "avg": avg}

    adapter_name = os.path.basename(os.path.normpath(adapter_id)) if adapter_id else "base"
    save_path = os.path.join(output_dir, adapter_name, "commonsense.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {save_path}")


if __name__ == "__main__":
    fire.Fire(eval_commonsense)