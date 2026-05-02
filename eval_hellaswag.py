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


def eval_hellaswag(model_id: str, adapter_id: str = None, n_shot: int = 1, batch_size: int = 8, limit: int = 10) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=False,
        device_map=device,
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
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters before pruning: {total_params}")
        
        freeze(model)
        prune_from_checkpoint(model)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters after pruning: {total_params}")
    else:
        logger.warning("No adapter provided, evaluating the base model")
    
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, device=device)

    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["hellaswag"],
        num_fewshot=n_shot,
        batch_size=batch_size,
        limit=limit,
    )

    # per-subtask accuracy
    for task, metrics in results["results"].items():
        logger.info(f"{task}: {metrics['acc,none']:.4f}")

    # overall MMLU average
    accs = [m["acc,none"] for m in results["results"].values()]
    logger.info(f"\HellaSwag avg: {np.mean(accs):.4f}")


if __name__ == "__main__":
    fire.Fire(eval_hellaswag)