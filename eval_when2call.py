from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from loguru import logger
import fire
import torch
import json
import numpy as np
from peft.utils.save_and_load import load_peft_weights
from tqdm import tqdm

from loraprune.peft_model import get_peft_model
from loraprune.utils import freeze, prune_from_checkpoint
from loraprune.lora import LoraConfig
from data_utils import prepare_tokenizer
from evaluation.When2Call.evaluation.mcq.lm_eval_harness.when2call.utils import process_docs_qwen2_5, process_docs_llama3_2
from evaluation.utils import compute_loglikelihood


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def main(base_model: str = "",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.,
        lora_target_modules: List[str] = [
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj"
            ],
        lora_weights: str | None = None,
        output_path: str = "./when2call_results.jsonl",
    ):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    logger.info(f"Using device: `{device}`")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, legacy=False)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params}")
    model_type = model.config.model_type
    if lora_weights:
        prepare_tokenizer(tokenizer, model_type)

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        
        adapter_weights = load_peft_weights(lora_weights)
        for name, param in adapter_weights.items():
            if 'lora_mask' in name:
                adapter_weights[name] = param.reshape(-1)
                
        # inject only adapter state dict
        # will return missing keys warning for base model's layers that
        # are not in the adapter state dict
        model.load_state_dict(adapter_weights, strict=False)

        model.to(device)
        
        freeze(model)
        prune_from_checkpoint(model)
        
        total_params_pruned = sum(p.numel() for p in model.parameters())
        logger.info(f"Total model parameters after pruning: {total_params_pruned}")
        logger.info(f"Parameters left after pruning: {round((total_params_pruned / total_params)*100, 2)}%")
    else:
        logger.warning("LoRA weights path is not specified, evaluating the base model...")

    model.half()  # seems to fix bugs for some users.

    # MCQ - multiple choice question evaluation, llm as a judge possible as well
    eval_dataset = load_dataset("nvidia/When2Call", "test")
    if model_type == "qwen2":
        dataset_prep = process_docs_qwen2_5(eval_dataset["mcq"])
    elif model_type == "llama":
        dataset_prep = process_docs_llama3_2(eval_dataset["mcq"])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # FIXME: batched
    result = []
    for i in tqdm(range(5)):
        sample = dataset_prep[i]
        choices = sample["answers"]
        correct_choice = sample["correct_answer"]
        answer_loglikelihoods = {}
        for type, choice in choices.items():
            prompt = f"{sample['prompt']} {choice}{tokenizer.eos_token}"
            ll = compute_loglikelihood(prompt, model, tokenizer)
            answer_loglikelihoods[type] = ll
        predicted_choice = max(answer_loglikelihoods, key=lambda x: answer_loglikelihoods.get(x))
        result.append({"gold": correct_choice, "predicted": predicted_choice})

    with open(output_path, "w") as f:
        for item in result:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    fire.Fire(main)