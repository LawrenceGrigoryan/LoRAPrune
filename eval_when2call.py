from typing import List
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from loguru import logger
import fire
import torch
import json
from peft.utils.save_and_load import load_peft_weights
from tqdm import tqdm
from dotenv import load_dotenv

from loraprune.peft_model import get_peft_model
from loraprune.utils import freeze, prune_from_checkpoint
from loraprune.lora import LoraConfig
from loraprune.data_utils import prepare_tokenizer
from evaluation.When2Call.evaluation.mcq.lm_eval_harness.when2call.utils import process_docs_qwen2_5, process_docs_llama3_2
from evaluation.utils import compute_loglikelihood, seed_everything

load_dotenv()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def _macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


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
        output_dir: str = "./outputs_dir/evaluation/results/",
        granular_gqa: bool = False,
        num_samples: int | None = None,
        seed: int = 42) -> None:
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    logger.info(
        "Parameters:\n"
        f"  base_model={base_model!r}\n"
        f"  lora_r={lora_r}\n"
        f"  lora_alpha={lora_alpha}\n"
        f"  lora_dropout={lora_dropout}\n"
        f"  lora_target_modules={lora_target_modules}\n"
        f"  lora_weights={lora_weights!r}\n"
        f"  output_dir={output_dir!r}\n"
        f"  granular_gqa={granular_gqa}\n"
        f"  num_samples={num_samples}\n"
        f"  seed={seed}"
    )
    seed_everything(seed)
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
        prepare_tokenizer(tokenizer, model_type, mode="inference")

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
        prune_from_checkpoint(model, granular_gqa=granular_gqa)
        
        total_params_pruned = sum(p.numel() for p in model.parameters())
        logger.info(f"Total model parameters after pruning: {total_params_pruned}")
        logger.info(f"Parameters left after pruning: {round((total_params_pruned / total_params)*100, 2)}%")
    else:
        logger.warning("LoRA weights path is not specified, evaluating the base model...")

    model.half()  # seems to fix bugs for some users.

    # MCQ - multiple choice question evaluation, llm as a judge possible as well
    eval_dataset = load_dataset(f"{os.getenv('HF_DATASETS_CACHE')}/nvidia___when2_call", split="test")
    if num_samples is not None and num_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle(seed=seed).select(range(num_samples))
        logger.info(f"Sampled {num_samples} examples with seed={seed}")
    if model_type in ["qwen2", "qwen3"]:
        dataset_prep = process_docs_qwen2_5(eval_dataset)
    elif model_type == "llama":
        dataset_prep = process_docs_llama3_2(eval_dataset)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # FIXME: batched
    result = []
    for i in tqdm(range(len(dataset_prep))):
        sample = dataset_prep[i]
        choices = sample["choices"]
        target_index = sample["target_index"]
        answer_types = list(sample["answers"].keys())
        lls = []
        lls_norm = []
        for choice in choices:
            ll, n_cont = compute_loglikelihood(sample["prompt"], choice, model, tokenizer)
            lls.append(ll)
            lls_norm.append(ll / n_cont)
        pred_idx = max(range(len(choices)), key=lambda idx: lls[idx])
        pred_idx_norm = max(range(len(choices)), key=lambda idx: lls_norm[idx])
        result.append({
            "gold": answer_types[target_index],
            "predicted": answer_types[pred_idx],
            "predicted_norm": answer_types[pred_idx_norm],
        })

    adapter_name = os.path.basename(os.path.normpath(lora_weights)) if lora_weights else "base"
    save_path = os.path.join(output_dir, adapter_name, "when2call.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for item in result:
            f.write(json.dumps(item) + "\n")

    acc = sum(item["gold"] == item["predicted"] for item in result) / len(result)
    acc_norm = sum(item["gold"] == item["predicted_norm"] for item in result) / len(result)
    macro_f1 = _macro_f1([r["gold"] for r in result], [r["predicted"] for r in result])
    macro_f1_norm = _macro_f1([r["gold"] for r in result], [r["predicted_norm"] for r in result])
    logger.info(f"acc:           {acc:.4f}")
    logger.info(f"acc_norm:      {acc_norm:.4f}")
    logger.info(f"macro_f1:      {macro_f1:.4f}")
    logger.info(f"macro_f1_norm: {macro_f1_norm:.4f}")
    

if __name__ == "__main__":
    fire.Fire(main)