"""Generate real When2Call responses from a base or LoRA-pruned model.

The MCQ task in ``eval_when2call.py`` scores four gold answer strings by
loglikelihood and never asks the model to generate. That is blind to whether a
pruned model can still *produce* a usable tool call, which is what this script
captures: it writes the model's actual output for every test item, for
``eval_when2call_judge.py`` to classify.

Prompts come from the vendored When2Call ``process_docs_*`` builders, the same
ones the MCQ task uses, so both evaluations see an identical prompt.
"""

import json
import os
from typing import List

import fire
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from loguru import logger
from peft.utils.save_and_load import load_peft_weights
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.utils import seed_everything
from evaluation.When2Call.evaluation.mcq.lm_eval_harness.when2call.utils import (
    process_docs_llama3_2,
    process_docs_qwen2_5,
)
from loraprune.data_utils import prepare_tokenizer
from loraprune.lora import LoraConfig
from loraprune.peft_model import get_peft_model
from loraprune.utils import freeze, prune_from_checkpoint

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

LLM_JUDGE_SPLIT_PATH = "./evaluation/When2Call/data/test/when2call_test_llm_judge.jsonl"

# Fields carried into the output so the judge needs only that one file
PASSTHROUGH_FIELDS = ("uuid", "correct_answer", "question", "tools", "target_tool", "answers", "source")

# Turn-ending markers beyond the tokenizer's configured eos. The prompts are raw
# template strings rather than `apply_chat_template` output, so the token that
# actually ends the assistant turn is not always `eos_token_id`
EXTRA_STOP_TOKENS = ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>")


def _stop_token_ids(tokenizer: AutoTokenizer) -> List[int]:
    """
    Return the token ids that should terminate a model's generation. This is the
    union of the tokenizer's configured eos token and any extra turn-ending tokens
    that appear in the When2Call prompts. 
    
    The latter are not always in the tokenizer's vocabulary,
    so we filter out any that are missing or map to the unk token.
    """
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    for token in EXTRA_STOP_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id >= 0 and token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    return sorted(ids)


def _load_llm_judge_split(path: str) -> Dataset:
    """
    Load the 300-row balanced judge split from the vendored When2Call jsonl.

    Note `from_list` takes its schema from the first record, so only fields
    present in every row survive.
    """
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(rows)


def _build_prompts(dataset: Dataset, model_type: str) -> Dataset:
    """
    Render each item into that model family's tool-calling prompt.
    Uses vendor's `process_docs_*` builders, which are the same ones the MCQ task uses.
    """
    if model_type in ["qwen2", "qwen3"]:
        return process_docs_qwen2_5(dataset)
    elif model_type == "llama":
        return process_docs_llama3_2(dataset)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


@torch.no_grad()
def _generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    stop_ids: List[int],
    decoding: dict,
) -> List[dict]:
    """
    Generate a continuation for each prompt in a batch and trim it to the assistant turn.

    Prompts are tokenized with ``add_special_tokens=False`` because the
    ``process_docs_*`` builders already emit a complete template string ending
    at the assistant header; letting the tokenizer prepend a BOS would corrupt
    it. Padding is left-side (set by ``prepare_tokenizer``), so generated tokens
    start at a common offset across the batch and can be sliced off by prompt
    length.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Model to generate with, either a base causal LM or a pruned
        ``LoraPeftModelForCausalLM``.
    tokenizer : transformers.PreTrainedTokenizer
        Left-padded tokenizer whose vocabulary matches ``model``'s resized
        embeddings.
    prompts : list of str
        Fully rendered When2Call prompts, ideally of similar length so padding
        waste stays low.
    max_new_tokens : int
        Cap on generated tokens. Reaching it is reported as
        ``finished_with_eos=False``.
    stop_ids : list of int
        Token ids that end an assistant turn. Passed to ``generate`` as
        ``eos_token_id`` and reused to trim the decoded output.
    decoding : dict
        Sampling or greedy keyword arguments forwarded to ``generate``, built
        once by the caller so nothing is inherited from the base model's
        ``generation_config``.

    Returns
    -------
    list of dict
        One entry per prompt, in the order given, with keys ``model_response``
        (decoded text, special tokens stripped), ``n_generated_tokens`` (length
        before the stop token) and ``finished_with_eos``.

    Notes
    -----
    ``finished_with_eos=False`` means the model was still generating when it hit
    the cap. It is recorded rather than silently accepted, since runaway
    generation is a pruning symptom the judge should see as truncated output
    rather than as a considered response.
    """
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    prompt_len = encoded["input_ids"].shape[1]

    outputs = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids or tokenizer.eos_token_id,
        **decoding,
    )

    stop_set = set(stop_ids)
    results = []
    for idx, row in enumerate(outputs[:, prompt_len:]):
        token_ids = row.tolist()
        # Cut at the first stop token; everything past it is padding.
        cut, finished = len(token_ids), False
        for i, token_id in enumerate(token_ids):
            if token_id in stop_set:
                cut, finished = i, True
                break
        body = token_ids[:cut]
        results.append(
            {
                "model_response": tokenizer.decode(body, skip_special_tokens=True).strip(),
                "n_generated_tokens": len(body),
                "finished_with_eos": finished,
            }
        )
        logger.info(f"Prompt: {tokenizer.decode(encoded['input_ids'][idx])}")
        logger.info(f"Response: {results[-1]['model_response']}")
    return results


def run_when2call_inference(
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
) -> None:
    """Generate When2Call responses for a base or LoRA-pruned model.

    Args:
        base_model: HuggingFace ID or local path of the base causal LM.
        lora_weights: Path to a LoRAPrune checkpoint. When provided, adapter
            weights are loaded and pruning masks applied before generation.
            When omitted, the base model is evaluated as-is.
        split: ``llm_judge`` for the balanced 300-item judge split (100 each of
            tool_call / request_for_info / cannot_answer), or ``mcq`` for the
            full 3652-item split used by ``eval_when2call.py``.
        granular_gqa: Must match the value the checkpoint was pruned with.
        batch_size: Prompts per generation batch. Items are length-sorted first
            so batches pad to a similar length.
        max_new_tokens: Generation cap. Hitting it is recorded as
            ``finished_with_eos=false``.
        do_sample: Sample rather than decode greedily. The run as a whole is
            reproducible from ``seed``, but only for a fixed ``batch_size`` and
            item order, since the RNG is shared across a batch.
        temperature: Sampling temperature. Must be > 0 when ``do_sample`` is set;
            transformers rejects 0.0 and points at ``do_sample=False`` instead.
        top_p, top_k, repetition_penalty: Passed explicitly and defaulted to
            neutral, rather than inherited from the **base model's**
            ``generation_config``. LoRAPrune checkpoints save no
            ``generation_config`` of their own, so a base run and its pruned run
            always shared these; what differs is decoding *between* base models
            — Qwen1.5-0.5B-Chat ships ``repetition_penalty=1.1``/``top_k=50``,
            Qwen3-0.6B ships ``1.0``/``20``. The repetition penalty matters most:
            wherever it is inherited it damps, in base and pruned alike, exactly
            the degenerate looping this evaluation exists to detect.
        num_samples: Generate for a shuffled subset instead of the whole split.

    Writes ``{output_dir}/{run_name}/when2call_responses.jsonl``.
    """
    # Passed explicitly so nothing is inherited from the base model's generation_config.
    decoding = (
        {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k}
        if do_sample
        else {"do_sample": False, "num_beams": 1}
    )
    decoding["repetition_penalty"] = repetition_penalty

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
        f"  batch_size={batch_size}\n"
        f"  max_new_tokens={max_new_tokens}\n"
        f"  decoding={decoding}\n"
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
    # Checkpoints save their own tokenizer, and Qwen gains a `<|pad|>` token
    # during training, so the vocabulary there is one larger than the base
    tokenizer = AutoTokenizer.from_pretrained(lora_weights or base_model, legacy=False)
    model_type = model.config.model_type
    # Same tokenizer setup as training, and applied for the base model too so that
    # base and pruned runs tokenize identically. Without it the base tokenizer pads
    # with <|endoftext|> at vocab 151646 while a checkpoint's pads with <|pad|> at
    # 151647, which is not a comparison we want to draw conclusions from.
    # Idempotent on a checkpoint tokenizer that already carries <|pad|>.
    prepare_tokenizer(tokenizer, model_type, mode="inference")  # also sets padding_side="left"
    model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params}")
    if lora_weights:
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

    # model.half()
    model.bfloat16()
    model.eval()

    eval_dataset = _load_llm_judge_split(LLM_JUDGE_SPLIT_PATH)

    if num_samples is not None and num_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle(seed=seed).select(range(num_samples))
        logger.info(f"Sampled {num_samples} examples with seed={seed}")

    dataset_prep = _build_prompts(eval_dataset, model_type)
    logger.info(f"Generating for {len(dataset_prep)} items, model_type={model_type!r}")

    stop_ids = _stop_token_ids(tokenizer)
    logger.info(f"Stop token ids: {stop_ids}")

    items = [dataset_prep[i] for i in range(len(dataset_prep))]
    # Length-sort so each batch pads to a similar length. Order does not matter
    # downstream because every record carries its uuid.
    order = sorted(range(len(items)), key=lambda i: -len(items[i]["prompt"]))

    result = []
    for start in tqdm(range(0, len(order), batch_size)):
        chunk = order[start : start + batch_size]
        outputs = _generate_batch(
            model, tokenizer, [items[i]["prompt"] for i in chunk], max_new_tokens, stop_ids, decoding
        )
        for idx, output in zip(chunk, outputs):
            record = {key: items[idx].get(key) for key in PASSTHROUGH_FIELDS}
            record["prompt"] = items[idx]["prompt"]
            record.update(output)
            result.append(record)

    run_name = os.path.basename(os.path.normpath(lora_weights)) if lora_weights else os.path.basename(os.path.normpath(base_model))
    save_path = os.path.join(output_dir, run_name, "when2call_responses.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for item in result:
            f.write(json.dumps(item) + "\n")

    empty = sum(1 for r in result if not r["model_response"].strip())
    truncated = sum(1 for r in result if not r["finished_with_eos"])
    logger.info(f"Wrote {len(result)} responses to {save_path}")
    logger.info(f"empty:     {empty} ({empty / len(result):.1%})")
    logger.info(f"truncated: {truncated} ({truncated / len(result):.1%})")


if __name__ == "__main__":
    fire.Fire(run_when2call_inference)
