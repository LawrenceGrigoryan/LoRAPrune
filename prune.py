import json
import os
from typing import List
from functools import partial

import fire
import torch
import transformers
from transformers import DataCollatorForSeq2Seq
from datasets import concatenate_datasets, load_dataset, load_from_disk
from loraprune.trainer import LoRAPruneTrainer
from loraprune.utils import freeze, is_gqa_model
from loraprune.lora import LoraConfig
from loraprune.peft_model import get_peft_model
from loraprune.data_utils import prepare_tokenizer, generate_and_tokenize_prompt, normalize_schema
from evaluation.utils import seed_everything
from loguru import logger
from safetensors.torch import save_file as safe_save_file

from peft import (
    prepare_model_for_kbit_training,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
IGNORE_INDEX = -100
SUPPORTED_MODELS = ["llama", "qwen2", "qwen3"]
_TOOL_CALLING_DATA_PATH = "./data/Salesforce___xlam-function-calling-60k"


def train(
    # model/data params
    base_model: str = "",  # the required argument
    data_path: str = "",  # the required argument
    output_dir: str = "output_dir",
    # training hyperparams
    train_set_size: int = 25000,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    warmup_steps: int = 0,
    gradient_checkpointing: bool = False,  # trade compute for activation memory; no effect on final metrics
    # optimizer params
    optimizer: str = 'adamw_torch',  # options: adamw_torch|lorapre
    lorapre_rank: int = 8,  # rank of LoRA-Pre's momentum factorisation; must be < lora_r to have any effect
    # pruning hyperparams
    ratio: float = 0.5,
    init_ratio: float = 0,
    warmup_iters: float = 0.1,
    cooldown_iters: float = 0.1,
    prune_freq: int = 10,
    prune_metric: str = 'lora',  # options: lora|grad|magnitude
    adaptive_ema: bool = False,  # use per-group variance-adaptive EMA instead of fixed 0.9/0.1
    granular_gqa: bool = False,  # prune Q heads independently; remove KV head only when all Q heads in its group are gone
    gqa_prune_signal: str = 'all',  # non-granular GQA only: which projections' sensitivity decide whole-KV-group pruning: query|keyvalue|all
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj"
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    load_in_8bit: bool = False,
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    fp16: bool = True,  # whether to use mixed precision training
    seed: int = 42,
    # tool calling samples
    num_tool_calling_samples: int = 0,  # number of tool-calling samples to include in training
):
    params = locals()
    logger.info("Pruning with params:\n" + "\n".join(f"  {k}: {v}" for k, v in params.items()))

    supported_optimizers = ('adamw_torch', 'lorapre')
    if optimizer not in supported_optimizers:
        raise ValueError(
            f"`{optimizer}` optimizer is not supported! Choose one of {supported_optimizers}."
        )

    supported_gqa_prune_signals = ('all', 'query', 'keyvalue')
    if gqa_prune_signal not in supported_gqa_prune_signals:
        raise ValueError(
            f"`{gqa_prune_signal}` gqa_prune_signal is not supported! "
            f"Choose one of {supported_gqa_prune_signals}."
        )

    if optimizer == 'lorapre':
        if lorapre_rank >= lora_r:
            # LoRA-Pre only factorises a matrix when min(shape) > rank, and the
            # trainable adapters have min(shape) == lora_r. See
            # LoRAPruneTrainer._log_lorapre_coverage.
            raise ValueError(
                f"lorapre_rank={lorapre_rank} must be strictly less than lora_r={lora_r}, "
                f"otherwise every LoRA adapter falls back to AdamW and LoRA-Pre does nothing."
            )
        if warmup_steps == 0:
            logger.warning(
                "LoRA-Pre with warmup_steps=0: the algorithm's second moment is "
                "under-estimated for the first few dozen steps, so early updates "
                "overshoot. Consider setting warmup_steps > 0."
            )

    seed_everything(seed)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=load_in_8bit,
        torch_dtype=torch.float32,
        device_map=device_map,
    )

    if is_gqa_model(model):
        logger.debug(f"GQA model detected with:\nNUM_KV_HEADS={model.config.num_key_value_heads}\nNUM_ATTN_HEADS={model.config.num_attention_heads}")
    
    # infer model type for tokenization and other stuff
    model_type = model.config.model_type
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"`{model_type}` model type is not supported!")
    
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)

        total_memory = props.total_memory / (1024**3)  # Convert to GB
        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)

        logger.info(f"GPU: {props.name}")
        logger.info(f"Total memory: {total_memory:.2f} GB")
        logger.info(f"Allocated memory: {allocated:.2f} GB")
        logger.info(f"Reserved memory: {reserved:.2f} GB")
        
        count = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {count}")

        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {props.name}")
            logger.info(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
            logger.info(f"  Compute capability: {props.major}.{props.minor}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    prepare_tokenizer(tokenizer, model_type)

    # resize embeddings (might be redundant)
    model.resize_token_embeddings(len(tokenizer))

    if load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    # TODO: convert sparseLinear for model here
    # utils.convert_sparse_network(model, target_modules=lora_target_modules)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        peft_type="LORA"
    )
    # from peft import get_peft_model
    model = get_peft_model(model, config)

    # skipped entirely for a tool-calling-only run, which needs no `data_path`
    data = None
    if train_set_size > 0:
        # c4 is too big to load into memory, so we save a smaller random sample
        try:
            data = load_from_disk(data_path)
        except Exception as e:
            logger.warning(f"Error occurred while loading dataset: {e}")
            data = load_dataset(data_path)

    # Load tool-calling samples if requested
    tool_calling_data = None
    if num_tool_calling_samples > 0:
        tool_calling_data = load_dataset(_TOOL_CALLING_DATA_PATH)["train"].shuffle(seed=seed).select(range(num_tool_calling_samples))
        logger.info(f"Loaded {len(tool_calling_data)} tool-calling samples from {_TOOL_CALLING_DATA_PATH}")
        generate_and_tokenize_prompt_partial_tool = partial(
            generate_and_tokenize_prompt, model_type=model_type, train_on_inputs=train_on_inputs, cutoff_len=cutoff_len, tokenizer=tokenizer, tool_calling=True
        )
        tool_calling_data = (
            tool_calling_data.map(generate_and_tokenize_prompt_partial_tool)
            .filter(lambda x: x["input_ids"] is not None)
        )
        # the tools JSON makes these prompts long, so a sizeable share can lose
        # their whole response to `cutoff_len` and get filtered out above
        dropped = num_tool_calling_samples - len(tool_calling_data)
        if dropped:
            logger.warning(
                f"Dropped {dropped}/{num_tool_calling_samples} "
                f"({dropped / num_tool_calling_samples:.1%}) tool-calling samples truncated past "
                f"cutoff_len={cutoff_len} - raise it to keep more of them"
            )

    # the SFT path reads `instruction`/`response`; train_on_inputs does plain
    # next-token prediction on `text` and needs no normalisation
    if not train_on_inputs and data is not None:
        data = normalize_schema(data)

    freeze(model)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    generate_and_tokenize_prompt_partial = partial(
        generate_and_tokenize_prompt, model_type=model_type, train_on_inputs=train_on_inputs, cutoff_len=cutoff_len, tokenizer=tokenizer
    )
    train_data, val_data = None, None
    if train_set_size > 0 and val_set_size > 0:
        train_val = data["train"].train_test_split(
            train_size=train_set_size, test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle()
            .map(generate_and_tokenize_prompt_partial)
            .filter(lambda x: x["input_ids"] is not None)
        )
        # validation stays instruction-only, so eval loss remains comparable
        # across runs with and without tool-calling data
        val_data = (
            train_val["test"].shuffle()
            .map(generate_and_tokenize_prompt_partial)
            .filter(lambda x: x["input_ids"] is not None)
        )
    elif train_set_size > 0:
        train_data = (
            data["train"].shuffle().select(range(train_set_size))
            .map(generate_and_tokenize_prompt_partial)
            .filter(lambda x: x["input_ids"] is not None)
        )

    if train_data is not None and tool_calling_data is not None:
        # both sets are already tokenized, but carry different leftover columns
        # (instruction/response vs id/query/answers/tools). Concatenating those
        # directly would null-fill the mismatch, so keep only what training reads.
        keep = ["input_ids", "attention_mask", "labels"]
        n_instruction, n_tool = len(train_data), len(tool_calling_data)
        train_data = concatenate_datasets(
            [train_data.select_columns(keep), tool_calling_data.select_columns(keep)]
        ).shuffle(seed=seed)  # interleave, so the two kinds aren't seen block by block
        logger.info(
            f"Mixed training set: {n_instruction} instruction + {n_tool} tool-calling "
            f"= {len(train_data)} samples"
        )
    elif tool_calling_data is not None:
        logger.warning("Using ONLY tool-calling samples as training data!")
        train_data = tool_calling_data
    elif train_data is None:
        raise ValueError("No training data specified! Please provide a train_set_size > 0 or num_tool_calling_samples > 0.")

    trainer = LoRAPruneTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=fp16,
            gradient_checkpointing=gradient_checkpointing,
            logging_steps=5,
            # only consulted when optimizer != 'lorapre'; LoRAPruneTrainer.create_optimizer
            # builds LoRA-Pre itself and ignores this field.
            optim="adamw_torch",
            # keyed on the data, not on val_set_size: a tool-calling-only run has
            # no validation set even when val_set_size is left at its default
            eval_strategy="steps" if val_data is not None else "no",
            save_strategy="no",  # don't save with save_pretrained, results in a corrupted save
            eval_steps=50 if val_data is not None else None,
            save_steps=50,
            output_dir=output_dir,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            max_grad_norm=1.0,  # avoid exploding grads
        ),
        # we already created labels, so this collator is just for padding - no need to worry about ignoring pad tokens
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        ratio=ratio,
        init_ratio=init_ratio,
        warmup_iters=warmup_iters,
        cooldown_iters=cooldown_iters,
        prune_freq=prune_freq,
        prune_metric=prune_metric,
        adaptive_ema=adaptive_ema,
        granular_gqa=granular_gqa,
        gqa_prune_signal=gqa_prune_signal,
        optimizer_name=optimizer,
        lorapre_rank=lorapre_rank,
    )

    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # manually save only adapter weights
    lora_state_dict = {
        k: v
        for k, v in model.state_dict().items()
        if "lora_" in k
    }
    safe_save_file(lora_state_dict, os.path.join(output_dir, ADAPTER_WEIGHTS_NAME))
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)