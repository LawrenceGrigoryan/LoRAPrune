import os
from typing import List
from functools import partial

import fire
import torch
import transformers
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset, load_from_disk
from loraprune.trainer import LoRAPruneTrainer
from loraprune.utils import freeze
from loraprune.lora import LoraConfig
from loraprune.peft_model import get_peft_model
from loraprune.data_utils import prepare_tokenizer, generate_and_tokenize_prompt
from loguru import logger
from safetensors.torch import save_file as safe_save_file

from peft import (
    prepare_model_for_kbit_training,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
IGNORE_INDEX = -100
SUPPORTED_MODELS = ["llama", "qwen2"]


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
    # pruning hyperparams
    ratio: float = 0.5,
    init_ratio: float = 0,
    warmup_iters: float = 0.1,
    cooldown_iters: float = 0.1,
    prune_freq: int = 10,
    prune_metric: str = 'lora',  # options: lora|grad|magnitude
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
):
    logger.info(
        f"Pruning with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"train_set_size: {train_set_size}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"load_in_8bit: {load_in_8bit}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"prune_metric: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
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
        torch_dtype=torch.float16,
        device_map=device_map,
    )
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

    # c4 is too big to load into memory, so we save a smaller random sample
    try:
        data = load_from_disk(data_path)
    except Exception as e:
        logger.warning(f"Error occurred while loading dataset: {e}")
        data = load_dataset(data_path)

    freeze(model)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    generate_and_tokenize_prompt_partial = partial(
        generate_and_tokenize_prompt, model_type=model_type, train_on_inputs=train_on_inputs, cutoff_len=cutoff_len, tokenizer=tokenizer
    )
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            train_size=train_set_size, test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt_partial)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt_partial)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt_partial)
        val_data = None

    trainer = LoRAPruneTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=fp16,
            logging_steps=5,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="no",  # don't save with save_pretrained, results in a corrupted save
            eval_steps=50 if val_set_size > 0 else None,
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
        prune_metric=prune_metric
    )

    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # manually save only adapter weights
    lora_state_dict = {
        k: v
        for k, v in model.state_dict().items()
        if "lora_" in k
    }
    os.makedirs(output_dir, exist_ok=True)
    safe_save_file(lora_state_dict, os.path.join(output_dir, ADAPTER_WEIGHTS_NAME))
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)