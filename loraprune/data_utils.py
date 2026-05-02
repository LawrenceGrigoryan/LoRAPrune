from transformers import AutoTokenizer
from loguru import logger


def prepare_tokenizer(tokenizer: AutoTokenizer, model_type: str) -> None:
    """
    Prepare tokenizer in-place
    """
    # FIXME: switch to sequence packing
    tokenizer.padding_side = "left"  # Allow batched inference
    if model_type == "llama":  # llama-3.2-1b
        tokenizer.pad_token_id = 128004  # set to <|finetune_right_pad_id|>, different from eos
    elif model_type == "qwen2":  # qwen-1.5-0.5b
        # pad == eos, add a new one
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        # qwen2 lacks bos token
        tokenizer.bos_token = "<|im_start|>"
    else:
        raise ValueError(f"Unsupported model type `{model_type}`!")


def tokenize(prompt: str, tokenizer: AutoTokenizer, cutoff_len: int, add_bos_token: bool = False, add_eos_token: bool = True, **kwargs):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len-2,  # reserve space for BOS/EOS tokens
        padding=False,
        return_tensors=None,
        add_special_tokens=False,  # we'll add these ourselves
        **kwargs,
    )
    # BOS token added always
    if add_bos_token:
        result["input_ids"] = [tokenizer.bos_token_id] + result["input_ids"]
        result["attention_mask"] = [1] + result["attention_mask"]
    
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):  
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    # models internally perform the label shift
    result["labels"] = result["input_ids"].copy()

    return result


def generate_sft_sample(data_point):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["response"]}"""


def generate_and_tokenize_prompt(data_point: dict, tokenizer: AutoTokenizer, model_type: str, cutoff_len: int, train_on_inputs: bool):
    if not train_on_inputs:
        if model_type == "qwen2":
            prompt = data_point["instruction"]
            assistant_response = data_point["response"]
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_response}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len=cutoff_len-1, add_eos_token=False)
            assistant_bos = "<|im_start|>assistant\n"
            user_prompt = full_prompt[:full_prompt.rfind(assistant_bos) + len(assistant_bos)]
            tokenized_user_prompt = tokenize(user_prompt, tokenizer, cutoff_len=cutoff_len, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            # could be sped up, probably
            response_labels = tokenized_full_prompt["labels"][user_prompt_len:]
            if len(response_labels) == 0:
                logger.warning(f"No response tokens - omitting the sample, cutoff_len={cutoff_len}")
                return {"input_ids": None, "attention_mask": None, "labels": None}
            else:
                tokenized_full_prompt["labels"] = [-100] * user_prompt_len + response_labels
        else:
            raise NotImplementedError(f"SFT Tokenization not implemented for model type: {model_type}")
    elif train_on_inputs:
        # no bos token for qwen1.5/qwen2 base models 
        if model_type == "qwen2":
            full_prompt = data_point["text"]
            tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len=cutoff_len, add_bos_token=False, add_eos_token=True)
        else:
            full_prompt = data_point["text"]
            tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len=cutoff_len, add_bos_token=True, add_eos_token=True)
    return tokenized_full_prompt