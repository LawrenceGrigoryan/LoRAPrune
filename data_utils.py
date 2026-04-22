from transformers import AutoTokenizer


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


def tokenize(prompt: str, tokenizer: AutoTokenizer, cutoff_len: int, add_eos_token: bool = True, **kwargs):
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


def generate_and_tokenize_prompt(data_point: dict, tokenizer: AutoTokenizer, cutoff_len: int, train_on_inputs: bool):
    if not train_on_inputs:
        tokenized_full_prompt = tokenize(generate_sft_sample(data_point), tokenizer, cutoff_len=cutoff_len-1, add_eos_token=True)
        user_prompt = generate_sft_sample({**data_point, "response": ""})
        tokenized_user_prompt = tokenize(user_prompt, tokenizer, cutoff_len=cutoff_len, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        # could be sped up, probably
        response_labels = tokenized_full_prompt["labels"][user_prompt_len:]
        if len(response_labels) == 0:
            # instruction fills the entire context — no response tokens to train on;
            # fall back to training on all tokens to avoid an all-masked batch (NaN loss)
            pass
        else:
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + response_labels
    else:
        full_prompt = data_point["text"]
        tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len=cutoff_len, add_eos_token=True)
    return tokenized_full_prompt