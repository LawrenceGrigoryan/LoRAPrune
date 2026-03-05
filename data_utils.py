from transformers import AutoTokenizer


def tokenize(prompt: str, tokenizer: AutoTokenizer, cutoff_len: int, add_eos_token: bool = True, **kwargs):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len-1,  # reserve space for EOS token
        padding=False,
        return_tensors=None,
        **kwargs,
    )
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


def generate_and_tokenize_prompt(data_point: dict, tokenizer: AutoTokenizer,cutoff_len: int, train_on_inputs: bool):
    if not train_on_inputs:
        tokenized_full_prompt = tokenize(generate_sft_sample(data_point), tokenizer, cutoff_len=cutoff_len-1, add_eos_token=True, add_bos_token=False)
        user_prompt = generate_sft_sample({**data_point, "response": ""})
        tokenized_user_prompt = tokenize(user_prompt, tokenizer, cutoff_len=cutoff_len, add_eos_token=False, add_bos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + [tokenizer.bos_token_id] + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    else:
        full_prompt = data_point["text"]
        tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len=cutoff_len, add_eos_token=True)
    return tokenized_full_prompt