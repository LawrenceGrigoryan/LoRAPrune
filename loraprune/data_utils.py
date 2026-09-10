from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from loguru import logger
from textwrap import dedent
import json

# generate_and_tokenize_prompt expects LaMini-instruction's schema
SFT_COLUMNS = ("instruction", "response")
# alpaca-gpt4 (vicgalle/alpaca-gpt4) and friends
ALPACA_COLUMNS = ("instruction", "input", "output")

TOOL_SYSTEM_PROMPT = dedent("""
You are a helpful assistant that can use tools to get information for the user.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_str}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
""")


def _alpaca_to_sft(data_point: dict) -> dict:
    # `input` is an optional extra context field, populated for roughly 40% of
    # alpaca rows; fold it into the user turn so it isn't silently dropped
    instruction = data_point["instruction"]
    if data_point["input"]:
        instruction = instruction + f"\nInput: {data_point['input']}"
    return {"instruction": instruction, "response": data_point["output"]}


def normalize_schema(data: Dataset | DatasetDict) -> Dataset | DatasetDict:
    """
    Rewrite instruction-tuning datasets into the `instruction`/`response` schema
    that generate_and_tokenize_prompt expects. Datasets already in that schema
    are returned untouched.
    """
    split = next(iter(data.values())) if isinstance(data, DatasetDict) else data
    columns = split.column_names

    if all(c in columns for c in SFT_COLUMNS):
        return data

    if all(c in columns for c in ALPACA_COLUMNS):
        logger.info("Alpaca-style schema detected - mapping to instruction/response")
        drop = [c for c in columns if c not in SFT_COLUMNS]
        return data.map(_alpaca_to_sft).remove_columns(drop)

    raise ValueError(
        f"Unsupported dataset schema with columns {columns}: expected either "
        f"{list(SFT_COLUMNS)} or {list(ALPACA_COLUMNS)}."
    )


def prepare_tokenizer(tokenizer: AutoTokenizer, model_type: str, mode: str = "train") -> None:
    """
    Prepare tokenizer in-place
    """
    # FIXME: switch to sequence packing
    tokenizer.padding_side = "right"  if mode == "train" else "left"
    if model_type == "llama":  # llama-3.2-1b
        tokenizer.pad_token_id = 128004  # set to <|finetune_right_pad_id|>, different from eos
    elif model_type in ["qwen2", "qwen3"]:  # qwen-1.5-0.5b/qwen-3-0.6b
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


def format_tools(tools: str) -> str:
    """
    Render tool signatures for the <tools></tools> block: one compact JSON
    object per line, the same layout the When2Call eval prompt uses.

    xLAM stores `tools` as a JSON *string*, so it is parsed first - dumping it
    as-is would quote and escape the whole list into one unreadable line.
    """
    return "\n".join(
        json.dumps(tool, ensure_ascii=False)
        for tool in json.loads(tools)
    )


def format_tool_calls(answers: str) -> str:
    """Render the assistant turn as one <tool_call></tool_call> block per call."""
    calls = json.loads(answers)

    return "\n".join(
        "<tool_call>\n"
        + json.dumps(call, ensure_ascii=False)
        + "\n</tool_call>"
        for call in calls
    )


def prepare_tool_calling_sample(data_point: dict) -> dict:
    """
    Prepare a single tool-calling sample for training.

    Takes a raw xLAM row (`query`/`answers`/`tools`) and returns the three chat
    turns: the tool signatures go into the system prompt and the answers become
    <tool_call> blocks.
    """
    return {
        "system": TOOL_SYSTEM_PROMPT.format(tools_str=format_tools(data_point["tools"])),
        "user": data_point["query"],
        "assistant": format_tool_calls(data_point["answers"]),
    }


def generate_and_tokenize_prompt(data_point: dict, tokenizer: AutoTokenizer, model_type: str, cutoff_len: int, train_on_inputs: bool, tool_calling: bool = False):
    if not train_on_inputs:
        if model_type in ["qwen2", "qwen3"]:
            assistant_bos = "<|im_start|>assistant\n"
        elif model_type == "llama":
            assistant_bos = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            raise NotImplementedError(f"SFT Tokenization not implemented for model type: {model_type}")

        if not tool_calling:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": data_point["instruction"]},
                {"role": "assistant", "content": data_point["response"]}
            ]
        else:
            # raw xLAM row: `query`/`answers`/`tools` rather than instruction/response
            tool_call_sample = prepare_tool_calling_sample(data_point)
            messages = [
                {"role": "system", "content": tool_call_sample["system"]},
                {"role": "user", "content": tool_call_sample["user"]},
                {"role": "assistant", "content": tool_call_sample["assistant"]}
            ]

        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,  # ignored by templates without a thinking mode
        )
        # qwen3 injects an empty reasoning block when thinking is disabled
        full_prompt = full_prompt.replace("<think>\n\n</think>\n\n", "")

        tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len=cutoff_len-1, add_eos_token=False)
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
            
    elif train_on_inputs:
        # no bos token for base models, simple next token prediction on any input
        full_prompt = data_point["text"]
        tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len=cutoff_len, add_bos_token=False, add_eos_token=True)

    return tokenized_full_prompt