"""Tests for loraprune/data_utils.py — dataset schema normalisation and tool-calling samples."""

import json

import pytest
from datasets import Dataset, DatasetDict, concatenate_datasets

from loraprune.data_utils import (
    format_tool_calls,
    format_tools,
    normalize_schema,
    prepare_tool_calling_sample,
)

# xLAM stores every field as a JSON *string*, which is what makes the
# double-encoding mistake easy to make
TOOLS = json.dumps([
    {"name": "get_weather", "description": "Weather by city.", "parameters": {"city": {"type": "str"}}},
    {"name": "get_time", "description": "Local time by city.", "parameters": {"city": {"type": "str"}}},
])
ANSWERS = json.dumps([
    {"name": "get_weather", "arguments": {"city": "Berlin"}},
    {"name": "get_time", "arguments": {"city": "Berlin"}},
])
XLAM_ROW = {"id": 0, "query": "Weather and time in Berlin?", "answers": ANSWERS, "tools": TOOLS}


def _alpaca(rows):
    return Dataset.from_list(rows)


def test_alpaca_maps_to_instruction_response():
    data = _alpaca([
        {"instruction": "Sort these.", "input": "3, 1, 2", "output": "1, 2, 3", "text": "ignored"},
    ])

    out = normalize_schema(data)

    assert out.column_names == ["instruction", "response"]
    assert out[0]["instruction"] == "Sort these.\nInput: 3, 1, 2"
    assert out[0]["response"] == "1, 2, 3"


def test_empty_input_leaves_instruction_untouched():
    data = _alpaca([
        {"instruction": "Name a color.", "input": "", "output": "Blue", "text": "ignored"},
    ])

    assert normalize_schema(data)[0]["instruction"] == "Name a color."


def test_lamini_schema_passes_through_unchanged():
    data = Dataset.from_list([{"instruction": "Hi", "response": "Hello"}])

    out = normalize_schema(data)

    assert out is data
    assert out.column_names == ["instruction", "response"]


def test_datasetdict_normalised_across_splits():
    rows = [{"instruction": "Sort these.", "input": "3, 1", "output": "1, 3", "text": "x"}]
    data = DatasetDict({"train": _alpaca(rows), "test": _alpaca(rows)})

    out = normalize_schema(data)

    for split in out.values():
        assert split.column_names == ["instruction", "response"]
        assert split[0]["response"] == "1, 3"


def test_unknown_schema_raises():
    data = Dataset.from_list([{"text": "just a document"}])

    with pytest.raises(ValueError, match="Unsupported dataset schema"):
        normalize_schema(data)


def test_tools_are_not_double_encoded():
    """xLAM's `tools` is already JSON; dumping it again escapes the whole list."""
    tools_str = format_tools(TOOLS)

    assert '\\"' not in tools_str  # no escaped quotes from a second encoding
    assert not tools_str.startswith('"')  # not one quoted blob
    lines = tools_str.splitlines()
    assert len(lines) == 2  # one signature per line
    assert [json.loads(line)["name"] for line in lines] == ["get_weather", "get_time"]


def test_one_tool_call_block_per_call():
    rendered = format_tool_calls(ANSWERS)

    assert rendered.count("<tool_call>") == 2
    assert rendered.count("</tool_call>") == 2
    calls = [json.loads(part.split("</tool_call>")[0]) for part in rendered.split("<tool_call>")[1:]]
    assert [c["name"] for c in calls] == ["get_weather", "get_time"]
    assert calls[0]["arguments"] == {"city": "Berlin"}


def test_prepare_tool_calling_sample_turns():
    sample = prepare_tool_calling_sample(XLAM_ROW)

    assert set(sample) == {"system", "user", "assistant"}
    assert sample["user"] == XLAM_ROW["query"]
    # signatures land in the system turn, calls in the assistant turn
    assert "<tools>" in sample["system"] and "get_weather" in sample["system"]
    assert sample["assistant"].startswith("<tool_call>")


def test_mixed_sets_concatenate_on_tokenized_columns_only():
    """
    Instruction and tool-calling rows keep different leftover columns, so the
    mix has to be cut down to what training reads before concatenating.
    """
    instruction = Dataset.from_list([
        {"instruction": "q", "response": "a", "input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]},
    ])
    tool = Dataset.from_list([
        {**XLAM_ROW, "input_ids": [3, 4], "attention_mask": [1, 1], "labels": [3, 4]},
    ])

    keep = ["input_ids", "attention_mask", "labels"]
    mixed = concatenate_datasets([instruction.select_columns(keep), tool.select_columns(keep)])

    assert mixed.column_names == keep
    assert len(mixed) == len(instruction) + len(tool)
    assert sorted(row["input_ids"] for row in mixed) == [[1, 2], [3, 4]]
