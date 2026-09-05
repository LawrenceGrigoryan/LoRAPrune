"""Tests for loraprune/data_utils.py — dataset schema normalisation."""

import pytest
from datasets import Dataset, DatasetDict

from loraprune.data_utils import normalize_schema


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
