"""
Tests for the When2Call judge's deterministic parts.

The LLM call itself is not tested here. What is tested is the prompt handed to
the judge and the aggregation applied to its verdicts — everything the reported
numbers depend on either side of the API call.
"""

import json

import pytest

from eval_when2call_judge import (
    GOLD_LABELS,
    JUDGE_CLASSES,
    JUDGE_TO_GOLD,
    When2CallJudgeOutput,
    _macro_f1,
    build_judge_prompt,
    summarize,
)

TOOL_SPEC = json.dumps({"name": "Movies_1_FindMovies", "description": "find movies", "parameters": {}})


# --- schema ----------------------------------------------------------------


def test_schema_constrains_classification_to_the_five_classes():
    from openai.lib._parsing._completions import type_to_response_format_param

    response_format = type_to_response_format_param(When2CallJudgeOutput)
    schema = response_format["json_schema"]["schema"]
    assert response_format["json_schema"]["strict"] is True
    assert schema["properties"]["classification"]["enum"] == list(JUDGE_CLASSES)
    assert schema["properties"]["coherence"]["enum"] == [1, 2, 3, 4, 5]


def test_rationale_is_generated_before_the_classification():
    """Strict structured output emits keys in schema order, so declaring
    `rationale` first is what makes the judge reason before it labels."""
    assert list(When2CallJudgeOutput.model_fields) == ["rationale", "classification", "coherence"]


def test_every_judge_class_maps_to_a_gold_label():
    assert set(JUDGE_CLASSES) == set(JUDGE_TO_GOLD)


# --- prompt ----------------------------------------------------------------


def test_prompt_renders_tools_as_json_not_python_repr():
    prompt = build_judge_prompt([TOOL_SPEC], "Any movies?", "Sure.")
    assert "Movies_1_FindMovies" in prompt
    assert "'name':" not in prompt  # upstream's str(list_of_dicts) leaked repr quoting


def test_prompt_spells_out_an_empty_response_and_an_empty_tool_list():
    prompt = build_judge_prompt([], "Any movies?", "")
    assert "(no tools are available)" in prompt
    assert "(the model returned an empty response)" in prompt


def test_prompt_defines_the_invalid_class_including_unavailable_tools():
    prompt = build_judge_prompt([TOOL_SPEC], "Any movies?", "Sure.")
    assert "invalid" in prompt
    # The tool-availability rule lives in the prompt now, not in code.
    assert "unavailable tool" in prompt


def test_prompt_accepts_already_decoded_tool_dicts():
    prompt = build_judge_prompt([json.loads(TOOL_SPEC)], "Any movies?", "Sure.")
    assert "Movies_1_FindMovies" in prompt


# --- macro-F1 --------------------------------------------------------------


def test_invalid_predictions_are_penalised_without_inventing_a_class():
    gold = ["tool_call", "cannot_answer", "request_for_info"] * 4
    assert _macro_f1(gold, gold) == 1.0  # a perfect run is not capped by unused label slots
    assert _macro_f1(gold, ["invalid"] * len(gold)) == 0.0


def test_macro_f1_ignores_classes_absent_from_gold():
    # A --limit run may not cover all three classes; it must not be scored
    # against a class that never appears.
    assert _macro_f1(["tool_call"] * 4, ["tool_call"] * 4) == 1.0


def test_abstaining_on_everything_does_not_beat_answering_correctly():
    gold = ["tool_call", "cannot_answer", "request_for_info"] * 10
    assert _macro_f1(gold, gold) > _macro_f1(gold, ["cannot_answer"] * 30)


# --- summary ---------------------------------------------------------------


def _verdict(uuid, gold, label, coherence=5):
    return {"uuid": uuid, "gold": gold, "label": label, "coherence": coherence}


def test_summary_reports_the_degradation_signals():
    results = [
        _verdict("1", "tool_call", "tool_call"),
        _verdict("2", "tool_call", "invalid", coherence=1),
        _verdict("3", "tool_call", "cannot_answer"),
        _verdict("4", "cannot_answer", "cannot_answer"),
        {"uuid": "5", "gold": "request_for_info", "label": None, "error": "boom"},
    ]
    summary = summarize(results)
    # The failed item is excluded, not silently scored as wrong.
    assert summary["n"] == 4 and summary["n_errors"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["invalid_rate"] == 0.25
    assert summary["over_abstention_rate"] == pytest.approx(1 / 3)
    assert summary["per_class_accuracy"]["tool_call"] == {"n": 3, "accuracy": pytest.approx(1 / 3)}
    assert summary["confusion_matrix"]["tool_call"] == {"tool_call": 1, "invalid": 1, "cannot_answer": 1}
    assert summary["mean_coherence"] == pytest.approx(4.0)


def test_summary_is_empty_when_every_item_failed():
    summary = summarize([{"uuid": "1", "gold": "tool_call", "label": None, "error": "boom"}])
    assert summary == {"n": 0, "n_errors": 1}


def test_over_abstention_is_absent_when_no_tool_call_gold_items():
    summary = summarize([_verdict("1", "cannot_answer", "cannot_answer")])
    assert "over_abstention_rate" not in summary
