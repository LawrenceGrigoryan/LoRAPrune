"""
Judge generated When2Call responses with an LLM into five classes.

Adapted from ``evaluation/When2Call/evaluation/llm_as_a_judge/run_openai_judge.py``.
The important difference is the fifth class, ``invalid``. The upstream prompt
offers only the four real categories, so a judge handed incoherent output *must*
file it under a valid label — and a degraded model collects credit for garbage,
which is exactly the failure mode that makes the loglikelihood MCQ score
untrustworthy for pruned models.

The judge decides everything, including whether an unavailable tool was called.
Classification is constrained by a strict JSON schema (``When2CallJudgeOutput``),
so a reply that decodes at all already carries one of the five valid classes.
"""

import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, get_args

import fire
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

# The dataset's gold vocabulary differs from the judge's for the `direct` class.
JUDGE_TO_GOLD = {
    "direct_answer": "direct",
    "tool_call": "tool_call",
    "request_for_info": "request_for_info",
    "cannot_answer": "cannot_answer",
    "invalid": "invalid",
}
# Categories that actually occur as gold labels in the test split.
GOLD_LABELS = ("tool_call", "request_for_info", "cannot_answer")


JudgeClass = Literal["direct_answer", "tool_call", "request_for_info", "cannot_answer", "invalid"]


class When2CallJudgeOutput(BaseModel):
    rationale: str = Field(description="One sentence on why the response is classified as such.")
    classification: JudgeClass = Field(description="One of the five classes, as a string.")
    coherence: Literal[1, 2, 3, 4, 5] = Field(description="1=not coherent, 5=fully coherent")


JUDGE_CLASSES = get_args(JudgeClass)

JUDGE_PROMPT = """You are an expert at classifying responses from AI models.

An AI model was given a user question and a set of tools it may call. Your task is to \
classify what the model's response actually did, into exactly one of five categories:

(1) direct_answer: The model answered the user's question from its own knowledge, without \
requesting more information and without using a tool.
(2) tool_call: The model used, or stated that it is using, one of the available tools to \
answer the question.
(3) request_for_info: The model asked the user for additional information it needs before \
it can proceed.
(4) cannot_answer: The model declined, acknowledging it lacks the capability or the \
necessary tool to help with this request.
(5) invalid: The response is not a usable reply at all. Choose this when an unavailable tool was used or \
the response is empty, unintelligible, degenerate or repetitive (loops, repeated phrases, runaway text), \
written in a different language than the question, cut off before it says anything \
meaningful, consists of chat-template or markup tokens, or merely restates the question or \
the tool definitions instead of responding.

Important: do not judge whether the response is factually correct, or whether it picked the \
best tool. Judge only what kind of response it is. A fluent but wrong answer is still \
direct_answer. An incoherent response is invalid, never one of the other four — do not try \
to guess what a broken response was trying to say.

- The tools available to the AI model are given in <AVAILABLE_TOOLS></AVAILABLE_TOOLS>
- The user's question is given in <USER_QUESTION></USER_QUESTION>
- The AI model's response is given in <AI_MODEL_RESPONSE></AI_MODEL_RESPONSE>

<AVAILABLE_TOOLS>
{tools}
</AVAILABLE_TOOLS>

<USER_QUESTION>
{question}
</USER_QUESTION>

<AI_MODEL_RESPONSE>
{response}
</AI_MODEL_RESPONSE>

Respond with only a JSON object.
"""


def build_judge_prompt(tools, question: str, response: str) -> str:
    """
    Render the judge prompt.

    Tools go through ``json.dumps`` rather than upstream's ``str(list_of_dicts)``,
    which leaked Python repr quoting into the prompt. An empty response is spelled
    out, since it would otherwise render as blank space inside the tags.
    """
    decoded = []
    for tool in tools or []:
        if isinstance(tool, str):
            try:
                decoded.append(json.loads(tool))
                continue
            except json.JSONDecodeError:
                pass
        decoded.append(tool)
    return JUDGE_PROMPT.format(
        tools=json.dumps(decoded, indent=2) if decoded else "(no tools are available)",
        question=question,
        response=response if (response and response.strip()) else "(the model returned an empty response)",
    )


def _make_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env before judging.")
    return OpenAI(api_key=api_key)


def _judge_once(client: OpenAI, judge_model: str, prompt: str, temperature: float) -> dict:
    """
    Judge one item.

    Structured output does the parsing: the request carries a strict JSON schema
    derived from ``When2CallJudgeOutput``, so a reply that decodes at all is
    guaranteed to have every field and a ``classification`` drawn from
    ``JUDGE_CLASSES``. Only two things can still go wrong, and neither is a
    formatting problem a retry would fix.
    """
    completion = client.chat.completions.parse(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format=When2CallJudgeOutput,
    )
    choice = completion.choices[0]
    if choice.message.refusal:
        raise ValueError(f"judge refused: {choice.message.refusal}")
    if choice.message.parsed is None:
        # Most often finish_reason == "length": the object was cut off mid-generation.
        raise ValueError(f"judge returned no parsed output (finish_reason={choice.finish_reason})")
    return choice.message.parsed.model_dump()


def _macro_f1(gold: list[str], pred: list[str], labels=None) -> float:
    """
    Macro-F1 over the gold-supported classes.

    ``invalid`` and ``direct`` never appear as gold, so averaging over them would
    score every model against classes it cannot possibly get right. They are
    still fully penalised: predicting one is a false negative for the true class
    and can never be a true positive.

    Labels default to those actually present in ``gold`` rather than the fixed
    three, so a subset run (``--limit``) that happens to miss a class is not
    scored against an empty class.
    """
    if labels is None:
        labels = [label for label in GOLD_LABELS if label in set(gold)]
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def summarize(results: list[dict]) -> dict:
    """
    Aggregate judged verdicts into the reported metrics.

    Items whose judge call failed carry no label and are counted in ``n_errors``
    rather than scored as wrong, so a broken run does not read as a bad model.
    """
    scored = [r for r in results if r.get("label")]
    if not scored:
        return {"n": 0, "n_errors": len(results)}
    gold = [r["gold"] for r in scored]
    pred = [r["label"] for r in scored]

    confusion: dict[str, dict[str, int]] = {}
    for g, p in zip(gold, pred):
        confusion.setdefault(g, {})
        confusion[g][p] = confusion[g].get(p, 0) + 1

    per_class = {}
    for label in GOLD_LABELS:
        subset = [(g, p) for g, p in zip(gold, pred) if g == label]
        if subset:
            per_class[label] = {"n": len(subset), "accuracy": sum(g == p for g, p in subset) / len(subset)}

    coherences = [r["coherence"] for r in scored if isinstance(r.get("coherence"), int)]
    summary = {
        "n": len(scored),
        "n_errors": len(results) - len(scored),
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(scored),
        "macro_f1": _macro_f1(gold, pred),
        "invalid_rate": sum(p == "invalid" for p in pred) / len(scored),
        "mean_coherence": sum(coherences) / len(coherences) if coherences else None,
        "prediction_distribution": dict(Counter(pred)),
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion,
    }

    tool_gold = [r for r in scored if r["gold"] == "tool_call"]
    if tool_gold:
        # The failure the MCQ metric rewards: gold says call a tool, model abstains.
        summary["over_abstention_rate"] = sum(
            r["label"] in ("cannot_answer", "request_for_info") for r in tool_gold
        ) / len(tool_gold)
    return summary


def main(
    responses_path: str = "",
    results_path: str | None = None,
    judge_model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    max_workers: int = 8,
    limit: int | None = None,
) -> None:
    """Classify generated When2Call responses into five categories.

    Args:
        responses_path: ``when2call_responses.jsonl`` from ``eval_when2call_inference.py``.
        results_path: Output JSONL. Defaults to ``when2call_judge.jsonl`` beside
            the responses. Reruns resume by uuid, so an interrupted run continues.
        judge_model: OpenAI model to judge with.
        temperature: Judge sampling temperature; 0.0 keeps verdicts stable.
        max_workers: Concurrent judge requests.
        limit: Judge only the first N unjudged items (for a cheap trial run).

    Writes the per-item verdicts and a ``when2call_judge_summary.json`` beside them.
    """
    assert responses_path, "Please specify --responses_path"
    logger.info(
        "Parameters:\n"
        f"  responses_path={responses_path!r}\n"
        f"  judge_model={judge_model!r}\n"
        f"  temperature={temperature}\n"
        f"  max_workers={max_workers}\n"
        f"  limit={limit}"
    )

    base_dir = os.path.dirname(responses_path)
    results_path = results_path or os.path.join(base_dir, "when2call_judge.jsonl")

    with open(responses_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(records)} responses from {responses_path}")

    done: dict[str, dict] = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            done = {r["uuid"]: r for r in (json.loads(line) for line in f if line.strip())}
        logger.info(f"Resuming: {len(done)} items already judged")

    pending = [r for r in records if r["uuid"] not in done]
    if limit is not None:
        pending = pending[:limit]
    logger.info(f"Judging {len(pending)} responses with {judge_model!r}")

    if not pending:
        logger.info("Nothing to judge.")
        return

    client = _make_client()
    judged: list[dict] = [None] * len(pending)
    lock = threading.Lock()
    progress = tqdm(total=len(pending))

    def run(index: int) -> None:
        record = pending[index]
        entry = {
            "uuid": record["uuid"],
            "gold": record.get("correct_answer"),
            "model_response": record.get("model_response"),
        }
        prompt = build_judge_prompt(record.get("tools"), record.get("question"), record.get("model_response") or "")
        try:
            verdict = _judge_once(client, judge_model, prompt, temperature)
            entry.update(verdict, label=JUDGE_TO_GOLD[verdict["classification"]])
        except Exception as exc:  # one bad item must not sink the run
            logger.warning(f"Judge failed for {entry['uuid']}: {exc}")
            entry.update(classification=None, label=None, error=str(exc))
        finally:
            judged[index] = entry
            with lock:
                progress.update(1)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(run, range(len(pending))))
    progress.close()

    with open(results_path, "a") as f:
        for entry in judged:
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Wrote {len(judged)} verdicts to {results_path}")

    summary = summarize(list(done.values()) + judged)
    summary_path = os.path.join(base_dir, "when2call_judge_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # A summary computed over the handful of items that happened to succeed looks
    # like a result but is noise, and it silently changes denominators between
    # runs. Say so loudly rather than letting it be read as model quality.
    n_scored, n_errors = summary.get("n", 0), summary.get("n_errors", 0)
    if n_errors:
        logger.error(
            f"{n_errors}/{n_scored + n_errors} items failed to judge; metrics below cover "
            f"only the {n_scored} that succeeded and are NOT comparable across runs. "
            f"Inspect the `error` field in {results_path}."
        )
        for entry in judged:
            if entry.get("error"):
                logger.error(f"  first error ({entry['uuid'][:8]}): {entry['error'][:200]}")
                break

    logger.info(f"Summary written to {summary_path}")
    logger.info(f"n:                    {summary.get('n', 0)}")
    logger.info(f"accuracy:             {summary.get('accuracy', 0):.4f}")
    logger.info(f"macro_f1:             {summary.get('macro_f1', 0):.4f}")
    logger.info(f"invalid_rate:         {summary.get('invalid_rate', 0):.4f}")
    if summary.get("over_abstention_rate") is not None:
        logger.info(f"over_abstention_rate: {summary['over_abstention_rate']:.4f}")
    if summary.get("mean_coherence") is not None:
        logger.info(f"mean_coherence:       {summary['mean_coherence']:.2f}")
    logger.info(f"predictions:          {summary.get('prediction_distribution', {})}")
    logger.info(f"per-class accuracy:   {summary.get('per_class_accuracy', {})}")


if __name__ == "__main__":
    fire.Fire(main)
