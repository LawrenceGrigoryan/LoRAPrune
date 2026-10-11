import os
import random
import time

import numpy as np
import torch
from transformers import PreTrainedModel


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy, and torch (CPU/CUDA/MPS) RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Number of parameters of the deployable model.

    Excludes LoRA tensors (`lora_A`, `lora_B`, `lora_mask`): after pruning the adapters
    are merged into the base weights, so these tensors are leftovers, not model size.
    """
    return sum(p.numel() for name, p in model.named_parameters() if "lora_" not in name)


def param_summary(params_before: int, params_after: int) -> dict:
    return {
        "params_original": params_before,
        "params_pruned": params_after,
        "params_remaining_ratio": params_after / params_before,
    }


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


class InferenceTimer:
    """
    Times every forward pass of the model while active.

    Direct calls enter through the outer (PEFT) model, whose forward reaches the causal
    LM via `.forward()` and so bypasses its hooks, while `generate()` calls the inner
    `*ForCausalLM` directly. Both are hooked; a depth counter makes sure a forward that
    passes through both is timed once. The device is synchronized around each forward
    so asynchronous CUDA kernels are included.

    Usage:
        with InferenceTimer(model) as timer:
            run_benchmark(model)
        timer.summary()  # {"forward_time_s", "num_forwards", "mean_forward_time_s", "wall_time_s"}
    """

    def __init__(self, model: torch.nn.Module):
        # PreTrainedModel check skips PEFT wrappers such as `LoraPeftModelForCausalLM`
        causal_lm = next(
            (m for m in model.modules() if isinstance(m, PreTrainedModel) and type(m).__name__.endswith("ForCausalLM")),
            model,
        )
        self.modules = [model] if causal_lm is model else [model, causal_lm]
        self.forward_times: list[float] = []
        self.wall_time = 0.0
        self._handles = []
        self._depth = 0
        self._t0 = None

    def _pre_hook(self, module, args):
        self._depth += 1
        if self._depth == 1:
            _synchronize()
            self._t0 = time.perf_counter()

    def _post_hook(self, module, args, output):
        self._depth -= 1
        if self._depth == 0:
            _synchronize()
            self.forward_times.append(time.perf_counter() - self._t0)

    def __enter__(self):
        self._handles = []
        for module in self.modules:
            self._handles.append(module.register_forward_pre_hook(self._pre_hook))
            self._handles.append(module.register_forward_hook(self._post_hook))
        self._wall_start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall_time = time.perf_counter() - self._wall_start
        for handle in self._handles:
            handle.remove()
        return False

    def summary(self) -> dict:
        total = float(sum(self.forward_times))
        n = len(self.forward_times)
        return {
            "forward_time_s": total,  # pure model compute, summed over all forwards
            "num_forwards": n,
            "mean_forward_time_s": total / n if n else 0.0,
            "wall_time_s": float(self.wall_time),  # incl. tokenization, data loading, metric computation
        }


def compute_loglikelihood(prompt: str, continuation: str, model, tokenizer) -> tuple[float, int]:
    """Score LL(continuation | prompt): sum of log-probs over continuation tokens only.

    Returns (loglikelihood, num_continuation_tokens).
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=False).input_ids
    prompt_len = prompt_ids.shape[1]
    num_cont_tokens = full_ids.shape[1] - prompt_len

    input_ids = full_ids.to(model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # (1, seq_len, vocab_size)

    # logits at position i predict token at position i+1. Continuation tokens sit at
    # [prompt_len, ..., seq_len-1] and are predicted by logits at [prompt_len-1, ..., seq_len-2].
    cont_logits = logits[:, prompt_len - 1:-1, :]
    cont_labels = input_ids[:, prompt_len:]

    log_probs = torch.nn.functional.log_softmax(cont_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=cont_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum().item(), num_cont_tokens