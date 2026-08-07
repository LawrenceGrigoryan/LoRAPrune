import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy, and torch (CPU/CUDA/MPS) RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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