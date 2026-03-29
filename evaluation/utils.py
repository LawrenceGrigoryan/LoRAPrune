import torch


def compute_loglikelihood(prompt: str, model, tokenizer) -> float:
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"] # shape: (1, seq_len)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)

    # Shift: logits at position i predict token at position i+1
    shift_logits = logits[:, :-1, :]         # (1, seq_len-1, vocab_size)
    shift_labels = input_ids[:, 1:]          # (1, seq_len-1)

    # Compute log softmax over vocab dimension
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather the log prob of the actual next token at each position
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)  # (1, seq_len-1, 1)
    ).squeeze(-1)  # (1, seq_len-1)

    return token_log_probs.sum().item()  # scalar: total log-likelihood