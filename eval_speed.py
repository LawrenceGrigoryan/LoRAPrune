import json
import os
import statistics
import time
from typing import List

import fire
import torch
from peft.utils.save_and_load import load_peft_weights
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from loraprune.peft_model import get_peft_model
from loraprune.utils import freeze, prune_from_checkpoint
from loraprune.lora import LoraConfig
from loraprune.data_utils import prepare_tokenizer
from evaluation.utils import _synchronize, count_parameters, param_summary, seed_everything

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def _as_list(value) -> list[int]:
    # fire parses "1,8,32" into a tuple and "8" into an int
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in str(value).split(",")]


def _reset_peak_memory():
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb():
    return torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None


def _timed(fn, warmup: int, repeats: int) -> list[float]:
    """Run `fn` `warmup` times untimed, then `repeats` times timed; the device is
    synchronized around every timed call so asynchronous kernels are included."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        _synchronize()
        t0 = time.perf_counter()
        fn()
        _synchronize()
        times.append(time.perf_counter() - t0)
    return times


def _stats(times: list[float]) -> dict:
    return {
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "std_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_s": min(times),
    }


@torch.inference_mode()
def benchmark(model, tokenizer, batch_sizes, seq_lens, gen_prompt_len, gen_tokens, warmup, repeats):
    """
    Prefill: one forward pass over a (batch_size, seq_len) batch of random tokens,
    latency in seconds per forward pass, throughput in processed tokens/s.

    Decode: greedy `generate()` of exactly `gen_tokens` new tokens after a
    `gen_prompt_len`-token prompt, per batch size. Decode time is generate time minus
    the prefill time of the prompt, throughput in generated tokens/s.
    """
    model.eval()
    vocab_size = model.config.vocab_size
    results = {"prefill": [], "decode": []}

    for bs in batch_sizes:
        for seq_len in seq_lens:
            entry = {"batch_size": bs, "seq_len": seq_len}
            try:
                _reset_peak_memory()
                input_ids = torch.randint(0, vocab_size, (bs, seq_len), device=device)
                times = _timed(lambda: model(input_ids=input_ids), warmup, repeats)
                stats = _stats(times)
                entry.update(stats)
                entry["tokens_per_s"] = bs * seq_len / stats["median_s"]
                entry["peak_memory_mb"] = _peak_memory_mb()
                logger.info(
                    f"prefill bs={bs} seq_len={seq_len}: {stats['median_s'] * 1000:.2f} ms "
                    f"({entry['tokens_per_s']:.0f} tok/s)"
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"prefill bs={bs} seq_len={seq_len}: OOM")
                entry["oom"] = True
            results["prefill"].append(entry)

    if gen_tokens > 0:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for bs in batch_sizes:
            entry = {"batch_size": bs, "prompt_len": gen_prompt_len, "new_tokens": gen_tokens}
            try:
                _reset_peak_memory()
                input_ids = torch.randint(0, vocab_size, (bs, gen_prompt_len), device=device)
                attention_mask = torch.ones_like(input_ids)

                def generate():
                    return model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=gen_tokens,
                        min_new_tokens=gen_tokens,  # never stop early on EOS
                        do_sample=False,
                        pad_token_id=pad_token_id,
                    )

                gen_times = _timed(generate, warmup=1, repeats=repeats)
                prefill_times = _timed(lambda: model(input_ids=input_ids), warmup=1, repeats=repeats)
                gen_median = statistics.median(gen_times)
                decode_time = gen_median - statistics.median(prefill_times)
                entry.update(_stats(gen_times))
                entry["decode_time_s"] = decode_time
                entry["decode_tokens_per_s"] = bs * gen_tokens / decode_time
                entry["ms_per_token"] = decode_time / gen_tokens * 1000
                entry["peak_memory_mb"] = _peak_memory_mb()
                logger.info(
                    f"decode bs={bs}: {entry['ms_per_token']:.2f} ms/token "
                    f"({entry['decode_tokens_per_s']:.0f} tok/s)"
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"decode bs={bs}: OOM")
                entry["oom"] = True
            results["decode"].append(entry)

    return results


def speedup(base: dict, pruned: dict) -> dict:
    """Pruned throughput / base throughput for every config that ran in both."""
    out = {"prefill": [], "decode": []}
    for b, p in zip(base["prefill"], pruned["prefill"]):
        if "tokens_per_s" in b and "tokens_per_s" in p:
            out["prefill"].append({
                "batch_size": b["batch_size"],
                "seq_len": b["seq_len"],
                "speedup": p["tokens_per_s"] / b["tokens_per_s"],
            })
    for b, p in zip(base["decode"], pruned["decode"]):
        if "decode_tokens_per_s" in b and "decode_tokens_per_s" in p:
            out["decode"].append({
                "batch_size": b["batch_size"],
                "speedup": p["decode_tokens_per_s"] / b["decode_tokens_per_s"],
            })
    return out


def main(
    base_model: str = "",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.,
    lora_target_modules: List[str] = [
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj"
        ],
    lora_weights: str | None = None,
    granular_gqa: bool = False,
    benchmark_base: bool = True,
    batch_sizes: str = "1,8,32",
    seq_lens: str = "128,512,2048",
    gen_prompt_len: int = 128,
    gen_tokens: int = 128,
    warmup: int = 5,
    repeats: int = 20,
    output_dir: str = "./outputs_dir/evaluation/speed/",
    seed: int = 42,
):
    """
    Measures inference speed of the base model and, if `lora_weights` is given, of the
    pruned model. With `benchmark_base=True` the unpruned model is benchmarked first in
    the same process, so both run on the same GPU and the speedup is directly comparable.
    """
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='./models/meta_llama_Llama-3.2-1B'"
    batch_sizes = _as_list(batch_sizes)
    seq_lens = _as_list(seq_lens)

    logger.info(
        "Parameters:\n"
        f"  base_model={base_model!r}\n"
        f"  lora_r={lora_r}\n"
        f"  lora_alpha={lora_alpha}\n"
        f"  lora_dropout={lora_dropout}\n"
        f"  lora_target_modules={lora_target_modules}\n"
        f"  lora_weights={lora_weights!r}\n"
        f"  granular_gqa={granular_gqa}\n"
        f"  benchmark_base={benchmark_base}\n"
        f"  batch_sizes={batch_sizes}\n"
        f"  seq_lens={seq_lens}\n"
        f"  gen_prompt_len={gen_prompt_len}\n"
        f"  gen_tokens={gen_tokens}\n"
        f"  warmup={warmup}\n"
        f"  repeats={repeats}\n"
        f"  output_dir={output_dir!r}\n"
        f"  seed={seed}"
    )
    seed_everything(seed)
    logger.info(f"Using device: `{device}`")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, legacy=False)
    prepare_tokenizer(tokenizer, model.config.model_type, mode="inference")

    total_params = count_parameters(model)
    logger.info(f"Total model parameters: {total_params}")
    bench_args = dict(
        tokenizer=tokenizer,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        gen_prompt_len=gen_prompt_len,
        gen_tokens=gen_tokens,
        warmup=warmup,
        repeats=repeats,
    )

    output = {
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "dtype": str(model.dtype),
    }

    if not lora_weights or benchmark_base:
        logger.info("Benchmarking base model...")
        output["base"] = benchmark(model, **bench_args)

    if lora_weights:
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

        adapter_weights = load_peft_weights(lora_weights)
        for name, param in adapter_weights.items():
            if 'lora_mask' in name:
                adapter_weights[name] = param.reshape(-1)

        # inject only adapter state dict
        # will return missing keys warning for base model's layers that
        # are not in the adapter state dict
        model.load_state_dict(adapter_weights, strict=False)

        model.to(device)

        freeze(model)
        # merges LoRA into the base weights and physically removes pruned rows/columns,
        # so the forward below is plain (smaller) linear layers
        prune_from_checkpoint(model, granular_gqa=granular_gqa)
        model.half()

        total_params_pruned = count_parameters(model)
        logger.info(f"Parameters left after pruning: {round((total_params_pruned / total_params)*100, 2)}%")
        output["params"] = param_summary(total_params, total_params_pruned)

        logger.info("Benchmarking pruned model...")
        output["pruned"] = benchmark(model, **bench_args)
        if "base" in output:
            output["speedup"] = speedup(output["base"], output["pruned"])
            for s in output["speedup"]["prefill"]:
                logger.info(f"prefill speedup bs={s['batch_size']} seq_len={s['seq_len']}: {s['speedup']:.2f}x")
            for s in output["speedup"]["decode"]:
                logger.info(f"decode speedup bs={s['batch_size']}: {s['speedup']:.2f}x")
    else:
        output["params"] = {"params_original": total_params}

    name = os.path.basename(os.path.normpath(lora_weights)) if lora_weights else os.path.basename(os.path.normpath(base_model))
    save_path = os.path.join(output_dir, name, "speed.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {save_path}")


if __name__ == "__main__":
    fire.Fire(main)
