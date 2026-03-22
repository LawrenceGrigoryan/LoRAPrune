import time
from typing import List

import fire
import torch
import numpy as np
from peft.utils.save_and_load import load_peft_weights
from peft.peft_model import set_peft_model_state_dict
from datasets import load_dataset
from loguru import logger
from torch.utils.data.dataset import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from loraprune.peft_model import get_peft_model
from loraprune.utils import freeze, prune_from_checkpoint
from loraprune.lora import LoraConfig
from data_utils import prepare_tokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


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
    cutoff_len: int = 128
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    logger.info(f"Using device: `{device}`")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, legacy=False)
    prepare_tokenizer(tokenizer, model.config.model_type)

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
        set_peft_model_state_dict(model, adapter_weights)
        
        model = model.to(device)
        
        freeze(model)
        prune_from_checkpoint(model)
    else:
        logger.warning("LoRA weights path is not specified, evaluating the base model...")

    model.half()  # seems to fix bugs for some users.
    model.eval()
    
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    def PPLMetric(model, loader, device=device):
        ppl = llama_eval(model, loader, device)
        print(ppl)
        return ppl

    @torch.no_grad()
    def llama_eval(model, loader, device):
        model.eval()
        nlls = []
        for batch in loader:
            batch = batch.to(device)
            with torch.cuda.amp.autocast():
                t1 = time.time()
                output = model(batch)
                times.append(time.time() - t1)
            lm_logits = output.logits

            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss)
        # print(torch.cat(nlls, dim=-1).mean())
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        return ppl.item()

    times = []
    eval_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_dataset = process_data(eval_data, tokenizer, cutoff_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    results = PPLMetric(model, loader=test_loader)
    times = np.mean(times)
    print("wikitext2 PPL: {:.2f}  inference time: {:2f}".format(results, times))

    times = []
    eval_data = load_dataset('ptb_text_only', 'penn_treebank', split='validation', trust_remote_code=True)
    test_dataset = process_data(eval_data, tokenizer, cutoff_len, 'sentence')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    results = PPLMetric(model, loader=test_loader)
    times = np.mean(times)
    print("PTB PPL: {:.2f}  inference time: {:2f}".format(results, times))

    return


if __name__ == "__main__":
    fire.Fire(main)