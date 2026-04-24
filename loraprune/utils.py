import numpy as np
import torch
from .lora import Linear
from loguru import logger

pruning_groups = {'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                  'mlp': ['up_proj', 'gate_proj'],
                  'block': ['o_proj', 'down_proj']}

NUM_ATTENTION_HEADS = 16
HEAD_DIM = 64
NUM_KV_HEADS = 16

def _is_target_larer(module):
    return isinstance(module, Linear) and module.is_prune

def unfreeze(model):
    for _, module in model.named_modules():
        if _is_target_larer(module):
            module.weight.requires_grad = True

def freeze(model):
    layers = len(model.model.model.layers)
    freeze_layer = int(layers * 0.1)
    for name, module in model.named_modules():
        if _is_target_larer(module):
            layer = int(name.split('.')[4])
            if layer < freeze_layer or layer == layers-1:
                module.is_prune = False

def init_sensitivity_dict(model):
    sensitivity_record = {}
    for name, module in model.named_modules():
        if _is_target_larer(module):
            weight_name = name.split('.')[-1]
            if weight_name in pruning_groups['self_attn']:
                groups = module.out_features // HEAD_DIM
            else:
                groups = module.out_features
                
            # different num of heads in GQA => can't add up improtance for the whole block together
            group_name = ".".join(name.split('.')[:-1])

            if group_name in sensitivity_record:
                continue
            
            sensitivity_record[group_name] = module.lora_A.weight.data.new_zeros(groups)
    return sensitivity_record

def update_sensitivity_dict(model, s_dict, pruning_type):
    s_all = init_sensitivity_dict(model)
    for name, module in model.named_modules():
        if _is_target_larer(module):
            weight_name = name.split('.')[-1]
            is_attn = weight_name in pruning_groups['self_attn']
            fan_in = weight_name in pruning_groups['block']
            s = compute_sensitivity(module, is_attn, pruning_type, fan_in)
            
            # different num of heads in GQA => can't add up improtance for the whole block together
            group_name = ".".join(name.split('.')[:-1])
            
            s_all[group_name] += s

    for group_name, imp in s_all.items():
        if torch.isnan(imp.sum()) or torch.isinf(imp.sum()):
            import warnings
            warnings.warn(f"NaN/inf sensitivity detected for group '{group_name}', skipping sensitivity update for this step.")
            return s_dict

    for group_name, imp in s_dict.items():
        s_dict[group_name] = imp * 0.9 + s_all[group_name] * 0.1

    return s_dict

def compute_sensitivity(layer, is_attn, prune_metric='lora', transpose=False, norm=True):
    a = layer.lora_A.weight.data
    b = layer.lora_B.weight.data
    if prune_metric == 'lora':
        grad_a = layer.lora_A.weight.grad
        grad_b = layer.lora_B.weight.grad
        grad_a = torch.nan_to_num(grad_a, nan=0.0, posinf=0.0, neginf=0.0)
        grad_b = torch.nan_to_num(grad_b, nan=0.0, posinf=0.0, neginf=0.0)
        grad = (grad_b @ a + b @ grad_a - grad_b @ grad_a)
    elif prune_metric == 'magnitude':
        grad = 1
    elif prune_metric == 'grad':
        grad = layer.weight.grad
    else:
        raise NotImplementedError
    if hasattr(layer, 'state'):
        weight = (layer.weight.data * layer.state.SCB.reshape(-1, 1)) / 127
    else:
        weight = layer.weight.data
    s = (grad * (b @ a * layer.scaling + weight)).abs()
    if transpose:
        s = s.t()

    if is_attn:
        # FIXME: all heads have the same head_dim, so hardcoding is fine for now
        s = s.reshape(s.shape[0] // HEAD_DIM, -1)

    s = s.sum(1)
    if norm:
        s = s / (torch.linalg.norm(s) + 1e-8)
    return s

def prune_fp16_module(module, mask, transpose):
    mask = mask.bool()
    module.train()
    if not transpose:
        module.weight.data = module.weight.data[mask]
        module.out_features = int(mask.sum())
        # none for llama-3.2, not none for qwen2
        if module.bias is not None:
            module.bias.data = module.bias.data[mask]
        module.lora_B.weight.data = module.lora_B.weight.data[mask]
        module.lora_B.out_features = int(mask.sum())
    else:
        module.weight.data = module.weight.data[:, mask]
        module.in_features = int(mask.sum())
        module.lora_A.weight.data = module.lora_A.weight.data[:, mask]
        module.lora_A.in_features = int(mask.sum())
    module.merge_weights = True
    module.train(False)

def prune_one_layer(layer):
    ## self_attn
    prune_fp16_module(layer.self_attn.q_proj, layer.self_attn.q_proj.lora_mask, False)
    prune_fp16_module(layer.self_attn.k_proj, layer.self_attn.k_proj.lora_mask, False)
    prune_fp16_module(layer.self_attn.v_proj, layer.self_attn.v_proj.lora_mask, False)
    # q_proj out_features = o_proj in_features
    # after removing some heads o_proj rows must be removed accordingly
    prune_fp16_module(layer.self_attn.o_proj, layer.self_attn.q_proj.lora_mask, True)
    layer.self_attn.num_heads = int(layer.self_attn.q_proj.lora_mask.sum()) // HEAD_DIM
    layer.self_attn.hidden_size = int(layer.self_attn.q_proj.lora_mask.sum())
    # for GQA
    layer.self_attn.num_key_value_heads = (
        layer.self_attn.k_proj.out_features // HEAD_DIM
    )

    ## mlp
    prune_fp16_module(layer.mlp.gate_proj, layer.mlp.gate_proj.lora_mask, False)
    prune_fp16_module(layer.mlp.up_proj, layer.mlp.up_proj.lora_mask, False)
    # gate/up outputs → down inputs
    # after removing 
    prune_fp16_module(layer.mlp.down_proj, layer.mlp.gate_proj.lora_mask, True)

    ## reset mask
    del(layer.self_attn.q_proj.lora_mask)
    del(layer.self_attn.k_proj.lora_mask)
    del(layer.self_attn.v_proj.lora_mask)
    del(layer.mlp.gate_proj.lora_mask)
    del(layer.mlp.up_proj.lora_mask)
    del(layer.self_attn.o_proj.lora_mask)
    del(layer.mlp.down_proj.lora_mask)

def prune(model):
    for layer_id, layer in enumerate(model.model.model.layers):
        logger.info("pruning layer {}".format(layer_id))
        prune_one_layer(layer)

def local_prune(model, s_dict, ratio, target_ratio):
    original_param_num = 0
    pruned_param_num = 0
    for name, module in model.named_modules():
        if _is_target_larer(module):
            original_param_num += np.prod(module.weight.shape)
            pruned_param_num += np.prod(module.weight.shape) * ratio
            module_name = name.split('.')[-1]
            is_attn = module_name in pruning_groups['self_attn']
            if module_name in pruning_groups['block']:
                continue
            
            group_name = ".".join(name.split('.')[:-1])

            if not hasattr(module, 'lora_mask'):
                continue
            if (1-module.lora_mask.mean()).item() >= target_ratio:
                continue
            total_num = module.lora_mask.numel()
            c_mask = module.lora_mask.data
            mask = torch.ones_like(c_mask)

            # consider GQA
            if module_name == "q_proj":
                num_heads = NUM_ATTENTION_HEADS  # 32 for llama-3.2
            elif module_name in ["k_proj", "v_proj"]:
                num_heads = NUM_KV_HEADS  # 8 for llama-3.2

            # for attention - reshape the mask to be of size [n_heads, head_dim] to prune full heads
            if is_attn:
                head_dim = module.out_features // num_heads
                mask = mask.reshape(-1, head_dim)[:, 0]
                c_mask = c_mask.reshape(-1, head_dim)[:, 0]
                total_num /= head_dim  # convert into number of heads instead of neurons
            need_prune_num = int(total_num * ratio)
            importance = s_dict[group_name] * c_mask
            can_prune = torch.argsort(importance)[:need_prune_num]
            mask[can_prune] = 0
            if is_attn:
                mask = (mask.new_ones(module.lora_mask.shape).reshape(-1, head_dim) * mask.unsqueeze(1)).reshape(-1)
            module.lora_mask.data = mask
        else:
            if hasattr(module, 'weight'):
                original_param_num += np.prod(module.weight.shape)
    logger.info("pruned/original parameters number:{:3f}/{:3f}  ratio:{:3f}".format(pruned_param_num*1e-9,
                                                                               original_param_num*1e-9,
                                                                               pruned_param_num/original_param_num))

def schedule_sparsity_ratio(
    step,
    total_step,
    initial_warmup,
    final_warmup,
    initial_sparsity,
    final_sparsity,
):
    if step <= initial_warmup * total_step:
        sparsity = initial_sparsity
    elif step > (total_step - final_warmup * total_step):
        sparsity = final_sparsity
    else:
        spars_warmup_steps = initial_warmup * total_step
        spars_schedu_steps = (final_warmup + initial_warmup) * total_step
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (mul_coeff ** 3)
    return sparsity

def prune_from_checkpoint(model):
    prune(model)

def print_trainable_parameters(model):
    total_params = 0
    trainable_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        total_params += p.numel()
    logger.info("total params:{}   trainable params:{}    ratio:{}".format(total_params * 1e-6, trainable_params * 1e-6, trainable_params / total_params))