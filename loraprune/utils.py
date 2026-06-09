import numpy as np
import torch
from dataclasses import dataclass
from .lora import Linear
from loguru import logger

pruning_groups = {'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                  'mlp': ['up_proj', 'gate_proj'],
                  'block': ['o_proj', 'down_proj']}


@dataclass
class AttentionConfig:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    is_gqa: bool
    q_heads_per_kv: int  # num_q_heads // num_kv_heads — Q heads sharing one KV head


def get_attn_config(model) -> AttentionConfig:
    cfg = model.config
    num_q = cfg.num_attention_heads
    num_kv = getattr(cfg, 'num_key_value_heads', num_q)
    head_dim = cfg.hidden_size // num_q
    return AttentionConfig(
        num_q_heads=num_q,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        is_gqa=(num_kv < num_q),
        q_heads_per_kv=num_q // num_kv,
    )


def _is_target_layer(module):
    return isinstance(module, Linear) and module.is_prune

def unfreeze(model):
    for _, module in model.named_modules():
        if _is_target_layer(module):
            module.weight.requires_grad = True

def freeze(model):
    layers = len(model.model.model.layers)
    freeze_layer = int(layers * 0.1)
    for name, module in model.named_modules():
        if _is_target_layer(module):
            layer = int(name.split('.')[4])
            if layer < freeze_layer or layer == layers-1:
                module.is_prune = False

def init_sensitivity_dict(model, attn_cfg: AttentionConfig):
    sensitivity_record = {}
    for name, module in model.named_modules():
        if _is_target_layer(module):
            weight_name = name.split('.')[-1]
            if weight_name == 'q_proj':
                groups = attn_cfg.num_q_heads
            elif weight_name in ('k_proj', 'v_proj'):
                groups = attn_cfg.num_kv_heads
            elif weight_name in pruning_groups['self_attn']:  # o_proj
                groups = attn_cfg.num_q_heads
            elif weight_name in pruning_groups['block']:  # down_proj: sensitivity is over in_features after transpose
                groups = module.in_features
            else:
                groups = module.out_features
            sensitivity_record[name] = module.lora_A.weight.data.new_zeros(groups)
    return sensitivity_record

def update_sensitivity_dict(model, s_dict, pruning_type, attn_cfg: AttentionConfig):
    s_all = init_sensitivity_dict(model, attn_cfg)
    for name, module in model.named_modules():
        if _is_target_layer(module):
            weight_name = name.split('.')[-1]
            is_attn = weight_name in pruning_groups['self_attn']
            fan_in = weight_name in pruning_groups['block']
            if weight_name == 'q_proj':
                num_heads = attn_cfg.num_q_heads
            elif weight_name in ('k_proj', 'v_proj'):
                num_heads = attn_cfg.num_kv_heads
            else:
                num_heads = None
            s = compute_sensitivity(module, is_attn, pruning_type, fan_in,
                                    head_dim=attn_cfg.head_dim, num_heads=num_heads)
            if not torch.isfinite(s).all():
                logger.warning(f"NaN/inf in sensitivity for layer '{name}': {s}")
            s_all[name] = s

    for name, imp in s_all.items():
        if torch.isnan(imp.sum()) or torch.isinf(imp.sum()):
            logger.warning(f"NaN/inf sensitivity detected for '{name}', skipping sensitivity update for this step.")
            return s_dict

    for name in s_dict:
        s_dict[name] = s_dict[name] * 0.9 + s_all[name] * 0.1

    return s_dict

def compute_sensitivity(layer, is_attn, prune_metric='lora', transpose=False, norm=True,
                        head_dim=64, num_heads=None):
    a = layer.lora_A.weight.data
    b = layer.lora_B.weight.data
    if prune_metric == 'lora':
        grad_a = layer.lora_A.weight.grad
        grad_b = layer.lora_B.weight.grad
        grad = (grad_b @ a + b @ grad_a - grad_b @ grad_a)
        if not torch.isfinite(grad).all():
            logger.warning(f"  compute_sensitivity: grad has NaN/inf  max={grad.float().abs().max():.3e}")
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
        if num_heads is not None:
            s = s.reshape(num_heads, -1)
        else:
            s = s.reshape(s.shape[0] // head_dim, -1)

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

def prune_one_layer(layer, attn_cfg: AttentionConfig):
    hd = attn_cfg.head_dim
    ## self_attn
    prune_fp16_module(layer.self_attn.q_proj, layer.self_attn.q_proj.lora_mask, False)
    prune_fp16_module(layer.self_attn.k_proj, layer.self_attn.k_proj.lora_mask, False)
    prune_fp16_module(layer.self_attn.v_proj, layer.self_attn.v_proj.lora_mask, False)
    # q_proj out_features = o_proj in_features
    # after removing some heads o_proj rows must be removed accordingly
    prune_fp16_module(layer.self_attn.o_proj, layer.self_attn.q_proj.lora_mask, True)
    layer.self_attn.num_heads = int(layer.self_attn.q_proj.lora_mask.sum()) // hd
    layer.self_attn.hidden_size = int(layer.self_attn.q_proj.lora_mask.sum())
    layer.self_attn.num_key_value_heads = layer.self_attn.k_proj.out_features // hd

    ## mlp
    prune_fp16_module(layer.mlp.gate_proj, layer.mlp.gate_proj.lora_mask, False)
    prune_fp16_module(layer.mlp.up_proj, layer.mlp.up_proj.lora_mask, False)
    # gate/up outputs → down inputs
    prune_fp16_module(layer.mlp.down_proj, layer.mlp.gate_proj.lora_mask, True)

    ## reset mask
    del(layer.self_attn.q_proj.lora_mask)
    del(layer.self_attn.k_proj.lora_mask)
    del(layer.self_attn.v_proj.lora_mask)
    del(layer.mlp.gate_proj.lora_mask)
    del(layer.mlp.up_proj.lora_mask)
    del(layer.self_attn.o_proj.lora_mask)
    del(layer.mlp.down_proj.lora_mask)

def prune(model, attn_cfg: AttentionConfig):
    for layer_id, layer in enumerate(model.model.model.layers):
        logger.info("pruning layer {}".format(layer_id))
        prune_one_layer(layer, attn_cfg)


def _gqa_coupled_prune(q_mod, k_mod, v_mod, q_name, k_name, v_name,
                        s_dict, ratio, attn_cfg: AttentionConfig):
    """
    Prune attention heads at KV-group granularity for GQA models.

    Each pruning decision removes one atomic group: ``q_heads_per_kv`` query
    heads plus the single K and V head they share.  Group importance is the
    sum of the EMA-smoothed sensitivity scores of all member heads, masked to
    zero for groups that are already fully pruned.

    Parameters
    ----------
    q_mod : lora.Linear
        LoRA-wrapped ``q_proj`` module whose ``lora_mask`` will be updated.
    k_mod : lora.Linear
        LoRA-wrapped ``k_proj`` module whose ``lora_mask`` will be updated.
    v_mod : lora.Linear
        LoRA-wrapped ``v_proj`` module whose ``lora_mask`` will be updated.
    q_name : str
        Full module path of ``q_proj``, used to look up ``s_dict``.
    k_name : str
        Full module path of ``k_proj``, used to look up ``s_dict``.
    v_name : str
        Full module path of ``v_proj``, used to look up ``s_dict``.
    s_dict : dict[str, torch.Tensor]
        EMA sensitivity scores keyed by full module name.  For ``q_proj`` the
        tensor has shape ``[num_q_heads]``; for ``k/v_proj`` shape
        ``[num_kv_heads]``.
    ratio : float
        Fraction of KV groups to prune at this step (e.g. ``0.2`` removes
        the 20 % least-important groups).
    attn_cfg : AttentionConfig
        Attention geometry of the model (head counts, head dim, group size).
    """
    G = attn_cfg.q_heads_per_kv
    num_kv = attn_cfg.num_kv_heads
    hd = attn_cfg.head_dim

    # Aggregate Q sensitivity per KV group: [num_kv, G] -> [num_kv]
    q_sens = s_dict[q_name].reshape(num_kv, G).sum(1)
    k_sens = s_dict[k_name]   # [num_kv]
    v_sens = s_dict[v_name]   # [num_kv]

    # Zero out already-dead groups so they sort first and are re-selected into prune_groups,
    # keeping them dead in the freshly-initialised kv_mask (which starts all-ones).
    k_alive = k_mod.lora_mask.reshape(-1, hd)[:, 0].float()   # [num_kv] binary
    group_imp = (q_sens + k_sens + v_sens) * k_alive

    need_prune = int(num_kv * ratio)
    prune_groups = torch.argsort(group_imp)[:need_prune]

    kv_mask = torch.ones(num_kv, device=k_alive.device)
    kv_mask[prune_groups] = 0

    k_mod.lora_mask.data = kv_mask.unsqueeze(1).expand(-1, hd).reshape(-1).contiguous()
    v_mod.lora_mask.data = k_mod.lora_mask.data.clone()

    q_mask_heads = kv_mask.unsqueeze(1).expand(-1, G).reshape(-1)   # [num_q]
    q_mod.lora_mask.data = q_mask_heads.unsqueeze(1).expand(-1, hd).reshape(-1).contiguous()


def _gqa_fine_grained_prune(q_mod, k_mod, v_mod, q_name,
                              s_dict, ratio, attn_cfg: AttentionConfig):
    """
    Prune query heads independently and derive KV-head masks from the result.

    Query heads are ranked by their individual EMA sensitivity and the
    lowest-importance ones are pruned first, without any constraint on keeping
    KV groups intact.  After Q-head masks are updated, a KV head is kept alive
    only if at least one query head in its group survived; otherwise it is also
    pruned.  This allows more surgical Q-head removal while ensuring K/V heads
    are never computed for groups with no remaining queries.

    Parameters
    ----------
    q_mod : lora.Linear
        LoRA-wrapped ``q_proj`` module whose ``lora_mask`` will be updated.
    k_mod : lora.Linear
        LoRA-wrapped ``k_proj`` module whose ``lora_mask`` will be updated.
    v_mod : lora.Linear
        LoRA-wrapped ``v_proj`` module whose ``lora_mask`` will be updated.
    q_name : str
        Full module path of ``q_proj``, used to look up ``s_dict``.
    s_dict : dict[str, torch.Tensor]
        EMA sensitivity scores keyed by full module name.  For ``q_proj`` the
        tensor has shape ``[num_q_heads]``.
    ratio : float
        Fraction of query heads to prune at this step (e.g. ``0.2`` removes
        the 20 % least-important Q heads).
    attn_cfg : AttentionConfig
        Attention geometry of the model (head counts, head dim, group size).
    """
    G = attn_cfg.q_heads_per_kv
    num_q = attn_cfg.num_q_heads
    num_kv = attn_cfg.num_kv_heads
    hd = attn_cfg.head_dim

    q_mask = q_mod.lora_mask.reshape(-1, hd)[:, 0].clone()   # [num_q] current alive
    q_imp = s_dict[q_name] * q_mask
    need_prune = int(num_q * ratio)
    prune_q = torch.argsort(q_imp)[:need_prune]
    q_mask[prune_q] = 0

    q_mod.lora_mask.data = q_mask.unsqueeze(1).expand(-1, hd).reshape(-1).contiguous()

    # KV head alive iff any Q head in its group remains
    kv_alive = q_mask.reshape(num_kv, G).any(dim=1).float()   # [num_kv]
    k_mod.lora_mask.data = kv_alive.unsqueeze(1).expand(-1, hd).reshape(-1).contiguous()
    v_mod.lora_mask.data = k_mod.lora_mask.data.clone()


def local_prune(model, s_dict, ratio, target_ratio, attn_cfg: AttentionConfig,
                gqa_prune_mode: str = 'coupled'):
    original_param_num = 0
    pruned_param_num = 0

    if attn_cfg.is_gqa:
        # Collect q/k/v modules per attention layer and prune jointly
        attn_layers = {}
        for name, module in model.named_modules():
            if _is_target_layer(module):
                weight_name = name.split('.')[-1]
                if weight_name in ('q_proj', 'k_proj', 'v_proj'):
                    layer_path = ".".join(name.split('.')[:-1])
                    attn_layers.setdefault(layer_path, {})[weight_name] = (name, module)

        for layer_path, projs in attn_layers.items():
            if not all(k in projs for k in ('q_proj', 'k_proj', 'v_proj')):
                continue
            q_name, q_mod = projs['q_proj']
            k_name, k_mod = projs['k_proj']
            v_name, v_mod = projs['v_proj']

            if not all(m.is_prune for _, m in projs.values()):
                continue
            if not hasattr(k_mod, 'lora_mask'):
                continue
            if (1 - k_mod.lora_mask.mean()).item() >= target_ratio:
                continue

            if gqa_prune_mode == 'coupled':
                _gqa_coupled_prune(q_mod, k_mod, v_mod, q_name, k_name, v_name,
                                   s_dict, ratio, attn_cfg)
            else:
                _gqa_fine_grained_prune(q_mod, k_mod, v_mod, q_name,
                                        s_dict, ratio, attn_cfg)

    # MLP and (for MHA) attention projections — prune per-module
    for name, module in model.named_modules():
        if _is_target_layer(module):
            original_param_num += np.prod(module.weight.shape)
            pruned_param_num += np.prod(module.weight.shape) * ratio
            module_name = name.split('.')[-1]
            is_attn = module_name in pruning_groups['self_attn']

            # Skip o_proj and down_proj (they track their paired projection's mask)
            if module_name in pruning_groups['block']:
                continue
            # In GQA mode, attention projections are handled above
            if attn_cfg.is_gqa and is_attn:
                continue

            if not hasattr(module, 'lora_mask'):
                continue
            if (1 - module.lora_mask.mean()).item() >= target_ratio:
                continue

            total_num = module.lora_mask.numel()
            c_mask = module.lora_mask.data
            mask = torch.ones_like(c_mask)

            if is_attn:
                # MHA: all projections use num_q_heads
                num_heads = attn_cfg.num_q_heads
                head_dim = module.out_features // num_heads
                mask = mask.reshape(-1, head_dim)[:, 0]
                c_mask = c_mask.reshape(-1, head_dim)[:, 0]
                total_num /= head_dim

            need_prune_num = int(total_num * ratio)
            importance = s_dict[name] * c_mask
            can_prune = torch.argsort(importance)[:need_prune_num]
            mask[can_prune] = 0

            if is_attn:
                mask = (mask.new_ones(module.lora_mask.shape).reshape(-1, head_dim) * mask.unsqueeze(1)).reshape(-1)
            module.lora_mask.data = mask
        else:
            if hasattr(module, 'weight'):
                original_param_num += np.prod(module.weight.shape)

    logger.info("pruned/original parameters number:{:3f}/{:3f}  ratio:{:3f}".format(
        pruned_param_num * 1e-9, original_param_num * 1e-9,
        pruned_param_num / original_param_num))

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

def prune_from_checkpoint(model, attn_cfg: AttentionConfig):
    prune(model, attn_cfg)

def print_trainable_parameters(model):
    total_params = 0
    trainable_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        total_params += p.numel()
    logger.info("total params:{}   trainable params:{}    ratio:{}".format(total_params * 1e-6, trainable_params * 1e-6, trainable_params / total_params))
