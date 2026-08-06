import numpy as np
import torch
from loraprune.lora import Linear
from loguru import logger

_ADAPTIVE_VAR_BETA = 0.99   # smoothing for running variance tracker
_ADAPTIVE_ALPHA_MIN = 0.70  # floor: most responsive to current step
_ADAPTIVE_ALPHA_MAX = 0.95  # ceiling: most smoothing (stable groups)

pruning_groups = {'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                  'mlp': ['up_proj', 'gate_proj'],
                  'block': ['o_proj', 'down_proj']}


def is_gqa_model(model) -> bool:
    is_gqa_model = (model.config.num_attention_heads > model.config.num_key_value_heads)
    return is_gqa_model


def _is_target_layer(module):
    return isinstance(module, Linear) and module.is_prune


def unfreeze(model):
    for _, module in model.named_modules():
        if _is_target_layer(module):
            module.weight.requires_grad = True


def freeze(model):
    """
    Exclude the bottom 10% of transformer layers and the final layer from pruning.

    Sets ``module.is_prune = False`` on every LoRA-linear module that belongs to
    a protected layer, making those modules invisible to all pruning routines
    (``_is_target_layer`` returns ``False`` when ``is_prune`` is ``False``).

    Early layers encode low-level token features that are hard to recover once
    pruned; the last layer drives the vocabulary projection directly, so pruning
    its heads collapses output quality disproportionately.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        A PEFT-wrapped causal LM whose transformer blocks are accessible at
        ``model.model.model.layers``. Each LoRA leaf is expected to be an
        instance of ``loraprune.lora.Linear`` with an ``is_prune`` attribute.

    Notes
    -----
    Mutates ``module.is_prune`` in-place; no return value.
    ``unfreeze`` re-enables weight gradients but does **not** restore
    ``is_prune``, so protected layers remain excluded from pruning for the
    full training run.
    """
    layers = len(model.model.model.layers)
    freeze_layer = int(layers * 0.1)
    for name, module in model.named_modules():
        if _is_target_layer(module):
            layer = int(name.split('.')[4])
            if layer < freeze_layer or layer == layers-1:
                module.is_prune = False


def init_sensitivity_dict(model, granular_gqa=False):
    """
    Allocate a zero-initialised sensitivity accumulator for every prunable group.

    A "group" is the coarsest unit that can be removed atomically:

    * **GQA attention block** (``num_attention_heads > num_key_value_heads``):
      one slot per KV head.  Pruning KV head *i* removes the entire KV head from
      ``k_proj`` / ``v_proj`` and the corresponding group of Q heads
      (indices ``i*G … (i+1)*G-1``) from ``q_proj`` / ``o_proj``, where
      ``G = num_attention_heads // num_key_value_heads``.  All four projections
      (``q``, ``k``, ``v``, ``o``) therefore share a single accumulator of length
      ``num_key_value_heads``.

      When ``granular_gqa=True``, the accumulator has length
      ``num_attention_heads`` instead — one slot per Q head — so that each Q
      head can be scored and pruned independently before its KV head is removed.

    * **MHA attention block** (``num_attention_heads == num_key_value_heads``):
      one slot per attention head (length ``num_attention_heads``).

    * **MLP block** (``up_proj``, ``gate_proj``): one slot per intermediate
      neuron (length ``out_features``).

    The dict is keyed by the *parent module path* (everything except the final
    weight name), so all projections within one attention or MLP block map to the
    same key.  Only the first projection encountered per block creates the entry;
    subsequent projections hit the ``continue`` guard and are skipped.  This
    relies on ``named_modules()`` returning ``q_proj`` before ``k/v/o_proj``,
    which holds for standard HuggingFace LLaMA / Qwen implementations.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        A PEFT-wrapped causal LM.  ``model.config`` must expose
        ``num_attention_heads`` and ``num_key_value_heads``.
    granular_gqa : bool, optional
        When ``True`` and the model is GQA, track sensitivity at Q-head
        granularity (``num_attention_heads``) rather than KV-head granularity.
    """
    sensitivity_record = {}
    for name, module in model.named_modules():
        if _is_target_layer(module):
            weight_name = name.split('.')[-1]
            if is_gqa_model(model) and (weight_name in pruning_groups['self_attn']):
                if granular_gqa:
                    n_groups = model.config.num_attention_heads
                else:
                    n_groups = model.config.num_key_value_heads
            elif weight_name in pruning_groups['self_attn']:
                n_groups = model.config.num_attention_heads
            else:
                n_groups = module.out_features

            # keep only the layer/group name without the specific weight name like `k_proj`
            group_name = ".".join(name.split('.')[:-1])

            if group_name in sensitivity_record:
                continue

            sensitivity_record[group_name] = module.lora_A.weight.data.new_zeros(n_groups)
    return sensitivity_record


def init_adaptive_ema_state(model, granular_gqa=False):
    """
    Allocate auxiliary state for adaptive per-group EMA (used when ``adaptive_ema=True``).

    Returns three dicts keyed by the same group names as ``init_sensitivity_dict``:

    * ``var_dict``   – zero-initialised running element-wise variance tensors (same
                       shape and device as the corresponding ``s_dict`` entry).
    * ``alpha_dict`` – float scalars per group, initialised to ``_ADAPTIVE_ALPHA_MAX``.
    * ``count_dict`` – number of projections that feed into each group accumulator
                       (4 for full attention blocks, 2 for MLP blocks).  Used to
                       normalise the summed ``s_all`` values so that attention and MLP
                       groups are on the same absolute scale before blending.

    Parameters
    ----------
    granular_gqa : bool, optional
        Must match the value passed to ``init_sensitivity_dict`` so that tensor
        shapes are consistent.
    """
    var_dict = {}
    alpha_dict = {}
    count_dict = {}
    for name, module in model.named_modules():
        if _is_target_layer(module):
            weight_name = name.split('.')[-1]
            if is_gqa_model(model) and (weight_name in pruning_groups['self_attn']):
                if granular_gqa:
                    n_groups = model.config.num_attention_heads
                else:
                    n_groups = model.config.num_key_value_heads
            elif weight_name in pruning_groups['self_attn']:
                n_groups = model.config.num_attention_heads
            else:
                n_groups = module.out_features
            group_name = ".".join(name.split('.')[:-1])
            # k/v don't contribute to sensitivity in granular GQA — exclude from count
            if granular_gqa and is_gqa_model(model) and weight_name in ["k_proj", "v_proj"]:
                continue
            if group_name in var_dict:
                count_dict[group_name] += 1
                continue
            var_dict[group_name] = module.lora_A.weight.data.new_zeros(n_groups)
            alpha_dict[group_name] = _ADAPTIVE_ALPHA_MAX
            count_dict[group_name] = 1
    return var_dict, alpha_dict, count_dict


def update_sensitivity_dict(
        model,
        s_dict: dict[str, torch.Tensor],
        pruning_type: str,
        adaptive_ema: bool = False,
        var_dict: dict | None = None,
        alpha_dict: dict | None = None,
        count_dict: dict | None = None,
        granular_gqa: bool = False,
    ) -> dict[str, torch.Tensor]:
    """
    Compute per-group sensitivity for the current step and fold it into the
    running EMA stored in ``s_dict``.

    For every prunable module, ``compute_sensitivity`` produces a score vector
    whose length matches the group granularity defined by ``init_sensitivity_dict``
    (KV heads for GQA attention, Q heads for MHA attention, neurons for MLP).
    Scores for all projections that share a ``group_name`` key are summed into a
    fresh accumulator ``s_all``, then blended into the historical estimate.

    When ``adaptive_ema=False`` (default), uses a fixed blend::

        s_dict[g] = 0.9 * s_dict[g] + 0.1 * s_all[g]

    When ``adaptive_ema=True``, the blend coefficient is adapted per group based
    on how stable that group's importance signal has been.  Groups with low
    variance get a high history weight (more smoothing); groups with high variance
    get a lower history weight so recent gradient signals matter more.  A
    cross-group normalization also divides each group's accumulated score by the
    number of contributing projections (via ``count_dict``) before blending, so
    attention blocks (4 projections) and MLP blocks (2 projections) are on the
    same absolute scale.

    If any group produces a NaN or Inf score a ``RuntimeError`` is raised.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        PEFT-wrapped causal LM; same object passed to ``init_sensitivity_dict``.
    s_dict : dict[str, torch.Tensor]
        Running EMA sensitivity accumulator, initialised by
        ``init_sensitivity_dict`` and mutated in-place.
    pruning_type : str
        Sensitivity metric forwarded to ``compute_sensitivity``
        (``'lora'``, ``'magnitude'``, or ``'grad'``).
    adaptive_ema : bool, optional
        When ``True``, use per-group adaptive EMA instead of fixed 0.9/0.1.
        Requires ``var_dict``, ``alpha_dict``, and ``count_dict`` to be provided
        (returned by ``init_adaptive_ema_state``).  Default ``False``.
    var_dict : dict[str, torch.Tensor] | None
        Running element-wise variance accumulators; mutated in-place.
    alpha_dict : dict[str, float] | None
        Current per-group EMA coefficient; mutated in-place for logging.
    count_dict : dict[str, int] | None
        Number of projections per group used for cross-group normalization.

    Returns
    -------
    dict[str, torch.Tensor]
        The updated ``s_dict`` (same object, mutated in-place).
    """
    s_all = init_sensitivity_dict(model, granular_gqa=granular_gqa)
    for name, module in model.named_modules():
        if _is_target_layer(module):
            weight_name = name.split('.')[-1]
            is_attn = weight_name in pruning_groups['self_attn']
            fan_in = weight_name in pruning_groups['block']

            # In granular GQA mode, k/v scores are identical within a KV group and
            # don't help discriminate between Q heads — only q_proj and o_proj score.
            if granular_gqa and is_gqa_model(model) and weight_name in ["k_proj", "v_proj"]:
                continue

            s = compute_sensitivity(model, module, is_attn, pruning_type, fan_in, granular_gqa=granular_gqa)

            group_name = ".".join(name.split('.')[:-1])

            # add up all lora importances for all projections of this layer
            try:
                s_all[group_name] += s
            except:
                logger.exception(f"Error for group name: {group_name}, weight name: {weight_name}")
                raise
            
    for group_name, imp in s_all.items():
        if torch.isnan(imp.sum()) or torch.isinf(imp.sum()):
            raise RuntimeError(f"NaN/inf sensitivity detected for group '{group_name}'")

    if adaptive_ema and count_dict is not None:
        # Normalize by projection count so attention (4) and MLP (2) groups
        # contribute at the same per-projection scale before blending.
        for group_name in s_all:
            s_all[group_name] = s_all[group_name] / count_dict[group_name]

    for group_name, imp in s_dict.items():
        if adaptive_ema and var_dict is not None and alpha_dict is not None:
            current = s_all[group_name]
            # Update running element-wise variance of (current - EMA).
            delta_sq = (current - imp) ** 2
            var_dict[group_name] = (_ADAPTIVE_VAR_BETA * var_dict[group_name]
                                    + (1.0 - _ADAPTIVE_VAR_BETA) * delta_sq)
            # Derive per-group alpha: high variance → low alpha (track changes);
            # low variance → high alpha (smooth aggressively).
            # exp(-3 * sigma): sigma=0 → 1.0 (alpha=MAX); sigma≥0.5 → ≈0.22 (alpha→MIN).
            sigma = var_dict[group_name].mean().sqrt()
            alpha = (_ADAPTIVE_ALPHA_MIN
                     + (_ADAPTIVE_ALPHA_MAX - _ADAPTIVE_ALPHA_MIN)
                     * torch.exp(-3.0 * sigma)).clamp(_ADAPTIVE_ALPHA_MIN, _ADAPTIVE_ALPHA_MAX)
            alpha_dict[group_name] = alpha.item()
            s_dict[group_name] = alpha * imp + (1.0 - alpha) * current
        else:
            s_dict[group_name] = imp * 0.9 + s_all[group_name] * 0.1

    return s_dict


def compute_sensitivity(model, layer, is_attn, prune_metric='lora', transpose=False, norm=True, granular_gqa=False):
    """
    Compute a per-group importance score for a single LoRA-wrapped linear module.

    The score combines the current weight magnitude with a gradient signal to
    estimate how much each prunable group (head or neuron) contributes to the
    loss.  Three metrics are supported:

    ``'lora'``
        First-order Taylor approximation of the weight change due to the LoRA
        update.  The effective gradient of the full weight matrix ``W_eff = B@A``
        is approximated as ``grad_B @ A + B @ grad_A - grad_B @ grad_A``, then
        multiplied element-wise with the reconstructed effective weight
        ``B @ A * scaling + W``.  Taking the absolute value gives a proxy for
        the loss increase that would result from zeroing out each element.

    ``'magnitude'``
        Ignores gradients entirely; scores are the absolute values of the
        effective weight ``B @ A * scaling + W``.

    ``'grad'``
        Uses the raw gradient of the base weight ``W`` directly.

    For attention projections the score matrix is reshaped into groups before
    summing.  In GQA models every group spans ``out_features // num_kv_heads``
    rows (or columns after transpose), so that ``q_proj`` and ``o_proj`` scores
    are aggregated to ``num_kv_heads`` values — matching the accumulator size
    allocated by ``init_sensitivity_dict``.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The full model, used only to read ``config.num_attention_heads`` and
        ``config.num_key_value_heads`` for head grouping.
    layer : loraprune.lora.Linear
        The LoRA-wrapped linear module to score.
    is_attn : bool
        Whether this projection belongs to the self-attention block.  Controls
        whether the score is reshaped into per-head groups.
    prune_metric : {'lora', 'magnitude', 'grad'}, optional
        Sensitivity metric to use (default ``'lora'``).
    transpose : bool, optional
        Set to ``True`` for fan-in projections (``o_proj``, ``down_proj``) where
        the pruned dimension is the input (columns of ``W``).  Transposes the
        score matrix before grouping so rows always correspond to the pruned axis.
    norm : bool, optional
        If ``True`` (default), L2-normalise the final score vector so scores are
        comparable across layers.

    Returns
    -------
    torch.Tensor
        1-D tensor of shape ``(num_groups,)`` where ``num_groups`` is
        ``num_kv_heads`` for GQA attention, ``num_attention_heads`` for MHA
        attention, or ``out_features`` for MLP projections.
    """
    a = layer.lora_A.weight.data
    b = layer.lora_B.weight.data
    if prune_metric == 'lora':
        grad_a = layer.lora_A.weight.grad
        grad_b = layer.lora_B.weight.grad
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

    # Only q_proj and o_proj contribute in granular mode — k/v projections are
    # skipped by update_sensitivity_dict so this branch is only reached for q/o.
    if is_attn and is_gqa_model(model) and not granular_gqa:
        s = s.reshape(model.config.num_key_value_heads, -1)
    elif is_attn:
        s = s.reshape(model.config.num_attention_heads, -1)

    s = s.sum(1)  # collapse (num_heads, head_dim * in_features) -> (num_heads,)
    if norm:
        s = s / (torch.linalg.norm(s) + 1e-8)

    return s


def prune_fp16_module(model, module, mask, transpose):
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


def prune_one_layer(model, layer, granular_gqa=False):
    is_gqa = is_gqa_model(model)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # self_attn
    # torch stores weights in [out_features, in_features]
    # => for GQA k_proj it will be [512, 2048] -> [num_kv_heads*head_dim, hidden_size]
    # for GQA model, the mask must be expanded to remove all related Q-heads
    # mask for k_proj removes head_dim consecutive rows -> removes 1 head
    # mask for q_proj removes head_dim*num_q_per_kv consecutive rows -> removes all q heads for 1 kv head
    if is_gqa and granular_gqa:
        num_q_per_kv = model.config.num_attention_heads // model.config.num_key_value_heads
        # q_proj mask is ground truth (Q-head granularity set by local_prune);
        # derive KV mask: KV head i is kept iff any Q head in its group is kept.
        q_mask = layer.self_attn.q_proj.lora_mask.reshape(-1, head_dim)[:, 0]    # (num_q_heads,)
        kv_mask = q_mask.reshape(-1, num_q_per_kv).any(dim=1).float()           # (num_kv_heads, num_q_per_kv) → any → (num_kv_heads,)
        layer.self_attn.k_proj.lora_mask.data = kv_mask.repeat_interleave(head_dim)  # e.g. kv_mask=[0.,1.] → [0,0,0,0, 1,1,1,1]  shape (num_kv_heads*head_dim,)
        layer.self_attn.v_proj.lora_mask.data = kv_mask.repeat_interleave(head_dim)
    elif is_gqa:
        num_q_per_kv = model.config.num_attention_heads // model.config.num_key_value_heads
        # k_proj mask is the ground truth for which KV heads are pruned;
        # derive q_proj mask from it so they stay in sync even if target_ratio
        # caused local_prune to stop updating one projection early.
        kv_mask = layer.self_attn.k_proj.lora_mask.reshape(-1, head_dim)[:, 0]
        layer.self_attn.q_proj.lora_mask.data = kv_mask.repeat_interleave(num_q_per_kv * head_dim)
        layer.self_attn.v_proj.lora_mask.data = layer.self_attn.k_proj.lora_mask.data.clone()


    prune_fp16_module(model, layer.self_attn.q_proj, layer.self_attn.q_proj.lora_mask, False)
    prune_fp16_module(model, layer.self_attn.k_proj, layer.self_attn.k_proj.lora_mask, False)
    prune_fp16_module(model, layer.self_attn.v_proj, layer.self_attn.v_proj.lora_mask, False)
    
    # q_proj out_features = o_proj in_features
    # after removing some heads o_proj rows must be removed accordingly
    prune_fp16_module(model, layer.self_attn.o_proj, layer.self_attn.q_proj.lora_mask, True)
    layer.self_attn.num_heads = int(layer.self_attn.q_proj.lora_mask.sum()) // head_dim
    layer.self_attn.hidden_size = int(layer.self_attn.q_proj.lora_mask.sum())
    layer.self_attn.num_key_value_heads = (
        layer.self_attn.k_proj.out_features // head_dim
    )

    ## mlp
    prune_fp16_module(model, layer.mlp.gate_proj, layer.mlp.gate_proj.lora_mask, False)
    prune_fp16_module(model, layer.mlp.up_proj, layer.mlp.up_proj.lora_mask, False)
    # gate/up outputs → down inputs
    # after removing 
    prune_fp16_module(model, layer.mlp.down_proj, layer.mlp.gate_proj.lora_mask, True)

    ## reset mask
    del(layer.self_attn.q_proj.lora_mask)
    del(layer.self_attn.k_proj.lora_mask)
    del(layer.self_attn.v_proj.lora_mask)
    del(layer.mlp.gate_proj.lora_mask)
    del(layer.mlp.up_proj.lora_mask)
    del(layer.self_attn.o_proj.lora_mask)
    del(layer.mlp.down_proj.lora_mask)


def prune(model, granular_gqa=False):
    for layer_id, layer in enumerate(model.model.model.layers):
        logger.info("pruning layer {}".format(layer_id))
        prune_one_layer(model, layer, granular_gqa=granular_gqa)


def local_prune(model, s_dict, ratio, target_ratio, granular_gqa=False):
    original_param_num = 0
    pruned_param_num = 0
    _is_gqa = is_gqa_model(model)
    for name, module in model.named_modules():
        if _is_target_layer(module):
            original_param_num += np.prod(module.weight.shape)
            pruned_param_num += np.prod(module.weight.shape) * ratio
            weight_name = name.split('.')[-1]
            is_attn = weight_name in pruning_groups['self_attn']
            if weight_name in pruning_groups['block']:
                continue

            # In granular GQA mode, k_proj/v_proj masks are derived from q_proj
            # after the main loop — skip them here.
            if granular_gqa and _is_gqa and weight_name in ["k_proj", "v_proj"]:
                continue

            group_name = ".".join(name.split('.')[:-1])

            if not hasattr(module, 'lora_mask'):
                continue

            if (1 - module.lora_mask.mean()).item() >= target_ratio:
                continue

            total_num = module.lora_mask.numel()
            c_mask = module.lora_mask.data
            mask = torch.ones_like(c_mask)

            # determine head count for reshaping
            if granular_gqa and _is_gqa and weight_name in ["q_proj", "k_proj", "v_proj"]:
                # granular mode: always use Q-head granularity
                num_heads = model.config.num_attention_heads
            elif _is_gqa and weight_name in ["q_proj", "k_proj", "v_proj"]:
                num_heads = model.config.num_key_value_heads
            elif weight_name in ["q_proj", "k_proj", "v_proj"]:
                num_heads = model.config.num_attention_heads

            # for attention - reshape the mask to be of size [n_heads, head_dim] to prune full heads
            if is_attn:
                head_dim = module.out_features // num_heads
                mask = mask.reshape(-1, head_dim)[:, 0]
                c_mask = c_mask.reshape(-1, head_dim)[:, 0]
                total_num /= head_dim  # convert into number of heads instead of neurons

            need_prune_num = int(total_num * ratio)
            # set already pruned weights' importances to 0
            importance = s_dict[group_name] * c_mask
            # the rest slots are new weights with lowest imporance to be pruned
            can_prune = torch.argsort(importance)[:need_prune_num]
            mask[can_prune] = 0

            if is_attn:
                mask = (mask.new_ones(module.lora_mask.shape).reshape(-1, head_dim) * mask.unsqueeze(1)).reshape(-1)
            module.lora_mask.data = mask
        else:
            if hasattr(module, 'weight'):
                original_param_num += np.prod(module.weight.shape)

    # Pass 2 (granular GQA only): derive k_proj/v_proj masks from updated q_proj masks.
    # KV head i is kept iff at least one Q head in its group is still alive.
    if granular_gqa and _is_gqa:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        num_q_per_kv = model.config.num_attention_heads // model.config.num_key_value_heads
        for layer in model.model.model.layers:
            q_proj = layer.self_attn.q_proj
            if not hasattr(q_proj, 'lora_mask'):
                continue
            q_mask = q_proj.lora_mask.data.reshape(-1, head_dim)[:, 0]            # (num_q_heads,)
            kv_mask = q_mask.reshape(-1, num_q_per_kv).any(dim=1).float()       # (num_kv_heads, num_q_per_kv) → any → (num_kv_heads,)
            kv_mask_full = kv_mask.repeat_interleave(head_dim)
            layer.self_attn.k_proj.lora_mask.data = kv_mask_full
            # v_proj gets its own tensor instead of sharing memory with k_proj.lora_mask
            layer.self_attn.v_proj.lora_mask.data = kv_mask_full.clone()

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


def prune_from_checkpoint(model, granular_gqa=False):
    prune(model, granular_gqa=granular_gqa)


def print_trainable_parameters(model):
    total_params = 0
    trainable_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        total_params += p.numel()
    logger.info("total params:{}   trainable params:{}    ratio:{}".format(total_params * 1e-6, trainable_params * 1e-6, trainable_params / total_params))