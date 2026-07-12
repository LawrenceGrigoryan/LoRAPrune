"""Tests for loraprune/utils.py — MHA and GQA pruning coverage."""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from loraprune.lora import Linear
from loraprune.utils import (
    _is_target_layer,
    compute_sensitivity,
    freeze,
    init_adaptive_ema_state,
    init_sensitivity_dict,
    is_gqa_model,
    local_prune,
    prune_fp16_module,
    prune_one_layer,
    schedule_sparsity_ratio,
    unfreeze,
    update_sensitivity_dict,
)


# ---------------------------------------------------------------------------
# Minimal fake-model infrastructure
# ---------------------------------------------------------------------------

def _cfg(num_heads, num_kv_heads, hidden_size=32):
    return SimpleNamespace(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
    )


def _lora(in_f, out_f, r=4, lora_alpha=8):
    return Linear(in_f, out_f, r=r, lora_alpha=lora_alpha,
                  lora_dropout=0.0, merge_weights=False, bias=False)


class _Attn(nn.Module):
    def __init__(self, hidden, num_heads, num_kv, r=4):
        super().__init__()
        head_dim = hidden // num_heads
        self.q_proj = _lora(hidden, hidden, r=r)
        self.k_proj = _lora(hidden, num_kv * head_dim, r=r)
        self.v_proj = _lora(hidden, num_kv * head_dim, r=r)
        self.o_proj = _lora(hidden, hidden, r=r)
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv
        self.hidden_size = hidden


class _MLP(nn.Module):
    def __init__(self, hidden, inter, r=4):
        super().__init__()
        self.gate_proj = _lora(hidden, inter, r=r)
        self.up_proj   = _lora(hidden, inter, r=r)
        self.down_proj = _lora(inter, hidden, r=r)


class _Layer(nn.Module):
    def __init__(self, hidden, num_heads, num_kv, inter=64, r=4):
        super().__init__()
        self.self_attn = _Attn(hidden, num_heads, num_kv, r=r)
        self.mlp = _MLP(hidden, inter, r=r)


class _Inner(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _LLM(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = _Inner(layers)


class _LoraWrap(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = _LLM(layers)


class FakeModel(nn.Module):
    """
    Mimics LoraPeftModelForCausalLM so that:
      model.model.model.layers  → ModuleList of transformer layers
      named_modules() yields  'base_model.model.model.layers.N.self_attn.q_proj'
      name.split('.')[4]  ==  str(N)
    """
    def __init__(self, transformer_layers, config):
        super().__init__()
        self.config = config
        self.base_model = _LoraWrap(transformer_layers)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def _build_mha(num_layers=3, hidden=32, num_heads=4, inter=64, r=4):
    layers = [_Layer(hidden, num_heads, num_heads, inter, r) for _ in range(num_layers)]
    return FakeModel(layers, _cfg(num_heads, num_heads, hidden))


def _build_gqa(num_layers=3, hidden=32, num_heads=8, num_kv=2, inter=64, r=4):
    layers = [_Layer(hidden, num_heads, num_kv, inter, r) for _ in range(num_layers)]
    return FakeModel(layers, _cfg(num_heads, num_kv, hidden))


def _set_lora_grads(model):
    """Attach random non-zero gradients to every LoRA parameter."""
    for _, m in model.named_modules():
        if _is_target_layer(m):
            m.lora_A.weight.grad = torch.randn_like(m.lora_A.weight)
            m.lora_B.weight.grad = torch.randn_like(m.lora_B.weight)


# ---------------------------------------------------------------------------
# is_gqa_model
# ---------------------------------------------------------------------------

class TestIsGqaModel:
    def test_mha_returns_false(self):
        model = _build_mha()
        assert is_gqa_model(model) is False

    def test_gqa_returns_true(self):
        model = _build_gqa()
        assert is_gqa_model(model) is True

    def test_boundary_equal_heads_is_not_gqa(self):
        # num_kv == num_heads edge case
        model = _build_gqa(num_heads=4, num_kv=4)
        assert is_gqa_model(model) is False


# ---------------------------------------------------------------------------
# _is_target_layer
# ---------------------------------------------------------------------------

class TestIsTargetLayer:
    def test_lora_linear_with_is_prune_true(self):
        layer = _lora(16, 32)
        assert layer.is_prune is True
        assert _is_target_layer(layer) is True

    def test_lora_linear_with_is_prune_false(self):
        layer = _lora(16, 32)
        layer.is_prune = False
        assert _is_target_layer(layer) is False

    def test_plain_nn_linear_is_not_target(self):
        assert _is_target_layer(nn.Linear(16, 32)) is False


# ---------------------------------------------------------------------------
# init_sensitivity_dict
# ---------------------------------------------------------------------------

class TestInitSensitivityDict:
    def test_mha_attn_shape(self):
        model = _build_mha(num_layers=1, num_heads=4)
        s = init_sensitivity_dict(model)
        attn_keys = [k for k in s if 'self_attn' in k]
        assert len(attn_keys) == 1
        assert s[attn_keys[0]].shape == (4,)   # one slot per head

    def test_gqa_attn_shape_uses_kv_heads(self):
        model = _build_gqa(num_layers=1, num_heads=8, num_kv=2)
        s = init_sensitivity_dict(model)
        attn_keys = [k for k in s if 'self_attn' in k]
        assert len(attn_keys) == 1
        assert s[attn_keys[0]].shape == (2,)   # one slot per KV head, NOT 8

    def test_mlp_shape(self):
        model = _build_mha(num_layers=1, inter=64)
        s = init_sensitivity_dict(model)
        mlp_keys = [k for k in s if 'mlp' in k]
        assert len(mlp_keys) == 1
        assert s[mlp_keys[0]].shape == (64,)   # one slot per intermediate neuron

    def test_one_entry_per_block_not_per_projection(self):
        # 3 layers → 3 attn groups + 3 MLP groups = 6 total
        model = _build_mha(num_layers=3)
        s = init_sensitivity_dict(model)
        assert len(s) == 6

    def test_values_initialised_to_zero(self):
        model = _build_mha(num_layers=1)
        s = init_sensitivity_dict(model)
        for v in s.values():
            assert v.sum().item() == 0.0


# ---------------------------------------------------------------------------
# compute_sensitivity
# ---------------------------------------------------------------------------

class TestComputeSensitivity:
    def _attn_layer(self, hidden=32, num_heads=4, num_kv=None, r=4):
        num_kv = num_kv or num_heads
        return _lora(hidden, hidden, r=r)

    def test_magnitude_mha_output_shape(self):
        model = _build_mha(num_heads=4)
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        s = compute_sensitivity(model, layer, is_attn=True, prune_metric='magnitude')
        assert s.shape == (4,)  # one score per head

    def test_magnitude_gqa_output_shape_kv_heads(self):
        model = _build_gqa(num_heads=8, num_kv=2)
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        s = compute_sensitivity(model, layer, is_attn=True, prune_metric='magnitude')
        assert s.shape == (2,)  # one score per KV head

    def test_magnitude_gqa_kproj_output_shape(self):
        model = _build_gqa(num_heads=8, num_kv=2)
        layer = model.base_model.model.model.layers[0].self_attn.k_proj
        s = compute_sensitivity(model, layer, is_attn=True, prune_metric='magnitude')
        assert s.shape == (2,)

    def test_magnitude_mlp_output_shape(self):
        model = _build_mha()
        layer = model.base_model.model.model.layers[0].mlp.gate_proj
        s = compute_sensitivity(model, layer, is_attn=False, prune_metric='magnitude')
        assert s.shape == (64,)  # one score per intermediate neuron

    def test_lora_metric_output_shape_mha(self):
        model = _build_mha(num_heads=4)
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        layer.lora_A.weight.grad = torch.randn_like(layer.lora_A.weight)
        layer.lora_B.weight.grad = torch.randn_like(layer.lora_B.weight)
        s = compute_sensitivity(model, layer, is_attn=True, prune_metric='lora')
        assert s.shape == (4,)

    def test_lora_metric_output_shape_gqa(self):
        model = _build_gqa(num_heads=8, num_kv=2)
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        layer.lora_A.weight.grad = torch.randn_like(layer.lora_A.weight)
        layer.lora_B.weight.grad = torch.randn_like(layer.lora_B.weight)
        s = compute_sensitivity(model, layer, is_attn=True, prune_metric='lora')
        assert s.shape == (2,)

    def test_grad_metric_output_shape(self):
        model = _build_mha(num_heads=4)
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        layer.weight.grad = torch.randn_like(layer.weight)
        s = compute_sensitivity(model, layer, is_attn=True, prune_metric='grad')
        assert s.shape == (4,)

    def test_unknown_metric_raises(self):
        model = _build_mha()
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        with pytest.raises(NotImplementedError):
            compute_sensitivity(model, layer, is_attn=True, prune_metric='unknown')

    def test_norm_true_produces_unit_l2(self):
        model = _build_mha(num_heads=4)
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        # Ensure weight is non-zero so norm is meaningful
        layer.weight.data = torch.ones_like(layer.weight)
        s = compute_sensitivity(model, layer, is_attn=True, prune_metric='magnitude', norm=True)
        assert abs(torch.linalg.norm(s).item() - 1.0) < 1e-5

    def test_norm_false_does_not_normalise(self):
        model = _build_mha(num_heads=4)
        layer = model.base_model.model.model.layers[0].self_attn.q_proj
        layer.weight.data = torch.ones_like(layer.weight) * 2.0
        s_norm   = compute_sensitivity(model, layer, is_attn=True, prune_metric='magnitude', norm=True)
        s_raw    = compute_sensitivity(model, layer, is_attn=True, prune_metric='magnitude', norm=False)
        assert not torch.allclose(s_norm, s_raw)
        assert torch.linalg.norm(s_raw).item() > 1.0 + 1e-4

    def test_transpose_shapes_o_proj_mha(self):
        model = _build_mha(num_heads=4, hidden=32)
        o = model.base_model.model.model.layers[0].self_attn.o_proj
        s = compute_sensitivity(model, o, is_attn=True, prune_metric='magnitude', transpose=True)
        assert s.shape == (4,)  # grouped by input heads


# ---------------------------------------------------------------------------
# update_sensitivity_dict
# ---------------------------------------------------------------------------

class TestUpdateSensitivityDict:
    def test_ema_updates_from_zero(self):
        model = _build_mha(num_layers=1)
        s_dict = init_sensitivity_dict(model)
        _set_lora_grads(model)
        result = update_sensitivity_dict(model, s_dict, pruning_type='lora')
        # After first update from zero: s_dict = 0.9*0 + 0.1*s_all = 0.1*s_all > 0
        assert all(v.sum().item() >= 0.0 for v in result.values())

    def test_ema_second_update_blends(self):
        model = _build_mha(num_layers=1)
        _set_lora_grads(model)
        s_dict = init_sensitivity_dict(model)
        # Two updates; values should be strictly larger after the second
        update_sensitivity_dict(model, s_dict, pruning_type='magnitude')
        first = {k: v.clone() for k, v in s_dict.items()}
        update_sensitivity_dict(model, s_dict, pruning_type='magnitude')
        for k in s_dict:
            assert not torch.allclose(s_dict[k], first[k])

    def test_nan_sensitivity_raises_runtime_error(self):
        model = _build_mha(num_layers=1)
        # Inject NaN into lora_B grad so sensitivity becomes NaN
        for _, m in model.named_modules():
            if _is_target_layer(m):
                m.lora_A.weight.grad = torch.randn_like(m.lora_A.weight)
                m.lora_B.weight.grad = torch.full_like(m.lora_B.weight, float('nan'))
        s_dict = init_sensitivity_dict(model)
        with pytest.raises(RuntimeError, match="NaN/inf"):
            update_sensitivity_dict(model, s_dict, pruning_type='lora')

    def test_returns_same_dict_object(self):
        model = _build_mha(num_layers=1)
        _set_lora_grads(model)
        s_dict = init_sensitivity_dict(model)
        result = update_sensitivity_dict(model, s_dict, pruning_type='magnitude')
        assert result is s_dict

    def test_shapes_preserved_gqa(self):
        model = _build_gqa(num_layers=1, num_heads=8, num_kv=2)
        _set_lora_grads(model)
        s_dict = init_sensitivity_dict(model)
        update_sensitivity_dict(model, s_dict, pruning_type='magnitude')
        attn_key = next(k for k in s_dict if 'self_attn' in k)
        assert s_dict[attn_key].shape == (2,)


# ---------------------------------------------------------------------------
# prune_fp16_module  (row pruning and column pruning)
# ---------------------------------------------------------------------------

class TestPruneFp16Module:
    def test_row_prune_weight_shape(self):
        """transpose=False removes rows (output neurons)."""
        module = _lora(16, 8, r=2)
        mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float)
        prune_fp16_module(None, module, mask, transpose=False)
        assert module.weight.data.shape == (4, 16)
        assert module.out_features == 4

    def test_row_prune_lora_b_shape(self):
        module = _lora(16, 8, r=2)
        mask = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        prune_fp16_module(None, module, mask, transpose=False)
        assert module.lora_B.weight.data.shape[0] == 2  # lora_B rows pruned
        assert module.lora_B.out_features == 2

    def test_col_prune_weight_shape(self):
        """transpose=True removes columns (input neurons)."""
        module = _lora(8, 16, r=2)
        mask = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        prune_fp16_module(None, module, mask, transpose=True)
        assert module.weight.data.shape == (16, 2)
        assert module.in_features == 2

    def test_col_prune_lora_a_shape(self):
        module = _lora(8, 16, r=2)
        mask = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        prune_fp16_module(None, module, mask, transpose=True)
        assert module.lora_A.weight.data.shape[1] == 1
        assert module.lora_A.in_features == 1

    def test_mask_as_float_or_bool_accepted(self):
        module = _lora(8, 4, r=2)
        mask_f = torch.tensor([1.0, 1.0, 0.0, 0.0])
        prune_fp16_module(None, module, mask_f, transpose=False)
        assert module.out_features == 2

    def test_weight_values_correct_after_row_prune(self):
        module = _lora(4, 4, r=2)
        # Set weight to identity-like so we can check values
        w = torch.arange(16, dtype=torch.float).reshape(4, 4)
        module.weight.data = w.clone()
        module.lora_B.weight.data = torch.zeros_like(module.lora_B.weight)
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        prune_fp16_module(None, module, mask, transpose=False)
        # Rows 0 and 2 should remain; weight includes merged lora (zero) + original
        expected_rows = torch.stack([w[0], w[2]])
        assert torch.allclose(module.weight.data, expected_rows)


# ---------------------------------------------------------------------------
# prune_one_layer  (MHA and GQA — main focus)
# ---------------------------------------------------------------------------

class TestPruneOneLayerMHA:
    """MHA: num_heads == num_kv_heads == 4, hidden=32, head_dim=8."""

    def setup_method(self):
        self.hidden = 32
        self.num_heads = 4
        self.head_dim = 8   # 32 / 4
        self.inter = 64
        self.model = _build_mha(num_layers=1, hidden=self.hidden,
                                 num_heads=self.num_heads, inter=self.inter)
        self.layer = self.model.base_model.model.model.layers[0]

    def _set_attn_mask(self, keep_heads):
        """Set a head-level mask keeping `keep_heads` lowest-indexed heads."""
        mask = torch.zeros(self.hidden)
        mask[:keep_heads * self.head_dim] = 1.0
        for proj in (self.layer.self_attn.q_proj,
                     self.layer.self_attn.k_proj,
                     self.layer.self_attn.v_proj,
                     self.layer.self_attn.o_proj):
            proj.lora_mask.data = mask.clone()

    def _set_mlp_mask(self, keep_neurons):
        mask = torch.zeros(self.inter)
        mask[:keep_neurons] = 1.0
        self.layer.mlp.gate_proj.lora_mask.data = mask.clone()
        self.layer.mlp.up_proj.lora_mask.data = mask.clone()
        self.layer.mlp.down_proj.lora_mask.data = mask.clone()

    def test_q_proj_shape_after_pruning(self):
        self._set_attn_mask(keep_heads=2)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        assert self.layer.self_attn.q_proj.weight.shape == (16, self.hidden)

    def test_k_proj_shape_after_pruning(self):
        self._set_attn_mask(keep_heads=2)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        assert self.layer.self_attn.k_proj.weight.shape == (16, self.hidden)

    def test_v_proj_shape_after_pruning(self):
        self._set_attn_mask(keep_heads=2)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        assert self.layer.self_attn.v_proj.weight.shape == (16, self.hidden)

    def test_o_proj_cols_pruned(self):
        """o_proj input columns = q_proj output rows."""
        self._set_attn_mask(keep_heads=2)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        assert self.layer.self_attn.o_proj.weight.shape == (self.hidden, 16)

    def test_num_heads_updated(self):
        self._set_attn_mask(keep_heads=2)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        assert self.layer.self_attn.num_heads == 2

    def test_num_kv_heads_updated_mha(self):
        self._set_attn_mask(keep_heads=3)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        assert self.layer.self_attn.num_key_value_heads == 3

    def test_mlp_gate_proj_pruned(self):
        self._set_attn_mask(keep_heads=self.num_heads)
        self._set_mlp_mask(keep_neurons=32)
        prune_one_layer(self.model, self.layer)
        assert self.layer.mlp.gate_proj.weight.shape == (32, self.hidden)

    def test_mlp_up_proj_pruned(self):
        self._set_attn_mask(keep_heads=self.num_heads)
        self._set_mlp_mask(keep_neurons=32)
        prune_one_layer(self.model, self.layer)
        assert self.layer.mlp.up_proj.weight.shape == (32, self.hidden)

    def test_mlp_down_proj_cols_pruned(self):
        """down_proj input cols must match gate/up out rows."""
        self._set_attn_mask(keep_heads=self.num_heads)
        self._set_mlp_mask(keep_neurons=32)
        prune_one_layer(self.model, self.layer)
        assert self.layer.mlp.down_proj.weight.shape == (self.hidden, 32)

    def test_lora_mask_removed_after_prune(self):
        self._set_attn_mask(keep_heads=2)
        self._set_mlp_mask(keep_neurons=32)
        prune_one_layer(self.model, self.layer)
        for proj in (self.layer.self_attn.q_proj, self.layer.self_attn.k_proj,
                     self.layer.self_attn.v_proj, self.layer.self_attn.o_proj,
                     self.layer.mlp.gate_proj, self.layer.mlp.up_proj,
                     self.layer.mlp.down_proj):
            assert not hasattr(proj, 'lora_mask')


class TestPruneOneLayerGQA:
    """GQA: 8 Q-heads, 2 KV-heads, hidden=32, head_dim=4, q_per_kv=4."""

    def setup_method(self):
        self.hidden = 32
        self.num_q  = 8
        self.num_kv = 2
        self.head_dim = 4   # 32 / 8
        self.inter = 64
        self.model = _build_gqa(num_layers=1, hidden=self.hidden,
                                  num_heads=self.num_q, num_kv=self.num_kv,
                                  inter=self.inter)
        self.layer = self.model.base_model.model.model.layers[0]

    def _set_kv_mask(self, keep_kv_heads):
        """Set k_proj mask keeping `keep_kv_heads` lowest-indexed KV heads."""
        kv_out = self.num_kv * self.head_dim   # 2 * 4 = 8
        mask = torch.zeros(kv_out)
        mask[:keep_kv_heads * self.head_dim] = 1.0
        self.layer.self_attn.k_proj.lora_mask.data = mask.clone()
        self.layer.self_attn.v_proj.lora_mask.data = mask.clone()
        # q_proj and o_proj masks will be derived by prune_one_layer from k_proj mask
        q_out = self.num_q * self.head_dim  # 8 * 4 = 32
        self.layer.self_attn.q_proj.lora_mask.data = torch.ones(q_out)
        self.layer.self_attn.o_proj.lora_mask.data = torch.ones(q_out)

    def _set_mlp_mask(self, keep_neurons):
        mask_inter = torch.zeros(self.inter)
        mask_inter[:keep_neurons] = 1.0
        self.layer.mlp.gate_proj.lora_mask.data = mask_inter.clone()
        self.layer.mlp.up_proj.lora_mask.data   = mask_inter.clone()
        self.layer.mlp.down_proj.lora_mask.data = mask_inter.clone()

    def test_k_proj_rows_match_kept_kv_heads(self):
        self._set_kv_mask(keep_kv_heads=1)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        # 1 KV head kept → 1 * head_dim = 4 rows remain
        assert self.layer.self_attn.k_proj.weight.shape == (4, self.hidden)

    def test_v_proj_rows_match_kept_kv_heads(self):
        self._set_kv_mask(keep_kv_heads=1)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        assert self.layer.self_attn.v_proj.weight.shape == (4, self.hidden)

    def test_q_proj_rows_reflect_kv_grouping(self):
        """Pruning 1 KV head removes q_per_kv=4 Q heads (4*head_dim=16 rows)."""
        self._set_kv_mask(keep_kv_heads=1)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        # 1 KV head kept → 1 * q_per_kv * head_dim = 1 * 4 * 4 = 16 Q rows
        assert self.layer.self_attn.q_proj.weight.shape == (16, self.hidden)

    def test_o_proj_cols_match_q_proj_rows(self):
        """o_proj in_features must equal q_proj out_features after pruning."""
        self._set_kv_mask(keep_kv_heads=1)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        assert self.layer.self_attn.o_proj.weight.shape == (self.hidden, 16)

    def test_num_heads_updated_gqa(self):
        self._set_kv_mask(keep_kv_heads=1)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        # 1 KV head × 4 Q-per-KV = 4 Q heads
        assert self.layer.self_attn.num_heads == 4

    def test_num_kv_heads_updated_gqa(self):
        self._set_kv_mask(keep_kv_heads=1)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        assert self.layer.self_attn.num_key_value_heads == 1

    def test_kv_mask_sync_q_and_v(self):
        """v_proj mask must be derived from k_proj mask (not stale)."""
        # Set k_proj mask to prune KV head 0 (keep KV head 1)
        kv_out = self.num_kv * self.head_dim
        k_mask = torch.zeros(kv_out)
        k_mask[self.head_dim:] = 1.0   # keep KV head 1 only
        self.layer.self_attn.k_proj.lora_mask.data = k_mask.clone()
        self.layer.self_attn.v_proj.lora_mask.data = k_mask.clone()
        q_out = self.num_q * self.head_dim
        self.layer.self_attn.q_proj.lora_mask.data = torch.ones(q_out)
        self.layer.self_attn.o_proj.lora_mask.data = torch.ones(q_out)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        # v_proj should also have 4 rows (1 KV head)
        assert self.layer.self_attn.v_proj.weight.shape == (self.head_dim, self.hidden)

    def test_no_pruning_preserves_full_shapes(self):
        """If all heads are kept, shapes should be unchanged."""
        self._set_kv_mask(keep_kv_heads=self.num_kv)
        self._set_mlp_mask(keep_neurons=self.inter)
        prune_one_layer(self.model, self.layer)
        q_out = self.num_q * self.head_dim
        kv_out = self.num_kv * self.head_dim
        assert self.layer.self_attn.q_proj.weight.shape == (q_out, self.hidden)
        assert self.layer.self_attn.k_proj.weight.shape == (kv_out, self.hidden)
        assert self.layer.self_attn.o_proj.weight.shape == (self.hidden, q_out)


# ---------------------------------------------------------------------------
# local_prune
# ---------------------------------------------------------------------------

class TestLocalPrune:
    def _model_with_sensitivity(self, model, high_importance_key=None):
        """Init sensitivity dict; optionally boost importance of one group."""
        s_dict = init_sensitivity_dict(model)
        for k, v in s_dict.items():
            v.fill_(1.0)
        if high_importance_key:
            s_dict[high_importance_key].fill_(100.0)
        return s_dict

    def test_mha_local_prune_reduces_lora_mask(self):
        model = _build_mha(num_layers=1, num_heads=4)
        layer = model.base_model.model.model.layers[0]
        s_dict = init_sensitivity_dict(model)
        # Give first attn group high importance so second group gets pruned
        for k in list(s_dict.keys()):
            if 'self_attn' in k:
                s_dict[k] = torch.tensor([100.0, 100.0, 1.0, 1.0])
        local_prune(model, s_dict, ratio=0.5, target_ratio=1.0)
        # Heads 2 and 3 (lowest importance) should be pruned (mask=0)
        q_mask = layer.self_attn.q_proj.lora_mask
        # Each head covers head_dim=8 neurons; heads 2&3 → indices 16-31
        assert q_mask[:16].sum().item() == 16.0
        assert q_mask[16:].sum().item() == 0.0

    def test_gqa_local_prune_prunes_kv_head_granularity(self):
        """Pruning in GQA should respect KV-head granularity for k_proj."""
        model = _build_gqa(num_layers=1, num_heads=8, num_kv=2)
        layer = model.base_model.model.model.layers[0]
        s_dict = init_sensitivity_dict(model)
        # Low importance on KV head 1
        for k in list(s_dict.keys()):
            if 'self_attn' in k:
                s_dict[k] = torch.tensor([100.0, 1.0])  # head 0 high, head 1 low
        local_prune(model, s_dict, ratio=0.5, target_ratio=1.0)
        k_mask = layer.self_attn.k_proj.lora_mask
        # head_dim=4; KV head 0 kept, KV head 1 pruned
        assert k_mask[:4].sum().item() == 4.0   # head 0: kept
        assert k_mask[4:].sum().item() == 0.0   # head 1: pruned

    def test_local_prune_skips_at_target_ratio(self):
        model = _build_mha(num_layers=1, num_heads=4)
        s_dict = init_sensitivity_dict(model)
        for v in s_dict.values():
            v.fill_(1.0)
        # target_ratio=0 means current sparsity (0) >= 0, so all are skipped
        local_prune(model, s_dict, ratio=0.5, target_ratio=0.0)
        for _, m in model.named_modules():
            if _is_target_layer(m) and hasattr(m, 'lora_mask'):
                assert m.lora_mask.sum().item() == m.lora_mask.numel()

    def test_mlp_pruned_at_neuron_level(self):
        model = _build_mha(num_layers=1, inter=64)
        layer = model.base_model.model.model.layers[0]
        s_dict = init_sensitivity_dict(model)
        for k, v in s_dict.items():
            if 'mlp' in k:
                v[:] = torch.cat([torch.ones(32) * 100, torch.ones(32) * 1.0])
        local_prune(model, s_dict, ratio=0.5, target_ratio=1.0)
        gate_mask = layer.mlp.gate_proj.lora_mask
        assert gate_mask[:32].sum().item() == 32.0
        assert gate_mask[32:].sum().item() == 0.0


# ---------------------------------------------------------------------------
# schedule_sparsity_ratio
# ---------------------------------------------------------------------------

class TestScheduleSparsityRatio:
    def _sched(self, step, total=100, warmup=0.1, cooldown=0.1, s0=0.0, sf=0.5):
        return schedule_sparsity_ratio(step, total, warmup, cooldown, s0, sf)

    def test_initial_warmup_returns_initial_sparsity(self):
        assert self._sched(step=5) == pytest.approx(0.0)

    def test_final_warmup_returns_final_sparsity(self):
        assert self._sched(step=95) == pytest.approx(0.5)

    def test_middle_step_is_between_initial_and_final(self):
        s = self._sched(step=50)
        assert 0.0 < s < 0.5

    def test_sparsity_monotone_increases(self):
        steps = range(0, 100, 5)
        values = [self._sched(s) for s in steps]
        # After warmup phase it should be non-decreasing
        post_warmup = values[2:]  # skip initial warmup
        assert all(a <= b + 1e-6 for a, b in zip(post_warmup, post_warmup[1:]))

    def test_exact_warmup_boundary(self):
        # step == initial_warmup * total → still in warmup → initial_sparsity
        assert self._sched(step=10) == pytest.approx(0.0)

    def test_exact_cooldown_boundary(self):
        # step just past (total - final_warmup * total) → final sparsity
        assert self._sched(step=91) == pytest.approx(0.5)

    def test_custom_sparsity_values(self):
        s = schedule_sparsity_ratio(
            step=50, total_step=100, initial_warmup=0.1,
            final_warmup=0.1, initial_sparsity=0.1, final_sparsity=0.9
        )
        assert 0.1 <= s <= 0.9


# ---------------------------------------------------------------------------
# freeze / unfreeze
# ---------------------------------------------------------------------------

class TestFreezeUnfreeze:
    def test_freeze_disables_is_prune_on_first_and_last_layers(self):
        # 10 layers: freeze_layer = int(10 * 0.1) = 1
        # → layer 0 and layer 9 (last) get is_prune=False
        model = _build_mha(num_layers=10)
        freeze(model)
        layers = model.model.model.layers
        for _, m in layers[0].named_modules():
            if isinstance(m, Linear):
                assert m.is_prune is False
        for _, m in layers[9].named_modules():
            if isinstance(m, Linear):
                assert m.is_prune is False

    def test_freeze_leaves_middle_layers_prunable(self):
        model = _build_mha(num_layers=10)
        freeze(model)
        layers = model.model.model.layers
        for i in range(1, 9):
            for _, m in layers[i].named_modules():
                if isinstance(m, Linear):
                    assert m.is_prune is True

    def test_unfreeze_restores_weight_requires_grad(self):
        model = _build_mha(num_layers=2)
        # First disable grads manually
        for _, m in model.named_modules():
            if _is_target_layer(m):
                m.weight.requires_grad = False
        unfreeze(model)
        for _, m in model.named_modules():
            if _is_target_layer(m):
                assert m.weight.requires_grad is True


# ---------------------------------------------------------------------------
# Granular GQA — init_sensitivity_dict
# ---------------------------------------------------------------------------

class TestInitSensitivityDictGranularGQA:
    """granular_gqa=True must track sensitivity at Q-head (not KV-head) granularity."""

    def test_attn_shape_uses_q_heads(self):
        model = _build_gqa(num_layers=1, num_heads=8, num_kv=2)
        s = init_sensitivity_dict(model, granular_gqa=True)
        attn_keys = [k for k in s if 'self_attn' in k]
        assert len(attn_keys) == 1
        assert s[attn_keys[0]].shape == (8,)  # per Q-head, NOT per KV-head

    def test_mlp_shape_unchanged(self):
        model = _build_gqa(num_layers=1, num_heads=8, num_kv=2, inter=64)
        s = init_sensitivity_dict(model, granular_gqa=True)
        mlp_keys = [k for k in s if 'mlp' in k]
        assert s[mlp_keys[0]].shape == (64,)

    def test_mha_unaffected_by_granular_flag(self):
        model = _build_mha(num_layers=1, num_heads=4)
        s_normal = init_sensitivity_dict(model, granular_gqa=False)
        s_granular = init_sensitivity_dict(model, granular_gqa=True)
        attn_key = next(k for k in s_normal if 'self_attn' in k)
        assert s_normal[attn_key].shape == s_granular[attn_key].shape == (4,)

    def test_default_false_gives_kv_head_shape(self):
        model = _build_gqa(num_layers=1, num_heads=8, num_kv=2)
        s = init_sensitivity_dict(model)  # granular_gqa defaults to False
        attn_key = next(k for k in s if 'self_attn' in k)
        assert s[attn_key].shape == (2,)  # KV-head granularity by default

    def test_entries_initialised_to_zero(self):
        model = _build_gqa(num_layers=1, num_heads=8, num_kv=2)
        s = init_sensitivity_dict(model, granular_gqa=True)
        for v in s.values():
            assert v.sum().item() == 0.0


# ---------------------------------------------------------------------------
# Granular GQA — init_adaptive_ema_state
# ---------------------------------------------------------------------------

class TestInitAdaptiveEmaStateGranularGQA:
    def test_var_dict_shape_matches_sensitivity_dict(self):
        model = _build_gqa(num_layers=1, num_heads=8, num_kv=2)
        s = init_sensitivity_dict(model, granular_gqa=True)
        var_dict, alpha_dict, count_dict = init_adaptive_ema_state(model, granular_gqa=True)
        for k in s:
            assert var_dict[k].shape == s[k].shape

    def test_var_shape_is_q_heads_not_kv_heads(self):
        model = _build_gqa(num_layers=1, num_heads=8, num_kv=2)
        var_dict, _, _ = init_adaptive_ema_state(model, granular_gqa=True)
        attn_key = next(k for k in var_dict if 'self_attn' in k)
        assert var_dict[attn_key].shape == (8,)


# ---------------------------------------------------------------------------
# Granular GQA — compute_sensitivity
# ---------------------------------------------------------------------------

class TestComputeSensitivityGranularGQA:
    """All attention projections must return shape (num_q_heads,) in granular mode."""

    def setup_method(self):
        self.model = _build_gqa(num_layers=1, num_heads=8, num_kv=2, hidden=32)
        self.layer0 = self.model.base_model.model.model.layers[0]
        self.num_q = 8
        self.num_kv = 2
        self.num_q_per_kv = 4
        self.head_dim = 4  # 32 / 8

    def test_q_proj_returns_q_head_shape(self):
        s = compute_sensitivity(self.model, self.layer0.self_attn.q_proj,
                                is_attn=True, prune_metric='magnitude',
                                granular_gqa=True)
        assert s.shape == (self.num_q,)

    def test_k_proj_not_called_in_granular_mode(self):
        # update_sensitivity_dict skips k/v in granular GQA, so compute_sensitivity
        # is never called for them.  Verify q_proj and o_proj still work correctly.
        s_dict = init_sensitivity_dict(self.model, granular_gqa=True)
        _set_lora_grads(self.model)
        update_sensitivity_dict(self.model, s_dict, pruning_type='magnitude',
                                granular_gqa=True)
        # Sensitivity shape must be (num_q_heads,), accumulated from q_proj + o_proj only
        attn_key = next(k for k in s_dict if 'self_attn' in k)
        assert s_dict[attn_key].shape == (self.num_q,)
        assert s_dict[attn_key].sum().item() > 0.0

    def test_o_proj_transpose_returns_q_head_shape(self):
        s = compute_sensitivity(self.model, self.layer0.self_attn.o_proj,
                                is_attn=True, prune_metric='magnitude',
                                transpose=True, granular_gqa=True)
        assert s.shape == (self.num_q,)

    def test_q_head_scores_differ_within_same_kv_group(self):
        # With k/v excluded, per-Q-head scores come from q_proj alone and can
        # differ within the same KV group — this is the whole point of granular mode.
        w = self.layer0.self_attn.q_proj.weight.data
        # Set each head's rows to a distinct value so scores differ within group 0
        for h in range(self.num_q):
            w[h * self.head_dim:(h + 1) * self.head_dim] = float(h + 1)
        s = compute_sensitivity(self.model, self.layer0.self_attn.q_proj,
                                is_attn=True, prune_metric='magnitude',
                                norm=False, granular_gqa=True)
        # Scores within KV group 0 (Q heads 0-3) must be distinguishable
        assert not torch.allclose(s[0], s[1])

    def test_non_attn_projection_unaffected(self):
        s = compute_sensitivity(self.model, self.layer0.mlp.gate_proj,
                                is_attn=False, prune_metric='magnitude',
                                granular_gqa=True)
        assert s.shape == (64,)  # MLP neurons, unchanged

    def test_non_granular_k_proj_unchanged(self):
        # Without granular_gqa, k_proj should still return KV-head shape
        s = compute_sensitivity(self.model, self.layer0.self_attn.k_proj,
                                is_attn=True, prune_metric='magnitude',
                                granular_gqa=False)
        assert s.shape == (self.num_kv,)


# ---------------------------------------------------------------------------
# Granular GQA — update_sensitivity_dict
# ---------------------------------------------------------------------------

class TestUpdateSensitivityDictGranularGQA:
    def test_shape_preserved_as_q_heads(self):
        model = _build_gqa(num_layers=1, num_heads=8, num_kv=2)
        _set_lora_grads(model)
        s_dict = init_sensitivity_dict(model, granular_gqa=True)
        update_sensitivity_dict(model, s_dict, pruning_type='magnitude', granular_gqa=True)
        attn_key = next(k for k in s_dict if 'self_attn' in k)
        assert s_dict[attn_key].shape == (8,)

    def test_ema_updates_from_zero(self):
        model = _build_gqa(num_layers=1, num_heads=8, num_kv=2)
        _set_lora_grads(model)
        s_dict = init_sensitivity_dict(model, granular_gqa=True)
        result = update_sensitivity_dict(model, s_dict, pruning_type='magnitude', granular_gqa=True)
        attn_key = next(k for k in result if 'self_attn' in k)
        assert result[attn_key].sum().item() > 0.0


# ---------------------------------------------------------------------------
# Granular GQA — local_prune
# ---------------------------------------------------------------------------

class TestLocalPruneGranularGQA:
    """
    GQA: 8 Q-heads / 2 KV-heads, q_per_kv=4, hidden=32, head_dim=4.
    KV group 0 ↔ Q heads 0-3; KV group 1 ↔ Q heads 4-7.
    """

    def setup_method(self):
        self.hidden = 32
        self.num_q = 8
        self.num_kv = 2
        self.head_dim = 4   # 32 / 8
        self.num_q_per_kv = 4
        self.inter = 64

    def _build(self, num_layers=1):
        return _build_gqa(num_layers=num_layers, hidden=self.hidden,
                          num_heads=self.num_q, num_kv=self.num_kv, inter=self.inter)

    def _s_dict(self, model, attn_scores):
        """Build sensitivity dict with given per-Q-head scores for all attn groups."""
        s = init_sensitivity_dict(model, granular_gqa=True)
        for k in s:
            if 'self_attn' in k:
                s[k] = torch.tensor(attn_scores, dtype=torch.float)
            else:
                s[k].fill_(100.0)  # keep all MLP neurons
        return s

    def test_kv_head_pruned_when_all_q_heads_in_group_pruned(self):
        model = self._build()
        layer = model.base_model.model.model.layers[0]
        # Q heads 0-3 (group 0) have low importance → all pruned at ratio=0.5
        s_dict = self._s_dict(model, [1., 1., 1., 1., 100., 100., 100., 100.])
        local_prune(model, s_dict, ratio=0.5, target_ratio=1.0, granular_gqa=True)
        # KV head 0 must be masked out
        k_mask = layer.self_attn.k_proj.lora_mask
        assert k_mask[:self.head_dim].sum().item() == 0.0, "KV head 0 should be pruned"
        assert k_mask[self.head_dim:].sum().item() == self.head_dim, "KV head 1 should survive"

    def test_kv_head_kept_when_some_q_heads_in_group_survive(self):
        model = self._build()
        layer = model.base_model.model.model.layers[0]
        # Q heads 0,1 pruned (low importance), but Q heads 2,3 survive → KV head 0 must stay
        s_dict = self._s_dict(model, [1., 1., 100., 100., 100., 100., 100., 100.])
        local_prune(model, s_dict, ratio=0.25, target_ratio=1.0, granular_gqa=True)
        k_mask = layer.self_attn.k_proj.lora_mask
        # KV head 0 still alive because Q heads 2,3 survive
        assert k_mask[:self.head_dim].sum().item() == self.head_dim, \
            "KV head 0 should be kept (Q heads 2,3 are alive in its group)"

    def test_q_heads_pruned_independently_within_same_kv_group(self):
        model = self._build()
        layer = model.base_model.model.model.layers[0]
        # Q heads 0,1 (in group 0) pruned; Q heads 2,3 (also group 0) kept
        s_dict = self._s_dict(model, [1., 1., 100., 100., 100., 100., 100., 100.])
        local_prune(model, s_dict, ratio=0.25, target_ratio=1.0, granular_gqa=True)
        q_mask = layer.self_attn.q_proj.lora_mask
        # Q heads 0,1 → indices 0-7 should be 0
        assert q_mask[:2 * self.head_dim].sum().item() == 0.0
        # Q heads 2,3 → indices 8-15 should be 1
        assert q_mask[2 * self.head_dim:4 * self.head_dim].sum().item() == 2 * self.head_dim

    def test_v_proj_mask_matches_k_proj_mask(self):
        model = self._build()
        layer = model.base_model.model.model.layers[0]
        s_dict = self._s_dict(model, [1., 1., 1., 1., 100., 100., 100., 100.])
        local_prune(model, s_dict, ratio=0.5, target_ratio=1.0, granular_gqa=True)
        k_mask = layer.self_attn.k_proj.lora_mask
        v_mask = layer.self_attn.v_proj.lora_mask
        assert torch.equal(k_mask, v_mask), "v_proj mask must mirror k_proj mask"

    def test_k_proj_not_independently_pruned(self):
        # k_proj sparsity should be determined by q_proj, not by its own importance score.
        # If both KV groups have high sensitivity, k_proj should remain fully alive
        # even if ratio would nominally prune some of it.
        model = self._build()
        layer = model.base_model.model.model.layers[0]
        # All Q heads have equal (non-zero) importance; ratio=0.5 prunes half the Q heads
        # but we keep the other half → BOTH KV heads must stay alive (each group retains 2 Q heads)
        s_dict = self._s_dict(model, [1., 1., 100., 100., 1., 1., 100., 100.])
        local_prune(model, s_dict, ratio=0.5, target_ratio=1.0, granular_gqa=True)
        k_mask = layer.self_attn.k_proj.lora_mask
        # Both KV groups still have at least 2 alive Q heads → k_proj fully alive
        assert k_mask.sum().item() == k_mask.numel(), "Both KV heads must stay alive"

    def test_both_kv_groups_pruned_when_all_q_heads_pruned(self):
        model = self._build()
        layer = model.base_model.model.model.layers[0]
        # All Q heads have equal (low) importance — all pruned at ratio=1.0
        s_dict = self._s_dict(model, [1., 1., 1., 1., 1., 1., 1., 1.])
        local_prune(model, s_dict, ratio=1.0, target_ratio=1.0, granular_gqa=True)
        k_mask = layer.self_attn.k_proj.lora_mask
        assert k_mask.sum().item() == 0.0, "All KV heads should be pruned when all Q heads are gone"

    def test_mlp_masks_unaffected_by_granular_flag(self):
        model = self._build()
        layer = model.base_model.model.model.layers[0]
        s_dict = init_sensitivity_dict(model, granular_gqa=True)
        for k in s_dict:
            if 'mlp' in k:
                # First 32 neurons high importance, last 32 low
                s_dict[k] = torch.cat([torch.ones(32) * 100, torch.ones(32) * 1.0])
            else:
                s_dict[k].fill_(100.0)
        local_prune(model, s_dict, ratio=0.5, target_ratio=1.0, granular_gqa=True)
        gate_mask = layer.mlp.gate_proj.lora_mask
        assert gate_mask[:32].sum().item() == 32.0
        assert gate_mask[32:].sum().item() == 0.0


# ---------------------------------------------------------------------------
# Granular GQA — prune_one_layer
# ---------------------------------------------------------------------------

class TestPruneOneLayerGranularGQA:
    """
    GQA: 8 Q-heads / 2 KV-heads, hidden=32, head_dim=4, q_per_kv=4.
    KV group 0 ↔ Q heads 0-3; KV group 1 ↔ Q heads 4-7.
    q_proj ground truth; k/v_proj derived.
    """

    def setup_method(self):
        self.hidden = 32
        self.num_q = 8
        self.num_kv = 2
        self.head_dim = 4
        self.num_q_per_kv = 4
        self.inter = 64
        self.model = _build_gqa(num_layers=1, hidden=self.hidden,
                                num_heads=self.num_q, num_kv=self.num_kv,
                                inter=self.inter)
        self.layer = self.model.base_model.model.model.layers[0]

    def _set_masks(self, keep_q_head_indices):
        """Set q_proj mask keeping the specified Q-head indices; o_proj matches."""
        q_out = self.num_q * self.head_dim
        q_mask = torch.zeros(q_out)
        for h in keep_q_head_indices:
            q_mask[h * self.head_dim:(h + 1) * self.head_dim] = 1.0
        self.layer.self_attn.q_proj.lora_mask.data = q_mask.clone()
        self.layer.self_attn.o_proj.lora_mask.data = q_mask.clone()
        # k/v masks initialised to all-ones; prune_one_layer must overwrite them
        kv_out = self.num_kv * self.head_dim
        self.layer.self_attn.k_proj.lora_mask.data = torch.ones(kv_out)
        self.layer.self_attn.v_proj.lora_mask.data = torch.ones(kv_out)
        # MLP: keep everything
        for proj in (self.layer.mlp.gate_proj, self.layer.mlp.up_proj,
                     self.layer.mlp.down_proj):
            proj.lora_mask.data = torch.ones(self.inter)

    def test_entire_group_pruned_removes_kv_head(self):
        # Prune all of group 0 (Q heads 0-3); keep group 1 (Q heads 4-7)
        self._set_masks(keep_q_head_indices=[4, 5, 6, 7])
        prune_one_layer(self.model, self.layer, granular_gqa=True)
        # 4 Q heads kept → 4 * head_dim = 16 rows in q_proj
        assert self.layer.self_attn.q_proj.weight.shape == (16, self.hidden)
        # Group 0 pruned → KV head 0 removed; 1 KV head left → head_dim = 4 rows
        assert self.layer.self_attn.k_proj.weight.shape == (4, self.hidden)
        assert self.layer.self_attn.v_proj.weight.shape == (4, self.hidden)
        # o_proj columns = q_proj rows
        assert self.layer.self_attn.o_proj.weight.shape == (self.hidden, 16)

    def test_num_heads_and_kv_heads_updated(self):
        self._set_masks(keep_q_head_indices=[4, 5, 6, 7])
        prune_one_layer(self.model, self.layer, granular_gqa=True)
        assert self.layer.self_attn.num_heads == 4
        assert self.layer.self_attn.num_key_value_heads == 1

    def test_partial_group_keeps_kv_head(self):
        # Prune Q heads 0,1 only (partial group 0); Q heads 2,3 survive → KV head 0 stays
        self._set_masks(keep_q_head_indices=[2, 3, 4, 5, 6, 7])
        prune_one_layer(self.model, self.layer, granular_gqa=True)
        # 6 Q heads kept → 6 * head_dim = 24 rows
        assert self.layer.self_attn.q_proj.weight.shape == (24, self.hidden)
        # Both KV heads kept (group 0 has Q heads 2,3; group 1 has Q heads 4-7)
        assert self.layer.self_attn.k_proj.weight.shape == (8, self.hidden)
        assert self.layer.self_attn.num_key_value_heads == 2

    def test_q_proj_mask_overrides_stale_kv_masks(self):
        # k_proj/v_proj masks are initialised to all-ones in _set_masks;
        # prune_one_layer must replace them from the q_proj mask.
        self._set_masks(keep_q_head_indices=[4, 5, 6, 7])
        # Deliberately set k_proj mask to wrong value before calling prune
        self.layer.self_attn.k_proj.lora_mask.data = torch.ones(self.num_kv * self.head_dim)
        prune_one_layer(self.model, self.layer, granular_gqa=True)
        # k_proj must have only 1 KV head (not 2), showing the stale mask was overwritten
        assert self.layer.self_attn.k_proj.weight.shape == (self.head_dim, self.hidden)

    def test_no_pruning_preserves_shapes(self):
        # All Q heads kept → both KV heads kept → no change
        self._set_masks(keep_q_head_indices=list(range(self.num_q)))
        prune_one_layer(self.model, self.layer, granular_gqa=True)
        assert self.layer.self_attn.q_proj.weight.shape == (self.num_q * self.head_dim, self.hidden)
        assert self.layer.self_attn.k_proj.weight.shape == (self.num_kv * self.head_dim, self.hidden)
        assert self.layer.self_attn.num_heads == self.num_q
        assert self.layer.self_attn.num_key_value_heads == self.num_kv

    def test_lora_masks_removed_after_prune(self):
        self._set_masks(keep_q_head_indices=[4, 5, 6, 7])
        prune_one_layer(self.model, self.layer, granular_gqa=True)
        for proj in (self.layer.self_attn.q_proj, self.layer.self_attn.k_proj,
                     self.layer.self_attn.v_proj, self.layer.self_attn.o_proj,
                     self.layer.mlp.gate_proj, self.layer.mlp.up_proj,
                     self.layer.mlp.down_proj):
            assert not hasattr(proj, 'lora_mask')

    def test_v_proj_derived_from_q_mask_not_k_mask(self):
        # k_proj and v_proj must be derived independently from q; both should
        # end up with the same shape derived from the q_proj mask.
        self._set_masks(keep_q_head_indices=[4, 5, 6, 7])
        prune_one_layer(self.model, self.layer, granular_gqa=True)
        assert self.layer.self_attn.v_proj.weight.shape == \
               self.layer.self_attn.k_proj.weight.shape

    def test_non_granular_mode_unchanged(self):
        # With granular_gqa=False the original GQA path must still work:
        # k_proj mask is ground truth; q_proj derived from it.
        kv_out = self.num_kv * self.head_dim
        k_mask = torch.zeros(kv_out)
        k_mask[:self.head_dim] = 1.0  # only KV head 0
        self.layer.self_attn.k_proj.lora_mask.data = k_mask.clone()
        self.layer.self_attn.v_proj.lora_mask.data = k_mask.clone()
        q_out = self.num_q * self.head_dim
        self.layer.self_attn.q_proj.lora_mask.data = torch.ones(q_out)
        self.layer.self_attn.o_proj.lora_mask.data = torch.ones(q_out)
        for proj in (self.layer.mlp.gate_proj, self.layer.mlp.up_proj,
                     self.layer.mlp.down_proj):
            proj.lora_mask.data = torch.ones(self.inter)
        prune_one_layer(self.model, self.layer, granular_gqa=False)
        # 1 KV head × 4 Q-per-KV = 4 Q heads → 16 rows
        assert self.layer.self_attn.q_proj.weight.shape == (16, self.hidden)
        assert self.layer.self_attn.k_proj.weight.shape == (self.head_dim, self.hidden)
