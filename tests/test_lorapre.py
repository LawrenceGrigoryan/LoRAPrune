"""Tests for loraprune/optimizer.py — the LoRA-Pre optimizer (arXiv:2602.24283)."""

import math
from unittest import mock

import pytest
import torch

from loraprune.optimizer import LoRAPre


def _quadratic_problem(rows=32, cols=24, seed=0):
    """Small least-squares problem: fit W so that X @ W.T matches X @ W_true.T."""
    g = torch.Generator().manual_seed(seed)
    w_true = torch.randn(rows, cols, generator=g)
    x = torch.randn(128, cols, generator=g)
    target = x @ w_true.T
    w = torch.nn.Parameter(torch.randn(rows, cols, generator=g) * 0.1)
    return w, x, target


def _loss(w, x, target):
    return (x @ w.T - target).pow(2).mean()


# ---------------------------------------------------------------- optimization


def test_converges_on_least_squares():
    w, x, target = _quadratic_problem()
    opt = LoRAPre([w], lr=0.05, rank=4)

    initial = _loss(w, x, target).item()
    for _ in range(300):
        opt.zero_grad()
        _loss(w, x, target).backward()
        opt.step()
    final = _loss(w, x, target).item()

    assert final < initial * 0.05, f"expected strong progress, got {initial} -> {final}"


def test_loss_decreases_steadily():
    w, x, target = _quadratic_problem(seed=1)
    opt = LoRAPre([w], lr=0.02, rank=4)

    losses = []
    for _ in range(120):
        opt.zero_grad()
        loss = _loss(w, x, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Not strictly monotone step-to-step, but every 20-step window must improve.
    windows = [sum(losses[i : i + 20]) / 20 for i in range(0, 120, 20)]
    assert all(b < a for a, b in zip(windows, windows[1:])), windows


def test_matches_shape_and_dtype_of_param():
    w = torch.nn.Parameter(torch.randn(16, 12))
    opt = LoRAPre([w], lr=1e-3, rank=4)
    w.grad = torch.randn_like(w)
    opt.step()

    assert w.shape == (16, 12)
    assert w.dtype == torch.float32
    assert torch.isfinite(w).all()


# ------------------------------------------------------------- gamma coupling


def test_gammas_derived_from_betas():
    """Appendix B.1: 1 - gamma1 = sqrt(beta1) and 1 - gamma2 = beta2 ** 0.25."""
    betas = (0.9, 0.999)
    gamma1, gamma2 = LoRAPre._resolve_gammas(betas, (None, None))

    # The product of two factors, each decaying by (1 - gamma1), decays by beta1.
    assert (1.0 - gamma1) ** 2 == pytest.approx(betas[0])
    # The second moment is additionally squared, so the exponent is 4.
    assert (1.0 - gamma2) ** 4 == pytest.approx(betas[1])


def test_explicit_gammas_are_respected():
    gamma1, gamma2 = LoRAPre._resolve_gammas((0.9, 0.999), (0.3, 0.4))
    assert gamma1 == 0.3
    assert gamma2 == 0.4

    # A single override still derives the other one.
    gamma1, gamma2 = LoRAPre._resolve_gammas((0.9, 0.999), (0.3, None))
    assert gamma1 == 0.3
    assert gamma2 == pytest.approx(1.0 - 0.999**0.25)


def test_gamma_defaults_land_in_unit_interval():
    for beta1, beta2 in [(0.0, 0.0), (0.5, 0.9), (0.9, 0.999), (0.99, 0.9999)]:
        gamma1, gamma2 = LoRAPre._resolve_gammas((beta1, beta2), (None, None))
        assert 0.0 < gamma1 <= 1.0
        assert 0.0 < gamma2 <= 1.0


# -------------------------------------------------------------- state / memory


def test_state_is_low_rank_factors():
    rows, cols, rank = 64, 48, 8
    w = torch.nn.Parameter(torch.randn(rows, cols))
    opt = LoRAPre([w], lr=1e-3, rank=rank)
    w.grad = torch.randn_like(w)
    opt.step()

    state = opt.state[w]
    assert state["low_rank"] is True
    assert state["m_B"].shape == (rows, rank)
    assert state["m_A"].shape == (rank, cols)
    assert state["v_B"].shape == (rows, rank)
    assert state["v_A"].shape == (rank, cols)
    assert "exp_avg" not in state


def test_state_is_smaller_than_full_adam():
    rows, cols, rank = 128, 128, 8
    w = torch.nn.Parameter(torch.randn(rows, cols))
    opt = LoRAPre([w], lr=1e-3, rank=rank)
    w.grad = torch.randn_like(w)
    opt.step()

    state = opt.state[w]
    lorapre_elems = sum(state[k].numel() for k in ("m_B", "m_A", "v_B", "v_A"))
    adam_elems = 2 * rows * cols

    assert lorapre_elems == 2 * (rows + cols) * rank
    assert lorapre_elems < adam_elems
    # 2*(128+128)*8 = 4096 vs 2*128*128 = 32768, i.e. an 8x reduction.
    assert lorapre_elems / adam_elems == pytest.approx(0.125)


def test_state_is_float32_for_half_precision_params():
    """Momenta and the r x r solves stay in fp32 even when the param is fp16."""
    w = torch.nn.Parameter(torch.randn(32, 24, dtype=torch.float16))
    opt = LoRAPre([w], lr=1e-3, rank=4)
    w.grad = torch.randn_like(w)
    opt.step()

    state = opt.state[w]
    assert state["m_B"].dtype == torch.float32
    assert state["v_A"].dtype == torch.float32
    assert w.dtype == torch.float16  # param dtype is preserved
    assert torch.isfinite(w).all()


def test_a_factors_initialized_nonzero_b_factors_zero():
    w = torch.nn.Parameter(torch.randn(32, 24))
    opt = LoRAPre([w], lr=0.0, rank=4, init_std=0.02)
    w.grad = torch.zeros_like(w)

    # Peek at initialization before any factor update runs.
    state = opt.state[w]
    opt._init_state(state, w, opt.param_groups[0])
    assert torch.count_nonzero(state["m_B"]) == 0
    assert torch.count_nonzero(state["v_B"]) == 0
    assert torch.count_nonzero(state["m_A"]) > 0
    assert state["m_A"].std().item() == pytest.approx(0.02, rel=0.4)


# --------------------------------------------------------- numerical stability


def test_first_step_is_finite_despite_singular_gram():
    """m_B starts at zero, so (m_B^T m_B) is exactly singular on step 1."""
    w = torch.nn.Parameter(torch.randn(32, 24))
    opt = LoRAPre([w], lr=1e-2, rank=8)
    w.grad = torch.randn_like(w)
    opt.step()

    state = opt.state[w]
    assert torch.isfinite(w).all()
    for key in ("m_B", "m_A", "v_B", "v_A"):
        assert torch.isfinite(state[key]).all(), key


def test_solve_falls_back_to_cpu_when_backend_lacks_linalg_solve():
    """MPS has no linalg.solve; the r x r system should move to CPU, not to SVD."""
    torch.manual_seed(0)
    gram = torch.randn(4, 8) @ torch.randn(8, 4)
    gram = gram @ gram.T  # symmetric PSD
    rhs = torch.randn(4, 3)
    expected = LoRAPre._solve_spd(gram, rhs, 1e-6)

    real_solve = torch.linalg.solve
    calls = []

    def fake_solve(a, b):
        # Stand in for a backend that does not implement the op: fail once, the
        # way MPS does, then let the CPU retry through.
        calls.append(a.device.type)
        if len(calls) == 1:
            raise NotImplementedError("aten::_linalg_solve_ex.result not implemented")
        return real_solve(a, b)

    def no_pinv(*args, **kwargs):
        raise AssertionError("fell back to SVD-based pinv instead of retrying on CPU")

    with mock.patch.object(torch.linalg, "solve", fake_solve), \
         mock.patch.object(torch.linalg, "pinv", no_pinv):
        got = LoRAPre._solve_spd(gram, rhs, 1e-6)

    assert len(calls) == 2, "expected exactly one retry after NotImplementedError"
    assert torch.allclose(got, expected, atol=1e-6)


def test_zero_damping_is_rejected():
    w = torch.nn.Parameter(torch.randn(8, 8))
    with pytest.raises(ValueError, match="damping"):
        LoRAPre([w], lr=1e-3, damping=0.0)


def test_survives_extreme_gradients():
    w = torch.nn.Parameter(torch.randn(32, 24))
    opt = LoRAPre([w], lr=1e-4, rank=4)

    for scale in (1e-8, 1.0, 1e6, 1e-8):
        w.grad = torch.randn_like(w) * scale
        opt.step()
        assert torch.isfinite(w).all(), f"diverged at gradient scale {scale}"


def test_zero_gradient_leaves_param_unchanged_without_weight_decay():
    w = torch.nn.Parameter(torch.randn(32, 24))
    before = w.detach().clone()
    opt = LoRAPre([w], lr=1e-2, rank=4, weight_decay=0.0)

    w.grad = torch.zeros_like(w)
    opt.step()

    assert torch.allclose(w, before, atol=1e-7)


# ------------------------------------------------------------ AdamW fallback


def test_fallback_for_1d_param():
    b = torch.nn.Parameter(torch.randn(32))
    opt = LoRAPre([b], lr=1e-3, rank=8)
    b.grad = torch.randn_like(b)
    opt.step()

    state = opt.state[b]
    assert state["low_rank"] is False
    assert state["exp_avg"].shape == (32,)
    assert "m_B" not in state


def test_fallback_when_rank_exceeds_smaller_dimension():
    # min(p, q) == 8 == rank, so factors would not be smaller than the matrix.
    w = torch.nn.Parameter(torch.randn(64, 8))
    opt = LoRAPre([w], lr=1e-3, rank=8)
    w.grad = torch.randn_like(w)
    opt.step()

    assert opt.state[w]["low_rank"] is False

    # One below the boundary, the low-rank path engages.
    w2 = torch.nn.Parameter(torch.randn(64, 8))
    opt2 = LoRAPre([w2], lr=1e-3, rank=7)
    w2.grad = torch.randn_like(w2)
    opt2.step()
    assert opt2.state[w2]["low_rank"] is True


@pytest.mark.parametrize(
    "shape, rank, expected",
    [
        ((64, 32), 8, True),    # min(shape)=32 > 8
        ((64, 8), 8, False),    # min(shape)=8 == rank, factors are not smaller
        ((64, 8), 7, True),     # one below the boundary
        ((8, 4096), 8, False),  # a LoRA adapter with lora_r == lorapre_rank
        ((8, 4096), 4, True),   # the same adapter with lorapre_rank < lora_r
        ((32,), 8, False),      # 1-D
        ((4, 8, 16), 2, False), # 3-D
    ],
)
def test_uses_low_rank_predicate(shape, rank, expected):
    """The trainer reports coverage with this before any step allocates state."""
    assert LoRAPre.uses_low_rank(torch.zeros(*shape), rank) is expected


@pytest.mark.parametrize("shape, rank", [((64, 32), 8), ((64, 8), 8), ((32,), 8)])
def test_uses_low_rank_agrees_with_actual_path(shape, rank):
    """The public predicate must match the path the optimizer really takes."""
    p = torch.nn.Parameter(torch.randn(*shape))
    opt = LoRAPre([p], lr=1e-3, rank=rank)
    p.grad = torch.randn_like(p)
    opt.step()

    assert opt.state[p]["low_rank"] is LoRAPre.uses_low_rank(p, rank)


@pytest.mark.parametrize("shape", [(32,), (64, 4)])
def test_fallback_matches_torch_adamw(shape):
    """The fallback path must be a faithful AdamW, not an approximation."""
    torch.manual_seed(0)
    init = torch.randn(*shape)
    grads = [torch.randn(*shape) for _ in range(10)]

    ours = torch.nn.Parameter(init.clone())
    ref = torch.nn.Parameter(init.clone())
    opt_ours = LoRAPre([ours], lr=1e-2, rank=8, weight_decay=0.01, betas=(0.9, 0.999))
    opt_ref = torch.optim.AdamW([ref], lr=1e-2, weight_decay=0.01, betas=(0.9, 0.999))

    for g in grads:
        ours.grad = g.clone()
        ref.grad = g.clone()
        opt_ours.step()
        opt_ref.step()

    assert opt_ours.state[ours]["low_rank"] is False
    assert torch.allclose(ours, ref, atol=1e-6), (ours - ref).abs().max()


def test_mixed_model_parameters_all_step():
    """model.parameters() can be passed wholesale: 2-D weights and 1-D biases."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(64),
        torch.nn.Linear(64, 8),
    )
    opt = LoRAPre(model.parameters(), lr=1e-3, rank=8)

    before = [p.detach().clone() for p in model.parameters()]
    for _ in range(5):
        opt.zero_grad()
        model(torch.randn(16, 64)).pow(2).mean().backward()
        opt.step()

    for p, b in zip(model.parameters(), before):
        assert torch.isfinite(p).all()
        assert not torch.allclose(p, b), "parameter did not move"

    paths = [opt.state[p]["low_rank"] for p in model.parameters()]
    assert any(paths) and not all(paths), "expected both code paths to be exercised"


# ---------------------------------------------------- weight decay & interface


def test_weight_decay_shrinks_param_when_gradient_is_zero():
    w = torch.nn.Parameter(torch.ones(32, 24))
    opt = LoRAPre([w], lr=0.1, rank=4, weight_decay=0.1)

    w.grad = torch.zeros_like(w)
    opt.step()

    # Decoupled decay: theta <- theta - lr * wd * theta = 1 - 0.1*0.1 = 0.99
    assert torch.allclose(w, torch.full_like(w, 0.99), atol=1e-6)


def test_closure_return_value():
    w = torch.nn.Parameter(torch.randn(16, 12))
    opt = LoRAPre([w], lr=1e-3, rank=4)

    def closure():
        opt.zero_grad()
        loss = w.pow(2).sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert math.isfinite(loss.item())


def test_param_without_grad_is_skipped():
    used = torch.nn.Parameter(torch.randn(16, 12))
    unused = torch.nn.Parameter(torch.randn(16, 12))
    before = unused.detach().clone()

    opt = LoRAPre([used, unused], lr=1e-2, rank=4)
    used.grad = torch.randn_like(used)
    opt.step()

    assert torch.equal(unused, before)
    assert len(opt.state[unused]) == 0


def test_sparse_gradient_rejected():
    w = torch.nn.Parameter(torch.randn(16, 12))
    opt = LoRAPre([w], lr=1e-3, rank=4)
    idx = torch.tensor([[0, 1], [0, 1]])
    w.grad = torch.sparse_coo_tensor(idx, torch.ones(2), (16, 12))

    with pytest.raises(RuntimeError, match="sparse"):
        opt.step()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -1.0},
        {"betas": (1.0, 0.999)},
        {"betas": (0.9, -0.1)},
        {"eps": -1e-8},
        {"weight_decay": -0.1},
        {"rank": 0},
        {"init_std": 0.0},
        {"gammas": (1.5, None)},
        {"gammas": (None, 0.0)},
    ],
)
def test_invalid_hyperparameters_rejected(kwargs):
    w = torch.nn.Parameter(torch.randn(8, 8))
    with pytest.raises(ValueError):
        LoRAPre([w], **kwargs)


def test_param_groups_carry_independent_hyperparameters():
    a = torch.nn.Parameter(torch.randn(32, 24))
    b = torch.nn.Parameter(torch.randn(32, 24))
    opt = LoRAPre(
        [{"params": [a], "lr": 0.0, "rank": 4}, {"params": [b], "lr": 0.1, "rank": 4}],
        lr=1e-3,
    )
    before_a = a.detach().clone()

    before_b = b.detach().clone()

    a.grad = torch.randn_like(a)
    b.grad = torch.randn_like(b)
    opt.step()

    assert torch.allclose(a, before_a), "lr=0 group must not move"
    assert not torch.allclose(b, before_b), "lr=0.1 group must move"


def test_state_dict_roundtrip():
    w = torch.nn.Parameter(torch.randn(32, 24))
    opt = LoRAPre([w], lr=1e-2, rank=4)
    w.grad = torch.randn_like(w)
    opt.step()
    opt.step()

    saved = opt.state_dict()

    w2 = torch.nn.Parameter(w.detach().clone())
    opt2 = LoRAPre([w2], lr=1e-2, rank=4)
    opt2.load_state_dict(saved)

    grad = torch.randn_like(w)
    w.grad, w2.grad = grad.clone(), grad.clone()
    opt.step()
    opt2.step()

    assert torch.allclose(w, w2, atol=1e-7)


# ------------------------------------------------------- algorithm invariants


def test_factor_step_uses_old_factors_for_both_updates():
    """Eq. 11-12 are a simultaneous (Jacobi) update: both read B_{t-1}, A_{t-1}."""
    torch.manual_seed(0)
    rows, cols, rank, gamma = 16, 12, 4, 0.3
    target = torch.randn(rows, cols)
    b = torch.randn(rows, rank)
    a = torch.randn(rank, cols)

    new_b, new_a = LoRAPre._factor_step(target, b, a, gamma, damping=1e-6)

    # Recompute the A-step by hand from the OLD b. If _factor_step had written
    # b in place first, this would not match.
    gram_b = b.T @ b + 1e-6 * torch.eye(rank)
    expected_a = (1 - gamma) * a + gamma * torch.linalg.solve(gram_b, b.T @ target)
    assert torch.allclose(new_a, expected_a, atol=1e-5)

    # And the B-step from the old a.
    gram_a = a @ a.T + 1e-6 * torch.eye(rank)
    expected_b = (1 - gamma) * b + gamma * torch.linalg.solve(gram_a, (target @ a.T).T).T
    assert torch.allclose(new_b, expected_b, atol=1e-5)


def test_factors_track_a_constant_target():
    """Repeatedly fed the same matrix, B @ A should approach its best rank-r fit."""
    torch.manual_seed(0)
    rows, cols, rank = 32, 24, 6
    target = torch.randn(rows, cols)

    b = torch.zeros(rows, rank)
    a = torch.randn(rank, cols) * 0.02
    for _ in range(200):
        b, a = LoRAPre._factor_step(target, b, a, gamma=0.2, damping=1e-6)

    # Best achievable rank-r error, from the SVD.
    sv = torch.linalg.svdvals(target)
    best = torch.sqrt(sv[rank:].pow(2).sum())
    achieved = torch.linalg.norm(b @ a - target)

    assert achieved < best * 1.5, f"got {achieved:.3f}, best rank-{rank} is {best:.3f}"


@pytest.mark.parametrize(
    "moment, beta, decay_exponent",
    [
        # m_B @ m_A stands in for a zero-init EMA of g with decay beta1.
        ("m", 0.9, 1.0),
        # v_B @ v_A stands in for a zero-init EMA of |g| with decay sqrt(beta2),
        # because (1 - gamma2) ** 2 == beta2 ** 0.5.
        ("v", 0.999, 0.5),
    ],
)
def test_factors_reproduce_the_ema_they_replace(moment, beta, decay_exponent):
    """
    The central correctness claim: the low-rank product must match the EMA it
    stands in for. Fed a constant gradient, a zero-initialised EMA with decay d
    reaches (1 - d ** t) * g, so the reconstruction should too.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(40, 30))
    grad = torch.randn(40, 30)
    # rank 25 < min(40, 30) keeps the low-rank path active while being ample for
    # a rank-1 constant target, so any error is the algorithm's, not truncation.
    opt = LoRAPre([p], lr=0.0, rank=25)

    decay = beta**decay_exponent
    reference = grad if moment == "m" else grad.abs()

    for t in range(1, 61):
        p.grad = grad.clone()
        opt.step()
        if t < 5:
            continue  # skip the first few steps, while m_B is still near zero
        state = opt.state[p]
        recon = (state[f"{moment}_B"] @ state[f"{moment}_A"]).norm().item()
        expected = (1.0 - decay**t) * reference.norm().item()
        assert recon == pytest.approx(expected, rel=0.05), f"step {t}"


def test_second_moment_ema_is_slower_than_first():
    """gamma2 << gamma1, so the v factors move far more slowly than the m ones."""
    gamma1, gamma2 = LoRAPre._resolve_gammas((0.9, 0.999), (None, None))
    assert gamma2 < gamma1 / 100


def test_second_moment_reconstruction_is_nonnegative():
    """v_t = beta2 * (v_B @ v_A)**2 + (1-beta2) * g**2 is a sum of squares."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(32, 24))
    opt = LoRAPre([w], lr=1e-3, rank=4)

    for _ in range(20):
        w.grad = torch.randn_like(w)
        opt.step()

    state = opt.state[w]
    v_recon = (state["v_B"] @ state["v_A"]).pow(2)
    assert (v_recon >= 0).all()
    assert torch.isfinite(v_recon).all()
