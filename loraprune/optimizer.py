"""
LoRA-Pre: an Adam optimizer whose momenta are stored as low-rank factors.

Implements Algorithm 1 of Wang et al., "Taming Momentum: Rethinking Optimizer
States Through Low-Rank Approximation" (arXiv:2602.24283).

The paper's starting point is that an exponential moving average

    m_t = b * m_{t-1} + (1 - b) * g_t
        = m_{t-1} - (1 - b) * (m_{t-1} - g_t)

is exactly one gradient-descent step on ``L(m, g) = 0.5 * ||m - g||_F^2`` with
learning rate ``1 - b``.  Once the EMA is seen as *training* a variable rather
than accumulating one, the variable can be re-parameterised.  LoRA-Pre writes
the momentum as a product of two thin matrices ``m = m_B @ m_A`` and derives
closed-form (Newton) updates for the factors, so the full ``p x q`` momentum is
never held in optimizer state.

Two details of the paper are intentionally left out of this implementation:

* The "scale factor of 0.25 as used in GaLore" appears in the experimental-setup
  prose, describing alignment with the GaLore baseline, and not in Algorithm 1.
  This module implements Algorithm 1 as printed; scale it via ``lr`` if you want
  to reproduce that setting.
* There is no transpose trick.  The paper factorises the gradient ``G``
  uniformly and never conditions the factorisation on which of ``p``, ``q`` is
  larger.
"""

import math

import torch
from torch.optim import Optimizer

_DEFAULT_INIT_STD = 0.02  # N(0, 0.02) init for the A factors, per Algorithm 1
_DEFAULT_DAMPING = 1e-6   # Tikhonov damping for the r x r solves, per Appendix C.1


class LoRAPre(Optimizer):
    """
    Adam with low-rank first- and second-order momenta (LoRA-Pre).

    For every 2-D parameter of shape ``(p, q)`` the optimizer keeps four thin
    factors instead of two full moment tensors::

        m_B, v_B : (p, r)      m_A, v_A : (r, q)

    so persistent state shrinks from ``2 * p * q`` to ``2 * (p + q) * r``
    elements.  The moments themselves are rebuilt from the factors at each step
    and discarded immediately.

    Parameters that cannot benefit from the factorisation -- anything that is
    not 2-D (biases, LayerNorm weights) and any matrix with
    ``min(p, q) <= rank`` -- fall back to a standard AdamW step, which is what
    the paper prescribes for "other parameters".  The choice is made per
    parameter on its first step, so ``model.parameters()`` can be handed to this
    optimizer directly.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize, or of dicts defining parameter
        groups, as for any ``torch.optim.Optimizer``.
    lr : float, default 1e-3
        Learning rate, ``gamma`` in the paper.
    betas : tuple of (float, float), default (0.9, 0.999)
        EMA coefficients ``(beta1, beta2)`` for the reconstructed first and
        second moments.  Both must lie in ``[0, 1)``.
    gammas : tuple of (float or None, float or None), default (None, None)
        Factorization learning rates ``(gamma1, gamma2)``.  ``None`` means
        "derive from ``betas``" using the paper's default coupling
        ``1 - gamma1 = sqrt(beta1)`` and ``1 - gamma2 = beta2 ** 0.25``
        (Appendix B.1).  Leave these alone unless you know why you are changing
        them -- see Notes for why the exponents are what they are.
    eps : float, default 1e-8
        Term added to the denominator for numerical stability.
    weight_decay : float, default 0.0
        Decoupled weight decay ``lambda``.  Applied as ``lr * lambda * theta``,
        matching the paper's final update line and AdamW's convention.
    rank : int, default 8
        Rank ``r`` of the momentum factorisation.  The paper uses 8 for
        fine-tuning and 128-512 for pre-training, scaling with model size.
    init_std : float, default 0.02
        Standard deviation of the Gaussian used to initialise ``m_A`` and
        ``v_A``.  The ``B`` factors start at zero, as in Algorithm 1.
    damping : float, default 1e-6
        Tikhonov damping added to the diagonal of the ``r x r`` systems before
        solving.  Required, not optional: ``m_B`` starts at zero, so
        ``m_B^T m_B`` is exactly singular on the first step.

    Notes
    -----
    **Why the default gammas use square and fourth roots.**  Each factor decays
    by ``(1 - gamma1)`` per step, so their product decays by ``(1 - gamma1) ** 2``.
    Setting ``1 - gamma1 = sqrt(beta1)`` therefore makes the reconstructed
    momentum decay at exactly ``beta1``, matching the EMA it stands in for.  The
    second moment is reconstructed as ``(v_B @ v_A) ** 2``, adding one more
    factor of two to the exponent, hence ``1 - gamma2 = beta2 ** 0.25``.

    **The second-moment factors track ``|g|``, not ``g ** 2``.**  The product
    ``v_B @ v_A`` approximates an EMA of the gradient magnitude and is squared
    only when the moment is reconstructed.  Factorising ``g ** 2`` directly would
    be a different (and worse-conditioned) algorithm.

    **Memory, stated honestly.**  The saving is in *persistent* state.
    Reconstructing ``m_B @ m_A`` allocates a transient ``p x q`` buffer, so peak
    memory falls by less than the state figures suggest.  The win is still real,
    because transient buffers are freed and reused while optimizer state lives
    for the whole run.

    **Epsilon placement.**  The rendered fraction in the paper is consistent with
    either ``sqrt(v_hat + eps)`` or ``sqrt(v_hat) + eps``.  This implementation
    uses the latter, matching standard Adam and PyTorch's ``AdamW``.

    **Use warmup.**  Algorithm 1 applies Adam's ``1/(1 - beta2 ** t)`` bias
    correction to a second moment whose factors decay at ``sqrt(beta2)``, not
    ``beta2``.  The two disagree early in training: at the default betas the
    reconstructed ``v_hat`` is roughly an order of magnitude too small around
    step 10, so the denominator is too small and the first few dozen updates
    overshoot.  It is self-correcting -- the mismatch vanishes as ``t`` grows and
    training converges normally -- but it is a property of the algorithm as
    published, not of this implementation, and it argues for a nonzero
    ``warmup_steps`` rather than the zero currently set in ``prune.py``.

    Examples
    --------
    >>> import torch
    >>> model = torch.nn.Linear(128, 128)
    >>> opt = LoRAPre(model.parameters(), lr=1e-3, rank=16)
    >>> loss = model(torch.randn(8, 128)).pow(2).mean()
    >>> loss.backward()
    >>> opt.step()
    >>> opt.zero_grad()

    References
    ----------
    .. [1] Z. Wang, J. Liang, R. He, Z. Wang, T. Tan, "Taming Momentum:
           Rethinking Optimizer States Through Low-Rank Approximation",
           arXiv:2602.24283, 2026.  Algorithm 1 (LoRA-Pre for Adam), Eq. 11-12,
           Appendix B.1 (gamma/beta coupling), Appendix C.1 (damped
           pseudo-inverses).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        gammas: tuple[float | None, float | None] = (None, None),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rank: int = 8,
        init_std: float = _DEFAULT_INIT_STD,
        damping: float = _DEFAULT_DAMPING,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if init_std <= 0.0:
            raise ValueError(f"Invalid init_std: {init_std}")
        if damping <= 0.0:
            # Zero damping makes the first step a singular solve, since m_B is
            # initialised to zeros. Refuse rather than emit silent NaNs.
            raise ValueError(f"Invalid damping (must be > 0): {damping}")

        gamma1, gamma2 = gammas
        if gamma1 is not None and not 0.0 < gamma1 <= 1.0:
            raise ValueError(f"Invalid gamma1: {gamma1}")
        if gamma2 is not None and not 0.0 < gamma2 <= 1.0:
            raise ValueError(f"Invalid gamma2: {gamma2}")

        defaults = dict(
            lr=lr,
            betas=betas,
            gammas=gammas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            init_std=init_std,
            damping=damping,
        )
        super().__init__(params, defaults)

    @staticmethod
    def uses_low_rank(param: torch.Tensor, rank: int) -> bool:
        """
        Report whether a parameter takes the low-rank path or the AdamW fallback.

        Parameters
        ----------
        param : torch.Tensor
            The parameter to classify.
        rank : int
            The factorization rank ``r``.

        Returns
        -------
        bool
            ``True`` if the low-rank momenta apply, ``False`` if this parameter
            falls back to a standard AdamW step.

        Notes
        -----
        Public so callers can report coverage *before* the first step, when the
        state has not been allocated yet.  The factorisation only pays off when
        the factors are strictly smaller than the matrix they replace, which
        needs ``min(p, q) > r``; at ``min(p, q) <= r`` they would not be.
        """
        return param.dim() == 2 and min(param.shape) > rank

    @staticmethod
    def _resolve_gammas(
        betas: tuple[float, float],
        gammas: tuple[float | None, float | None],
    ) -> tuple[float, float]:
        """
        Fill in unset factorization rates from the EMA coefficients.

        Parameters
        ----------
        betas : tuple of (float, float)
            ``(beta1, beta2)``.
        gammas : tuple of (float or None, float or None)
            ``(gamma1, gamma2)``; ``None`` entries are derived.

        Returns
        -------
        tuple of (float, float)
            The resolved ``(gamma1, gamma2)``.

        Notes
        -----
        Appendix B.1 of the paper: ``1 - gamma1 = sqrt(beta1)`` so that the
        product of two factors, each decaying by ``1 - gamma1``, decays by
        ``beta1``; ``1 - gamma2 = beta2 ** 0.25`` because the second moment is
        additionally squared on reconstruction.
        """
        beta1, beta2 = betas
        gamma1, gamma2 = gammas
        if gamma1 is None:
            gamma1 = 1.0 - math.sqrt(beta1)
        if gamma2 is None:
            gamma2 = 1.0 - beta2**0.25
        return gamma1, gamma2

    @staticmethod
    def _solve_spd(gram: torch.Tensor, rhs: torch.Tensor, damping: float) -> torch.Tensor:
        """
        Solve ``(gram + damping * I) @ X = rhs`` for ``X``.

        Parameters
        ----------
        gram : torch.Tensor
            Symmetric positive semi-definite matrix of shape ``(r, r)``.
        rhs : torch.Tensor
            Right-hand side of shape ``(r, k)``.
        damping : float
            Tikhonov damping added to the diagonal.

        Returns
        -------
        torch.Tensor
            Solution of shape ``(r, k)``.

        Notes
        -----
        This stands in for the damped pseudo-inverses of Appendix C.1.  Solving
        is preferred over forming an explicit inverse: it is cheaper and better
        conditioned.  The systems are only ``r x r``, so the cost is negligible
        next to the ``p x q`` matmuls around it.

        Two fallbacks, for two different failures.  A backend that does not
        implement ``linalg.solve`` at all (MPS, as of torch 2.5) is handled by
        solving on the CPU: the system is only ``r x r``, so the round trip is
        cheap and the result is identical.  A genuinely singular system -- which
        damping should prevent, but low precision can still produce -- falls
        back to an explicit pseudo-inverse.
        """
        rank = gram.shape[0]
        eye = torch.eye(rank, dtype=gram.dtype, device=gram.device)
        damped = gram + damping * eye
        try:
            return torch.linalg.solve(damped, rhs)
        except NotImplementedError:
            # Device gap, not a numerical problem. Must be caught before
            # RuntimeError, which it subclasses.
            return torch.linalg.solve(damped.cpu(), rhs.cpu()).to(rhs.device)
        except RuntimeError:
            # torch.linalg.solve raises on a singular system; pinv always works.
            return torch.linalg.pinv(damped) @ rhs

    @classmethod
    def _factor_step(
        cls,
        target: torch.Tensor,
        factor_b: torch.Tensor,
        factor_a: torch.Tensor,
        gamma: float,
        damping: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance one low-rank pair by a single online-Newton step toward ``target``.

        Parameters
        ----------
        target : torch.Tensor
            Matrix of shape ``(p, q)`` the product ``B @ A`` should track.  This
            is ``g`` for the first moment and ``|g|`` for the second.
        factor_b : torch.Tensor
            Current left factor, shape ``(p, r)``.
        factor_a : torch.Tensor
            Current right factor, shape ``(r, q)``.
        gamma : float
            Factorization learning rate.
        damping : float
            Tikhonov damping for the two ``r x r`` solves.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            The new ``(B, A)`` factors.

        Notes
        -----
        Implements Eq. 11-12 of the paper::

            B <- (1 - gamma) * B + gamma * target @ A^T (A A^T)^-1
            A <- (1 - gamma) * A + gamma * (B^T B)^-1 B^T @ target

        Both lines read the *old* ``B`` and ``A``: the paper subscripts every
        right-hand-side factor ``t-1``, making this a simultaneous (Jacobi)
        update rather than a sequential (Gauss-Seidel) one.  That is why the
        results are returned instead of written in place -- overwriting ``B``
        before computing ``A`` would silently implement a different algorithm.
        """
        decay = 1.0 - gamma

        # B-step: least-squares projection of `target` onto the row space of A.
        # Solving (A A^T) X^T = (target A^T)^T is the transposed form of
        # target @ A^T (A A^T)^-1.
        gram_a = factor_a @ factor_a.T                              # (r, r)
        proj_b = cls._solve_spd(gram_a, (target @ factor_a.T).T, damping).T  # (p, r)
        new_b = decay * factor_b + gamma * proj_b

        # A-step: least-squares projection of `target` onto the column space of
        # B. On the very first step B is all zeros, so the damped solve returns
        # zeros and this reduces to pure decay -- well defined, no special case.
        gram_b = factor_b.T @ factor_b                              # (r, r)
        proj_a = cls._solve_spd(gram_b, factor_b.T @ target, damping)  # (r, q)
        new_a = decay * factor_a + gamma * proj_a

        return new_b, new_a

    def _init_state(self, state: dict, param: torch.Tensor, group: dict) -> None:
        """
        Allocate optimizer state for a parameter on its first step.

        Parameters
        ----------
        state : dict
            The (empty) per-parameter state dict to populate.
        param : torch.Tensor
            The parameter being initialised.
        group : dict
            The parameter group, supplying ``rank`` and ``init_std``.

        Returns
        -------
        None

        Notes
        -----
        Decides here, once, whether this parameter uses the low-rank path or the
        AdamW fallback, and records it in ``state['low_rank']``.  State is always
        float32: the ``r x r`` solves are ill-conditioned by construction (the
        Gram matrices start at or near zero) and are not safe in fp16/bf16, and
        keeping momenta in fp32 is standard practice for mixed-precision runs.
        """
        rank = group["rank"]
        low_rank = self.uses_low_rank(param, rank)

        state["step"] = 0
        state["low_rank"] = low_rank

        if not low_rank:
            state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            return

        rows, cols = param.shape
        std = group["init_std"]
        opts = dict(dtype=torch.float32, device=param.device)

        # Algorithm 1: B factors start at zero, A factors at N(0, init_std).
        # A must be non-degenerate or its row space is empty and the B-step
        # projects onto nothing.
        state["m_B"] = torch.zeros(rows, rank, **opts)
        state["m_A"] = torch.randn(rank, cols, **opts) * std
        state["v_B"] = torch.zeros(rows, rank, **opts)
        state["v_A"] = torch.randn(rank, cols, **opts) * std

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        float or None
            The loss returned by ``closure``, or ``None`` if no closure was
            given.

        Raises
        ------
        RuntimeError
            If any gradient is sparse; LoRA-Pre needs dense gradients to form
            the low-rank projections.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            gamma1, gamma2 = self._resolve_gammas(group["betas"], group["gammas"])
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            damping = group["damping"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("LoRAPre does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    self._init_state(state, param, group)

                state["step"] += 1
                step = state["step"]
                grad = param.grad.to(torch.float32)

                if state["low_rank"]:
                    m_B, m_A = state["m_B"], state["m_A"]
                    v_B, v_A = state["v_B"], state["v_A"]

                    # Steps 4 and 8 of Algorithm 1: rebuild both moments from the
                    # *previous* factors plus the fresh gradient. This must happen
                    # before the factors are advanced below.
                    exp_avg = beta1 * (m_B @ m_A) + (1.0 - beta1) * grad
                    # v's factors track |g|, so the reconstruction is squared here
                    # rather than the factors tracking g ** 2 directly.
                    exp_avg_sq = beta2 * (v_B @ v_A).pow(2) + (1.0 - beta2) * grad.pow(2)

                    # Steps 5-6 and 9-10: advance the factors. Written back only
                    # after both new factors are computed (see _factor_step).
                    state["m_B"], state["m_A"] = self._factor_step(
                        grad, m_B, m_A, gamma1, damping
                    )
                    state["v_B"], state["v_A"] = self._factor_step(
                        grad.abs(), v_B, v_A, gamma2, damping
                    )
                else:
                    # Standard Adam moments for parameters the paper leaves out
                    # of the factorisation.
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Steps 11-13, identical for both paths: bias-correct, normalise,
                # then apply decoupled weight decay inside the lr-scaled update.
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                update = m_hat / (v_hat.sqrt() + eps)
                if weight_decay != 0.0:
                    update = update + weight_decay * param.to(torch.float32)

                param.add_(update.to(param.dtype), alpha=-lr)

        return loss
