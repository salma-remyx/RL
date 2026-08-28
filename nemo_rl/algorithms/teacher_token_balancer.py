# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token-share balancing for multi-teacher on-policy distillation (MOPD).

Adapted from Open-MOPD (arXiv:2608.19098), which diagnoses the capability
integration gap in multi-teacher on-policy distillation as a misallocation of
the token-level optimization budget: teachers whose domains produce longer
sequences consume a larger share of the loss tokens each step, so concise-task
teachers are starved and stagnate. The paper's token-share balancing mechanism
restores balance by reweighting samples so each teacher's share of the
loss-token budget matches a target share. Written from the paper's described
mechanism (the upstream repo carries no license file).

Scope: only token-share balancing is implemented here. The paper's gap-aware
dynamic budget allocation and student reward refresh are intentionally out of
scope; target shares are configured statically (uniform by default).

The balancing acts on ``repeated_batch["loss_multiplier"]``: every sample's
multiplier is scaled by its teacher's weight, which scales all of that
sample's token losses. Weights are budget-normalized (total weighted token
mass is preserved) and clamped to ``[min_weight, max_weight]`` for stability.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import torch
from pydantic import BaseModel, Field

from nemo_rl.algorithms.opd import _opd_cfg, resolve_reference_aliases

_METRIC_PREFIX = "on_policy_distillation/token_share"


class TokenShareBalanceConfig(BaseModel, extra="allow"):
    """User-facing config for ``on_policy_distillation.token_share_balancing``."""

    enabled: bool = False
    # Teacher alias -> target share of the loss-token budget. Empty means
    # uniform across configured teachers. Shares are normalized to sum to 1.
    target_token_shares: dict[str, float] = Field(default_factory=dict)
    # Stability clamp on the per-teacher loss weight.
    min_weight: float = 0.25
    max_weight: float = 4.0


def get_token_share_balance_config(master_config: Any) -> TokenShareBalanceConfig:
    """Parse the ``token_share_balancing`` sub-config (absent means disabled)."""
    raw = _opd_cfg(master_config).get("token_share_balancing") or {}
    return TokenShareBalanceConfig(**raw)


def _sanitize_metric_key(alias: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", alias)


def resolve_target_shares(
    balance_cfg: TokenShareBalanceConfig, present_aliases: list[str]
) -> dict[str, float]:
    """Target token share per present teacher alias, normalized to sum to 1.

    Teachers absent from the batch cannot receive budget, so configured
    targets are renormalized over the aliases that are present.
    """
    if balance_cfg.target_token_shares:
        raw = {
            alias: balance_cfg.target_token_shares.get(alias, 0.0)
            for alias in present_aliases
        }
    else:
        raw = {alias: 1.0 for alias in present_aliases}
    total = sum(raw.values())
    if total <= 0:
        # Every present alias was explicitly assigned a zero share; fall back
        # to uniform rather than dividing by zero.
        return {alias: 1.0 / len(present_aliases) for alias in present_aliases}
    return {alias: share / total for alias, share in raw.items()}


def validate_token_share_balance(
    master_config: Any,
) -> Optional[TokenShareBalanceConfig]:
    """Validate the balancing config at launch; return it when enabled.

    Raises on misconfiguration (unknown aliases, non-positive shares, invalid
    clamp, or balancing enabled while OPD itself is disabled) so a bad config
    fails fast instead of silently training unbalanced.
    """
    balance_cfg = get_token_share_balance_config(master_config)
    if not balance_cfg.enabled:
        return None

    opd_cfg = _opd_cfg(master_config)
    if not opd_cfg.get("enabled", False):
        raise ValueError(
            "on_policy_distillation.token_share_balancing.enabled=true requires "
            "on_policy_distillation.enabled=true."
        )
    if not (0.0 < balance_cfg.min_weight <= 1.0 <= balance_cfg.max_weight):
        raise ValueError(
            "token_share_balancing requires 0 < min_weight <= 1 <= max_weight, got "
            f"min_weight={balance_cfg.min_weight}, max_weight={balance_cfg.max_weight}."
        )

    configured_aliases = set(opd_cfg.get("teacher_model_by_agent_name", {}))
    unknown = set(balance_cfg.target_token_shares) - configured_aliases
    if unknown:
        raise ValueError(
            f"token_share_balancing.target_token_shares names unknown teacher "
            f"alias(es) {sorted(unknown)}; configured aliases: "
            f"{sorted(configured_aliases)}."
        )
    non_positive = {
        alias: share
        for alias, share in balance_cfg.target_token_shares.items()
        if share <= 0
    }
    if non_positive:
        raise ValueError(
            "token_share_balancing.target_token_shares must be positive, "
            f"got {non_positive}."
        )
    missing = configured_aliases - set(balance_cfg.target_token_shares)
    if balance_cfg.target_token_shares and missing:
        print(
            f"[token_share_balancing] target_token_shares omits configured "
            f"teacher alias(es) {sorted(missing)}; they get a zero target "
            "share (downweighted to min_weight).",
            flush=True,
        )

    if len(configured_aliases) < 2:
        print(
            "[token_share_balancing] Enabled with fewer than two configured "
            "teachers; balancing is a no-op for single-teacher runs.",
            flush=True,
        )
    resolved = resolve_target_shares(balance_cfg, sorted(configured_aliases))
    plan = ", ".join(f"{alias}={share:.3f}" for alias, share in resolved.items())
    print(
        f"[token_share_balancing] Target loss-token shares: {plan} "
        f"(clamp [{balance_cfg.min_weight}, {balance_cfg.max_weight}])",
        flush=True,
    )
    return balance_cfg


def _normalize_agent_refs(agent_refs: Any) -> list[dict[str, Any]]:
    """Coerce per-sample agent refs to the dicts resolve_reference_aliases wants."""
    refs = list(agent_refs)
    normalized: list[dict[str, Any]] = []
    for ref in refs:
        if isinstance(ref, dict):
            normalized.append(ref)
        elif isinstance(ref, str):
            normalized.append({"name": ref})
        else:
            raise TypeError(
                "token_share_balancing expects repeated_batch['agent_ref'] entries "
                f"to be dicts or strings, got {type(ref).__name__}."
            )
    return normalized


def compute_teacher_token_shares(
    teacher_aliases: list[str],
    loss_token_counts: torch.Tensor,
    loss_multiplier: torch.Tensor,
) -> dict[str, float]:
    """Each teacher's current share of the weighted loss-token budget."""
    counts = loss_token_counts.detach().to(dtype=torch.float64, device="cpu")
    multipliers = loss_multiplier.detach().to(dtype=torch.float64, device="cpu")
    mass = counts * multipliers
    total = float(mass.sum())
    shares: dict[str, float] = {}
    for alias in set(teacher_aliases):
        sel = torch.tensor(
            [a == alias for a in teacher_aliases], dtype=torch.bool
        )
        shares[alias] = float(mass[sel].sum()) / total if total > 0 else 0.0
    return shares


def apply_token_share_balancing(
    repeated_batch: Any,
    loss_token_counts: torch.Tensor,
    master_config: Any,
) -> dict[str, float]:
    """Reweight per-sample loss multipliers toward target teacher token shares.

    Reads ``repeated_batch["agent_ref"]`` to attribute each sample to a
    teacher alias, reassigns ``repeated_batch["loss_multiplier"]`` with the
    reweighted values (matching ``_apply_mask_sample_filter``), and returns
    balancing metrics to merge into the step's rollout metrics. Returns an
    empty dict when balancing is disabled or cannot run.
    """
    balance_cfg = get_token_share_balance_config(master_config)
    if not balance_cfg.enabled:
        return {}

    agent_refs = repeated_batch.get("agent_ref")
    if agent_refs is None:
        print(
            "[token_share_balancing] enabled but repeated_batch has no "
            "'agent_ref'; skipping balancing for this step.",
            flush=True,
        )
        return {}

    opd_cfg = _opd_cfg(master_config)
    aliases = resolve_reference_aliases(
        _normalize_agent_refs(agent_refs),
        dict(opd_cfg.get("teacher_model_by_agent_name", {})),
        default_teacher_alias=opd_cfg.get("default_teacher_alias"),
        strict_agent_name_match=bool(opd_cfg.get("strict_agent_name_match", False)),
    )

    loss_multiplier = repeated_batch["loss_multiplier"]
    if not isinstance(loss_multiplier, torch.Tensor):
        loss_multiplier = torch.tensor(loss_multiplier, dtype=torch.float32)
    loss_multiplier = loss_multiplier.to(dtype=torch.float32)

    present_aliases = sorted(set(aliases))
    shares_before = compute_teacher_token_shares(
        aliases, loss_token_counts, loss_multiplier
    )
    if not any(share > 0 for share in shares_before.values()):
        # No unmasked loss tokens this step; nothing to rebalance.
        return {}

    # Only teachers with unmasked loss tokens can receive budget; renormalize
    # targets over them.
    active_aliases = [
        alias for alias in present_aliases if shares_before[alias] > 0
    ]
    target_shares = resolve_target_shares(balance_cfg, active_aliases)

    # w_k = t_k / s_k. Budget is exactly preserved before clamping: both the
    # actual and the target shares sum to 1 over the active teachers, so
    # sum(w_k * M_k) = sum(t_k / s_k * s_k * T) = T. Clamping trades exact
    # budget preservation for stability.
    weights = {alias: 1.0 for alias in present_aliases}
    for alias in active_aliases:
        raw = target_shares[alias] / shares_before[alias]
        weights[alias] = min(
            max(raw, balance_cfg.min_weight), balance_cfg.max_weight
        )

    weight_column = torch.tensor(
        [weights[alias] for alias in aliases], dtype=torch.float32
    )
    repeated_batch["loss_multiplier"] = loss_multiplier * weight_column.to(
        loss_multiplier.device
    )

    shares_after = compute_teacher_token_shares(
        aliases, loss_token_counts, repeated_batch["loss_multiplier"]
    )
    metrics: dict[str, float] = {
        f"{_METRIC_PREFIX}/num_teachers_present": float(len(present_aliases)),
    }
    for alias in present_aliases:
        key = _sanitize_metric_key(alias)
        metrics[f"{_METRIC_PREFIX}/share_before__{key}"] = shares_before[alias]
        metrics[f"{_METRIC_PREFIX}/share_after__{key}"] = shares_after[alias]
        metrics[f"{_METRIC_PREFIX}/weight__{key}"] = weights[alias]
    return metrics
