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

import pytest
import torch

from examples.run_grpo import _validate_opd_token_share_balance
from nemo_rl.algorithms.teacher_token_balancer import (
    apply_token_share_balancing,
    compute_teacher_token_shares,
    resolve_target_shares,
    TokenShareBalanceConfig,
)


def _opd_master_config(token_share_balancing):
    """Minimal dict master-config carrying an OPD block with two teachers."""
    return {
        "on_policy_distillation": {
            "enabled": True,
            "teacher_model_by_agent_name": {
                "math_teacher": "org/math-expert",
                "if_teacher": "org/if-expert",
            },
            "default_teacher_alias": "math_teacher",
            "token_share_balancing": token_share_balancing,
        }
    }


def _skewed_batch():
    """Two teachers where math_teacher holds 60% of the loss-token budget."""
    repeated_batch = {
        "agent_ref": [
            {"name": "math_teacher"},
            {"name": "math_teacher"},
            {"name": "if_teacher"},
            {"name": "if_teacher"},
        ],
        "loss_multiplier": torch.ones(4),
    }
    loss_token_counts = torch.tensor([320.0, 280.0, 200.0, 200.0])
    return repeated_batch, loss_token_counts


class TestLauncherWiring:
    """Exercise the examples/run_grpo.py validation hook (the call-site edit)."""

    def test_valid_config_returns_none_and_does_not_raise(self, capsys):
        _validate_opd_token_share_balance(
            _opd_master_config({"enabled": True})
        )
        out = capsys.readouterr().out
        assert "math_teacher=0.500" in out
        assert "if_teacher=0.500" in out

    def test_disabled_config_is_silent_noop(self, capsys):
        _validate_opd_token_share_balance(
            _opd_master_config({"enabled": False})
        )
        assert capsys.readouterr().out == ""

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="unknown teacher"):
            _validate_opd_token_share_balance(
                _opd_master_config(
                    {"enabled": True, "target_token_shares": {"ghost": 1.0}}
                )
            )

    def test_non_positive_share_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            _validate_opd_token_share_balance(
                _opd_master_config(
                    {"enabled": True, "target_token_shares": {"math_teacher": 0.0}}
                )
            )

    def test_invalid_clamp_raises(self):
        with pytest.raises(ValueError, match="min_weight"):
            _validate_opd_token_share_balance(
                _opd_master_config(
                    {"enabled": True, "min_weight": 2.0, "max_weight": 1.0}
                )
            )

    def test_balancing_requires_opd_enabled(self):
        config = _opd_master_config({"enabled": True})
        config["on_policy_distillation"]["enabled"] = False
        with pytest.raises(ValueError, match="on_policy_distillation.enabled"):
            _validate_opd_token_share_balance(config)


class TestResolveTargetShares:
    def test_uniform_default_over_present_aliases(self):
        cfg = TokenShareBalanceConfig(enabled=True)
        shares = resolve_target_shares(cfg, ["a", "b"])
        assert shares == {"a": 0.5, "b": 0.5}

    def test_configured_shares_normalized_over_present_aliases(self):
        cfg = TokenShareBalanceConfig(
            enabled=True,
            target_token_shares={"a": 3.0, "b": 1.0, "absent": 4.0},
        )
        shares = resolve_target_shares(cfg, ["a", "b"])
        assert shares == {"a": 0.75, "b": 0.25}

    def test_all_zero_configured_shares_fall_back_to_uniform(self):
        cfg = TokenShareBalanceConfig(
            enabled=True, target_token_shares={"other": 1.0}
        )
        shares = resolve_target_shares(cfg, ["a", "b"])
        assert shares == {"a": 0.5, "b": 0.5}


class TestComputeTeacherTokenShares:
    def test_shares_track_weighted_token_mass(self):
        aliases = ["a", "a", "b"]
        counts = torch.tensor([300.0, 300.0, 200.0])
        multipliers = torch.tensor([1.0, 1.0, 1.0])
        shares = compute_teacher_token_shares(aliases, counts, multipliers)
        assert shares["a"] == pytest.approx(0.75)
        assert shares["b"] == pytest.approx(0.25)

    def test_masked_samples_contribute_no_mass(self):
        aliases = ["a", "b"]
        counts = torch.tensor([900.0, 100.0])
        multipliers = torch.tensor([0.0, 1.0])
        shares = compute_teacher_token_shares(aliases, counts, multipliers)
        assert shares["a"] == 0.0
        assert shares["b"] == 1.0


class TestApplyTokenShareBalancing:
    def test_reweights_toward_uniform_token_shares(self):
        repeated_batch, loss_token_counts = _skewed_batch()
        metrics = apply_token_share_balancing(
            repeated_batch,
            loss_token_counts,
            _opd_master_config({"enabled": True}),
        )

        # math_teacher starts at 60% of the token budget, if_teacher at 40%.
        assert metrics[
            "on_policy_distillation/token_share/share_before__math_teacher"
        ] == pytest.approx(0.6)
        assert metrics[
            "on_policy_distillation/token_share/share_before__if_teacher"
        ] == pytest.approx(0.4)

        # Weights move each teacher toward the 50% target: the dominant
        # teacher is scaled down, the starved teacher up.
        assert (
            metrics["on_policy_distillation/token_share/weight__math_teacher"]
            < 1.0
        )
        assert (
            metrics["on_policy_distillation/token_share/weight__if_teacher"] > 1.0
        )
        assert metrics[
            "on_policy_distillation/token_share/share_after__math_teacher"
        ] == pytest.approx(0.5)
        assert metrics[
            "on_policy_distillation/token_share/share_after__if_teacher"
        ] == pytest.approx(0.5)

        # Total weighted token mass is preserved by the reweighting.
        new_multiplier = repeated_batch["loss_multiplier"]
        assert float((new_multiplier * loss_token_counts).sum()) == pytest.approx(
            float(loss_token_counts.sum())
        )

    def test_weights_respect_clamp(self):
        # 90/10 skew: the unclamped if_teacher weight would be 0.5/0.1 = 5.0.
        repeated_batch = {
            "agent_ref": [
                {"name": "math_teacher"},
                {"name": "if_teacher"},
            ],
            "loss_multiplier": torch.ones(2),
        }
        loss_token_counts = torch.tensor([900.0, 100.0])
        metrics = apply_token_share_balancing(
            repeated_batch,
            loss_token_counts,
            _opd_master_config({"enabled": True, "max_weight": 2.0}),
        )
        assert metrics[
            "on_policy_distillation/token_share/weight__if_teacher"
        ] == pytest.approx(2.0)

    def test_disabled_returns_empty_and_leaves_batch_untouched(self):
        repeated_batch, loss_token_counts = _skewed_batch()
        original = repeated_batch["loss_multiplier"].clone()
        metrics = apply_token_share_balancing(
            repeated_batch,
            loss_token_counts,
            _opd_master_config({"enabled": False}),
        )
        assert metrics == {}
        assert torch.equal(repeated_batch["loss_multiplier"], original)

    def test_missing_agent_ref_skips_with_warning(self, capsys):
        repeated_batch = {"loss_multiplier": torch.ones(2)}
        metrics = apply_token_share_balancing(
            repeated_batch,
            torch.tensor([10.0, 10.0]),
            _opd_master_config({"enabled": True}),
        )
        assert metrics == {}
        assert "no 'agent_ref'" in capsys.readouterr().out

    def test_fully_masked_batch_is_noop(self):
        repeated_batch, loss_token_counts = _skewed_batch()
        repeated_batch["loss_multiplier"] = torch.zeros(4)
        metrics = apply_token_share_balancing(
            repeated_batch,
            loss_token_counts,
            _opd_master_config({"enabled": True}),
        )
        assert metrics == {}
