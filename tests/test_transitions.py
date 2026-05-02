"""
tests/test_transitions.py
─────────────────────────
Unit tests for Transition.fires() — every operator, guard_type="always",
and missing-feature edge cases.
"""
import pytest
from adaptive_prompt_automaton.core.automaton import Transition, TransitionConfig


def make_transition(**kwargs) -> Transition:
    defaults = dict(
        source_state="a",
        target_state="b",
        guard_type="threshold",
        feature_name="uncertainty_score",
        threshold=0.5,
        operator=">",
        priority=1,
    )
    defaults.update(kwargs)
    return Transition(TransitionConfig(**defaults))


class TestThresholdOperators:
    def test_gt_fires_when_above(self):
        t = make_transition(operator=">", threshold=0.5)
        assert t.fires({"uncertainty_score": 0.6})

    def test_gt_does_not_fire_at_exact(self):
        t = make_transition(operator=">", threshold=0.5)
        assert not t.fires({"uncertainty_score": 0.5})

    def test_gt_does_not_fire_below(self):
        t = make_transition(operator=">", threshold=0.5)
        assert not t.fires({"uncertainty_score": 0.4})

    def test_lt_fires_when_below(self):
        t = make_transition(operator="<", threshold=0.5)
        assert t.fires({"uncertainty_score": 0.4})

    def test_lt_does_not_fire_at_exact(self):
        t = make_transition(operator="<", threshold=0.5)
        assert not t.fires({"uncertainty_score": 0.5})

    def test_gte_fires_at_exact(self):
        t = make_transition(operator=">=", threshold=0.7)
        assert t.fires({"uncertainty_score": 0.7})

    def test_gte_fires_above(self):
        t = make_transition(operator=">=", threshold=0.7)
        assert t.fires({"uncertainty_score": 0.9})

    def test_gte_does_not_fire_below(self):
        t = make_transition(operator=">=", threshold=0.7)
        assert not t.fires({"uncertainty_score": 0.6})

    def test_lte_fires_at_exact(self):
        t = make_transition(operator="<=", threshold=0.3)
        assert t.fires({"uncertainty_score": 0.3})

    def test_lte_fires_below(self):
        t = make_transition(operator="<=", threshold=0.3)
        assert t.fires({"uncertainty_score": 0.1})

    def test_lte_does_not_fire_above(self):
        t = make_transition(operator="<=", threshold=0.3)
        assert not t.fires({"uncertainty_score": 0.31})

    def test_eq_fires_within_tolerance(self):
        t = make_transition(operator="==", threshold=0.5)
        assert t.fires({"uncertainty_score": 0.5})
        assert t.fires({"uncertainty_score": 0.5 + 1e-7})

    def test_eq_does_not_fire_outside_tolerance(self):
        t = make_transition(operator="==", threshold=0.5)
        assert not t.fires({"uncertainty_score": 0.51})


class TestAlwaysGuard:
    def test_always_fires_regardless_of_features(self):
        t = make_transition(guard_type="always", operator="always", threshold=0.0)
        assert t.fires({})
        assert t.fires({"uncertainty_score": 0.0})
        assert t.fires({"uncertainty_score": 1.0})

    def test_operator_always_fires_regardless_of_guard_type(self):
        t = make_transition(guard_type="threshold", operator="always")
        assert t.fires({})


class TestMissingFeature:
    def test_missing_feature_defaults_to_zero(self):
        # Missing feature → 0.0, so "0.0 > 0.5" = False
        t = make_transition(operator=">", threshold=0.5, feature_name="nonexistent")
        assert not t.fires({})

    def test_missing_feature_fires_lte_zero(self):
        # Missing feature → 0.0, so "0.0 <= 0.0" = True
        t = make_transition(operator="<=", threshold=0.0, feature_name="nonexistent")
        assert t.fires({})


class TestUnknownOperator:
    def test_unknown_operator_does_not_fire(self):
        t = make_transition(operator="!=")
        assert not t.fires({"uncertainty_score": 0.9})
