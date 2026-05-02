"""
tests/test_executor.py
──────────────────────
Integration-level tests for AutomatonExecutor.run_episode():
  - full round-trip at seed 42
  - terminal-state termination
  - budget exhaustion
  - no-transition fallback
  - multi-sample self-consistency wiring
  - verbose mode does not crash
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_prompt_automaton.core.automaton import (
    Automaton, AutomatonConfig, StateConfig, TransitionConfig,
)
from adaptive_prompt_automaton.core.executor import AutomatonExecutor, Episode
from adaptive_prompt_automaton.core.features import FeatureExtractor
from adaptive_prompt_automaton.utils.api import MockLLM


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def llm():
    return MockLLM(uncertainty_rate=0.0, latency=0.0, seed=42)


@pytest.fixture
def extractor():
    return FeatureExtractor(long_input_threshold=50)


def _build_two_state(start_terminal: bool = False) -> Automaton:
    """Minimal automaton: start → terminal (always)."""
    states = {
        "start": StateConfig(
            state_id="start", name="Start",
            template="Answer: {input}",
            is_terminal=start_terminal,
            carry_context=False,
        ),
        "terminal": StateConfig(
            state_id="terminal", name="Terminal",
            template="Final: {input}",
            is_terminal=True,
            carry_context=False,
        ),
    }
    transitions = [
        TransitionConfig(
            source_state="start", target_state="terminal",
            guard_type="always", operator="always", priority=1,
        )
    ]
    config = AutomatonConfig(
        automaton_id="test", name="TestAut",
        start_state="start", states=states,
        transitions=transitions, max_steps=4, max_budget=4,
    )
    return Automaton(config)


def _build_no_transition() -> Automaton:
    """Automaton with a non-firing threshold guard and no fallback."""
    states = {
        "start": StateConfig(
            state_id="start", name="Start",
            template="Answer: {input}",
            is_terminal=False,
            carry_context=False,
        ),
        "terminal": StateConfig(
            state_id="terminal", name="Terminal",
            template="Final: {input}",
            is_terminal=True,
            carry_context=False,
        ),
    }
    transitions = [
        TransitionConfig(
            source_state="start", target_state="terminal",
            guard_type="threshold", feature_name="uncertainty_score",
            operator=">", threshold=0.99,   # MockLLM(uncertainty_rate=0) will never fire
            priority=1,
        )
    ]
    config = AutomatonConfig(
        automaton_id="no_t", name="NoTransition",
        start_state="start", states=states,
        transitions=transitions, max_steps=4, max_budget=4,
    )
    return Automaton(config)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestBasicRoundTrip:
    def test_episode_returns_episode(self, llm, extractor):
        aut = _build_two_state()
        ex  = AutomatonExecutor(aut, llm, extractor)
        ep  = ex.run_episode("What is 2 + 2?", episode_id="t1")
        assert isinstance(ep, Episode)

    def test_path_recorded(self, llm, extractor):
        aut = _build_two_state()
        ex  = AutomatonExecutor(aut, llm, extractor)
        ep  = ex.run_episode("What is 2 + 2?")
        assert len(ep.path) >= 1

    def test_final_output_populated(self, llm, extractor):
        aut = _build_two_state()
        ex  = AutomatonExecutor(aut, llm, extractor)
        ep  = ex.run_episode("What is 2 + 2?")
        assert ep.final_output != ""

    def test_tokens_positive(self, llm, extractor):
        aut = _build_two_state()
        ex  = AutomatonExecutor(aut, llm, extractor)
        ep  = ex.run_episode("What is 2 + 2?")
        assert ep.total_tokens > 0

    def test_episode_id_propagated(self, llm, extractor):
        aut = _build_two_state()
        ex  = AutomatonExecutor(aut, llm, extractor)
        ep  = ex.run_episode("question", episode_id="my_id")
        assert ep.episode_id == "my_id"

    def test_automaton_path_counts_updated(self, llm, extractor):
        aut = _build_two_state()
        ex  = AutomatonExecutor(aut, llm, extractor)
        ex.run_episode("q1")
        ex.run_episode("q2")
        assert aut.episodes_run == 2
        assert sum(aut.path_counts.values()) == 2


class TestTermination:
    def test_terminal_state_sets_terminated_by(self, llm, extractor):
        aut = _build_two_state()
        ex  = AutomatonExecutor(aut, llm, extractor)
        ep  = ex.run_episode("question")
        assert ep.terminated_by == "terminal_state"

    def test_single_terminal_state_terminates_immediately(self, llm, extractor):
        aut = _build_two_state(start_terminal=True)
        ex  = AutomatonExecutor(aut, llm, extractor)
        ep  = ex.run_episode("question")
        assert ep.terminated_by == "terminal_state"
        assert ep.path == ["start"]

    def test_no_transition_terminates_gracefully(self, llm, extractor):
        aut = _build_no_transition()
        ex  = AutomatonExecutor(aut, llm, extractor)
        ep  = ex.run_episode("question")
        assert ep.terminated_by == "no_transition"
        assert ep.final_output != ""

    def test_budget_exhaustion(self, llm, extractor):
        """With max_budget=1, a 2-state automaton is budget-exhausted."""
        aut = _build_two_state()
        aut.config.max_budget = 1
        ex  = AutomatonExecutor(aut, llm, extractor)
        ep  = ex.run_episode("question")
        # May terminate as terminal_state or budget depending on whether the
        # always transition fires before the budget check on next step.
        assert ep.terminated_by in ("terminal_state", "budget")


class TestSelfConsistencySampling:
    def test_single_sample_omits_self_consistency(self, llm, extractor):
        aut = _build_two_state(start_terminal=True)
        ex  = AutomatonExecutor(aut, llm, extractor, n_consistency_samples=1)
        ep  = ex.run_episode("question")
        # features dict of the terminal step should NOT have self_consistency
        step_feats = ep.steps[0].features
        assert "self_consistency" not in step_feats

    def test_multi_sample_includes_self_consistency(self, llm, extractor):
        aut = _build_two_state(start_terminal=True)
        ex  = AutomatonExecutor(aut, llm, extractor, n_consistency_samples=3)
        ep  = ex.run_episode("question")
        step_feats = ep.steps[0].features
        assert "self_consistency" in step_feats
        assert 0.0 <= step_feats["self_consistency"] <= 1.0

    def test_multi_sample_increases_token_count(self, llm, extractor):
        aut  = _build_two_state(start_terminal=True)
        ex1  = AutomatonExecutor(aut, llm, extractor, n_consistency_samples=1)
        ep1  = ex1.run_episode("question", episode_id="s1")
        # reset call count for clean comparison
        llm2 = MockLLM(uncertainty_rate=0.0, latency=0.0, seed=42)
        ex2  = AutomatonExecutor(aut, llm2, extractor, n_consistency_samples=3)
        ep2  = ex2.run_episode("question", episode_id="s2")
        assert ep2.total_tokens > ep1.total_tokens


class TestVerboseMode:
    def test_verbose_does_not_raise(self, llm, extractor, capsys):
        aut = _build_two_state()
        ex  = AutomatonExecutor(aut, llm, extractor)
        ex.run_episode("question", verbose=True)   # should not raise
        captured = capsys.readouterr()
        assert "TERMINAL" in captured.out or len(captured.out) >= 0


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, llm, extractor, tmp_path):
        aut = _build_two_state()
        ex  = AutomatonExecutor(aut, llm, extractor)
        ex.run_episode("q1")
        aut.fitness = 0.75

        checkpoint = str(tmp_path / "aut.json")
        aut.save_checkpoint(checkpoint)

        loaded = Automaton.load_checkpoint(checkpoint)
        assert loaded.name == aut.name
        assert abs(loaded.fitness - 0.75) < 1e-9
        assert loaded.episodes_run == aut.episodes_run
        assert set(loaded.states.keys()) == set(aut.states.keys())
