"""
tests/test_benchmarks.py
────────────────────────
Unit tests for composite_reward() boundary conditions and the
path-independence of the revised reward function.
"""
import pytest
import json
from adaptive_prompt_automaton.core.executor import Episode, ExecutionStep
from adaptive_prompt_automaton.eval.benchmarks import composite_reward
from adaptive_prompt_automaton.eval.ifbench_official import _row_to_example


def make_episode(
    output:         str,
    path:           list  = None,
    terminated_by:  str   = "terminal_state",
    total_tokens:   int   = 100,
) -> Episode:
    ep = Episode(episode_id="test", task_input="test question")
    ep.final_output  = output
    ep.path          = path or ["start"]
    ep.terminated_by = terminated_by
    ep.total_tokens  = total_tokens
    return ep


class TestBaseReward:
    def test_empty_output_gives_no_base(self):
        ep = make_episode("")
        assert composite_reward(ep) < 0.5

    def test_short_output_under_threshold(self):
        ep = make_episode("yes")    # < 10 words
        score = composite_reward(ep)
        assert score < 0.5

    def test_substantive_output_gets_base(self):
        ep = make_episode("The answer is forty two based on the given information here.")
        score = composite_reward(ep)
        assert score >= 0.5


class TestUncertaintyPenalty:
    def test_two_hedge_phrases_triggers_full_penalty(self):
        ep_hedge  = make_episode("I'm not sure about this, and possibly it is wrong.")
        ep_plain  = make_episode("The answer is forty two based on evidence provided here.")
        assert composite_reward(ep_hedge) < composite_reward(ep_plain)

    def test_one_hedge_phrase_triggers_half_penalty(self):
        ep_one   = make_episode("This is possibly the correct answer to the question asked.")
        ep_plain = make_episode("The answer is forty two based on evidence provided here.")
        assert composite_reward(ep_one) < composite_reward(ep_plain)


class TestCompletenessBonus:
    def test_long_sentence_ending_response_gets_bonus(self):
        long_complete = "The capital of France is Paris, a city known for its history and culture."
        ep_good = make_episode(long_complete)
        ep_bad  = make_episode("Paris is capital")   # short, no terminal punctuation
        assert composite_reward(ep_good) > composite_reward(ep_bad)

    def test_response_without_terminal_punctuation_no_bonus(self):
        ep = make_episode(
            "The capital of France is Paris a city famous for the Eiffel Tower and museums",
            terminated_by="terminal_state",
        )
        # No terminal punctuation �� no completeness_bonus
        ep_with_punct = make_episode(
            "The capital of France is Paris, a city famous for the Eiffel Tower.",
            terminated_by="terminal_state",
        )
        assert composite_reward(ep_with_punct) >= composite_reward(ep)


class TestPathIndependence:
    def test_routing_markers_do_not_inflate_reward(self):
        """
        The old reward gave a structure_bonus for 'step 1', 'decompos', 'verified'.
        The new reward must not reward these routing markers differently from any
        other content of the same quality.
        """
        ep_marker = make_episode(
            "Step 1: identify components. Step 2: apply formula. "
            "Final answer: the result follows from decomposition."
        )
        ep_plain = make_episode(
            "The answer is forty two based on the given information here provided."
        )
        # Both should receive similar scores (no big bonus for routing markers).
        # We allow a small delta for word count / punctuation differences.
        delta = abs(composite_reward(ep_marker) - composite_reward(ep_plain))
        assert delta < 0.30, (
            f"Routing markers inflated reward by {delta:.3f} — "
            "check that structure_markers are not back in composite_reward."
        )

    def test_multi_step_bonus_is_small(self):
        """Multi-step bonus (0.05) should not dominate over output quality."""
        ep_multi  = make_episode("yes", path=["start", "verify", "terminal"])
        ep_single = make_episode(
            "The capital is Paris, well known for its culture and history.",
            path=["start", "terminal"],
        )
        assert composite_reward(ep_single) > composite_reward(ep_multi)


class TestTerminalBonus:
    def test_terminal_state_bonus_applied(self):
        ep_term  = make_episode("answer here with some extra words for base score.",
                                terminated_by="terminal_state")
        ep_budget = make_episode("answer here with some extra words for base score.",
                                 terminated_by="budget")
        assert composite_reward(ep_term) > composite_reward(ep_budget)


class TestTokenPenalty:
    def test_over_budget_reduces_score(self):
        ep_cheap = make_episode(
            "The answer is forty two based on the information.",
            total_tokens=100,
        )
        ep_expensive = make_episode(
            "The answer is forty two based on the information.",
            total_tokens=5000,
        )
        assert composite_reward(ep_cheap) > composite_reward(ep_expensive)

    def test_token_penalty_capped(self):
        ep = make_episode("x " * 10, total_tokens=1_000_000)
        assert composite_reward(ep) >= -1.0


class TestLexicalBonus:
    def test_expected_term_overlap_gives_bonus(self):
        ep_match   = make_episode("The answer is Paris, the capital of France.")
        ep_no_match = make_episode("The answer is forty two based on the data.")
        r_match    = composite_reward(ep_match,    expected="Paris capital France")
        r_no_match = composite_reward(ep_no_match, expected="Paris capital France")
        assert r_match > r_no_match

    def test_empty_expected_no_bonus(self):
        ep = make_episode("The answer is forty two based on the data here.")
        r_with    = composite_reward(ep, expected="forty two")
        r_without = composite_reward(ep, expected="")
        assert r_with >= r_without


class TestBoundaries:
    def test_reward_always_in_range(self):
        for output in ["", "a", "word " * 200, "I'm not sure possibly unclear uncertain"]:
            r = composite_reward(make_episode(output))
            assert -1.0 <= r <= 1.0, f"Out of range: {r} for output={output!r}"


class TestIFBenchOfficialLoader:
    def test_hf_if_rlvr_schema_extracts_prompt_and_constraints(self):
        row = {
            "key": "example",
            "messages": [
                {
                    "role": "user",
                    "content": "Answer the math question. Use exactly two sentences.",
                }
            ],
            "ground_truth": json.dumps([
                {
                    "instruction_id": ["length_constraints:number_sentences"],
                    "kwargs": [{"num_sentences": 2}],
                }
            ]),
            "constraint": "Use exactly two sentences.",
        }

        ex = _row_to_example(row, fallback_key=0)

        assert ex.prompt == "Answer the math question. Use exactly two sentences."
        assert ex.instruction_id_list == ["length_constraints:number_sentences"]
        assert ex.kwargs == [{"num_sentences": 2}]
        assert "Use exactly two sentences" in ex.to_apa_task_input()
