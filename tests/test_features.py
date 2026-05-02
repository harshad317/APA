"""
tests/test_features.py
──────────────────────
Unit tests for FeatureExtractor.extract() covering:
  - always-present features
  - conditional self_consistency (omitted when no samples supplied)
  - conditional verifier_score (omitted when not provided, present when provided)
  - boundary / edge cases (empty output, maximum word counts)
"""
import pytest
from adaptive_prompt_automaton.core.features import FeatureExtractor, FeatureVector


@pytest.fixture
def extractor():
    return FeatureExtractor(long_input_threshold=10)


class TestAlwaysPresentFeatures:
    def test_returns_feature_vector(self, extractor):
        fv = extractor.extract("hello world", "this is a response")
        assert isinstance(fv, FeatureVector)

    def test_core_keys_present(self, extractor):
        fv = extractor.extract("hello world", "this is a response")
        for key in FeatureExtractor.feature_names():
            assert key in fv.features, f"Missing key: {key}"

    def test_answer_confidence_is_complement_of_uncertainty(self, extractor):
        fv = extractor.extract("what", "I'm not sure about this at all, possibly wrong")
        assert abs(fv["answer_confidence"] - (1.0 - fv["uncertainty_score"])) < 1e-9

    def test_step_number_normalised(self, extractor):
        fv0 = extractor.extract("q", "a", step=0)
        fv5 = extractor.extract("q", "a", step=5)
        assert fv0["step_number"] == 0.0
        assert fv5["step_number"] == 1.0   # clamped at 5/5 = 1.0


class TestInputLengthFeatures:
    def test_short_input_not_flagged_as_long(self, extractor):
        # extractor threshold = 10 words; input has 3 words
        fv = extractor.extract("short input here", "response")
        assert fv["is_long_input"] == 0.0

    def test_long_input_flagged(self, extractor):
        # 12 words > threshold of 10
        long_input = " ".join(["word"] * 12)
        fv = extractor.extract(long_input, "response")
        assert fv["is_long_input"] == 1.0

    def test_input_length_normalised_to_one(self, extractor):
        very_long = " ".join(["word"] * 600)
        fv = extractor.extract(very_long, "response")
        assert fv["input_length"] <= 1.0


class TestUncertainty:
    def test_no_uncertainty_phrases(self, extractor):
        fv = extractor.extract("question", "The answer is 42.")
        assert fv["uncertainty_score"] == 0.0

    def test_single_uncertainty_phrase(self, extractor):
        fv = extractor.extract("question", "I think the answer is 42.")
        assert fv["uncertainty_score"] > 0.0

    def test_many_uncertainty_phrases_clamped_to_one(self, extractor):
        hedged = "I'm not sure, uncertain, possibly, perhaps, might be, could be"
        fv = extractor.extract("question", hedged)
        assert fv["uncertainty_score"] <= 1.0


class TestSelfConsistency:
    def test_omitted_when_no_samples(self, extractor):
        """self_consistency must NOT be present when samples is None."""
        fv = extractor.extract("question", "answer")
        assert "self_consistency" not in fv.features

    def test_omitted_when_single_sample(self, extractor):
        """Only one sample = no pairwise comparison possible → omit."""
        fv = extractor.extract("question", "answer", samples=["answer"])
        assert "self_consistency" not in fv.features

    def test_present_when_multiple_samples(self, extractor):
        fv = extractor.extract(
            "question", "answer A",
            samples=["answer A", "answer B", "answer C"]
        )
        assert "self_consistency" in fv.features
        val = fv["self_consistency"]
        assert 0.0 <= val <= 1.0

    def test_identical_samples_give_consistency_one(self, extractor):
        samples = ["the cat sat on the mat"] * 3
        fv = extractor.extract("q", samples[0], samples=samples)
        assert abs(fv["self_consistency"] - 1.0) < 1e-6


class TestVerifierScore:
    def test_omitted_when_not_provided(self, extractor):
        """verifier_score must NOT be in the vector by default."""
        fv = extractor.extract("question", "answer")
        assert "verifier_score" not in fv.features

    def test_present_and_correct_when_provided(self, extractor):
        fv = extractor.extract("question", "answer", verifier_score=0.75)
        assert "verifier_score" in fv.features
        assert abs(fv["verifier_score"] - 0.75) < 1e-9

    def test_independent_from_answer_confidence(self, extractor):
        """verifier_score should differ from answer_confidence when provided."""
        fv = extractor.extract(
            "question",
            "I'm not sure — possibly the answer",
            verifier_score=0.95,
        )
        # answer_confidence is low (hedging present), verifier_score is high
        assert fv["verifier_score"] > fv["answer_confidence"]


class TestStructuredFormat:
    def test_plain_prose_not_structured(self, extractor):
        fv = extractor.extract("q", "This is a plain prose answer with no formatting.")
        assert fv["has_structured_format"] == 0.0

    def test_bullet_list_detected(self, extractor):
        fv = extractor.extract("q", "- item one\n- item two\n- item three")
        assert fv["has_structured_format"] == 1.0

    def test_numbered_list_detected(self, extractor):
        fv = extractor.extract("q", "1. First point\n2. Second point")
        assert fv["has_structured_format"] == 1.0

    def test_code_block_detected(self, extractor):
        fv = extractor.extract("q", "Here is code:\n```python\nprint('hi')\n```")
        assert fv["has_structured_format"] == 1.0


class TestOutputToInputRatio:
    def test_equal_lengths_gives_mid_ratio(self, extractor):
        inp = " ".join(["word"] * 10)
        out = " ".join(["word"] * 10)
        fv  = extractor.extract(inp, out)
        # ratio = 10/10 = 1.0; normalised = min(1.0/3.0, 1.0) ≈ 0.333
        assert abs(fv["output_to_input_ratio"] - (1.0 / 3.0)) < 0.01

    def test_ratio_clamped_at_one(self, extractor):
        inp = "short"
        out = " ".join(["word"] * 200)
        fv  = extractor.extract(inp, out)
        assert fv["output_to_input_ratio"] == 1.0
