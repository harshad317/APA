"""
core/features.py
────────────────
Runtime feature extraction from LLM inputs and outputs.

Features captured (all normalised to [0, 1]):
  - input_length          normalised word count of the task input
  - is_long_input         1.0 if input ≥ long_input_threshold words
  - output_length         normalised word count of the LLM response
  - uncertainty_score     proxy for expressed uncertainty in the output
  - answer_confidence     1 - uncertainty_score
  - self_consistency      Jaccard-based agreement across multiple samples
                          (ONLY present when ≥2 samples are supplied; omitted
                          otherwise to avoid the misleading hardcoded-0.8 fallback)
  - has_structured_format 1.0 if the output contains lists / code / headings
  - verifier_score        external judge score in [0,1]
                          (ONLY present when an actual score is supplied; omitted
                          when None to prevent collinearity with answer_confidence)
  - tool_success          1.0 / 0.0 / 0.5 (success / failure / unknown)
  - output_to_input_ratio relative verbosity
  - step_number           normalised execution step index
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Phrase lists
# ──────────────────────────────────────────────────────────────────────────────

_UNCERTAINTY_PHRASES: List[str] = [
    "i'm not sure", "i am not sure", "i don't know", "i do not know",
    "uncertain", "unclear", "not certain", "not entirely clear",
    "might be", "could be", "possibly", "perhaps", "approximately",
    "roughly", "seems like", "it appears", "i believe", "i think",
    "hard to say", "difficult to determine", "may be", "not confident",
]

_FORMAT_PATTERNS: Dict[str, str] = {
    "has_bullet_list":  r"[\*\-•]\s+\w",
    "has_numbered_list": r"^\s*\d+[\.\)]\s+\w",
    "has_code_block":   r"```",
    "has_equation":     r"\b\d+\s*[=\+\-\*\/]\s*\d+",
    "has_answer_label": r"\b(answer|result|solution|conclusion)\s*:",
    "has_step_label":   r"\b(step\s*\d+|step-by-step|first[,:]|second[,:])",
}


# ──────────────────────────────────────────────────────────────────────────────
# FeatureVector — thin wrapper for easy dict access
# ──────────────────────────────────────────────────────────────────────────────

class FeatureVector:
    """Typed container for runtime features."""

    def __init__(self, **kwargs: float):
        self.features: Dict[str, float] = {k: float(v) for k, v in kwargs.items()}

    # ------------------------------------------------------------------ #
    def __getitem__(self, key: str) -> float:
        return self.features.get(key, 0.0)

    def __setitem__(self, key: str, value: float) -> None:
        self.features[key] = float(value)

    def get(self, key: str, default: float = 0.0) -> float:
        return self.features.get(key, default)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.features)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v:.3f}" for k, v in self.features.items())
        return f"FeatureVector({items})"


# ──────────────────────────────────────────────────────────────────────────────
# FeatureExtractor
# ──────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Derives a FeatureVector from a (task_input, llm_output) pair plus optional
    side-channel signals (multiple samples for self-consistency, verifier
    score, tool return status).

    All features are normalised to [0, 1].
    """

    def __init__(self, long_input_threshold: int = 150):
        self.long_input_threshold = long_input_threshold

    # ------------------------------------------------------------------ #
    def extract(
        self,
        task_input: str,
        llm_output: str,
        samples: Optional[List[str]] = None,
        verifier_score: Optional[float] = None,
        tool_success: Optional[bool] = None,
        step: int = 0,
    ) -> FeatureVector:
        """
        Parameters
        ----------
        task_input      : raw task string fed to the LLM
        llm_output      : the LLM's response text
        samples         : optional list of additional sampled responses for
                          self-consistency estimation
        verifier_score  : external judge score in [0, 1], or None
        tool_success    : True / False / None
        step            : current execution step index (0-based)
        """
        input_words  = len(task_input.split())
        output_words = len(llm_output.split())
        lower_out    = llm_output.lower()

        # ── Input complexity ──────────────────────────────────────────
        input_length_norm = min(input_words / 300.0, 1.0)
        is_long_input     = 1.0 if input_words >= self.long_input_threshold else 0.0

        # ── Output uncertainty ────────────────────────────────────────
        uncertainty_count = sum(
            1 for phrase in _UNCERTAINTY_PHRASES if phrase in lower_out
        )
        uncertainty_score = min(uncertainty_count / 3.0, 1.0)
        answer_confidence = 1.0 - uncertainty_score

        # ── Format compliance ─────────────────────────────────────────
        format_hits = sum(
            1
            for pattern in _FORMAT_PATTERNS.values()
            if re.search(pattern, llm_output, re.IGNORECASE | re.MULTILINE)
        )
        has_structured_format = 1.0 if format_hits > 0 else 0.0

        # ── Self-consistency ──────────────────────────────────────────
        # Only included when ≥2 real samples are provided.  The previous
        # hardcoded fallback of 0.8 was a constant with no routing signal —
        # any guard on self_consistency was a no-op.
        consistency: Optional[float] = (
            self._self_consistency(samples)
            if samples and len(samples) > 1
            else None
        )

        # ── Verifier score ────────────────────────────────────────────
        # Only included when a real external verifier score is supplied.
        # Falling back to answer_confidence makes the two features perfectly
        # collinear, doubling signal while appearing independent.
        v_score: Optional[float] = verifier_score  # None when not provided

        # ── Tool status ───────────────────────────────────────────────
        if tool_success is True:
            tool_ok = 1.0
        elif tool_success is False:
            tool_ok = 0.0
        else:
            tool_ok = 0.5          # unknown

        # ── Relative verbosity ────────────────────────────────────────
        ratio = output_words / max(input_words, 1)
        output_to_input_ratio = min(ratio / 3.0, 1.0)

        # ── Output length ─────────────────────────────────────────────
        output_length_norm = min(output_words / 200.0, 1.0)

        # Build the feature dict, conditionally including optional features
        feat_kwargs: Dict[str, float] = dict(
            input_length          = input_length_norm,
            is_long_input         = is_long_input,
            output_length         = output_length_norm,
            uncertainty_score     = uncertainty_score,
            answer_confidence     = answer_confidence,
            has_structured_format = has_structured_format,
            tool_success          = tool_ok,
            output_to_input_ratio = output_to_input_ratio,
            step_number           = min(step / 5.0, 1.0),
        )
        if consistency is not None:
            feat_kwargs["self_consistency"] = consistency
        if v_score is not None:
            feat_kwargs["verifier_score"] = v_score

        return FeatureVector(**feat_kwargs)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _self_consistency(samples: List[str]) -> float:
        """
        Estimate consistency as the mean pairwise Jaccard overlap of word sets
        across sampled responses.  Returns 1.0 for a single sample.
        """
        if len(samples) < 2:
            return 1.0
        word_sets = [set(s.lower().split()) for s in samples]
        scores: List[float] = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                union = word_sets[i] | word_sets[j]
                inter = word_sets[i] & word_sets[j]
                scores.append(len(inter) / max(len(union), 1))
        return sum(scores) / len(scores) if scores else 1.0

    # ------------------------------------------------------------------ #
    @staticmethod
    def feature_names() -> List[str]:
        """
        Returns the names of features that are always present in the vector.
        self_consistency and verifier_score are conditional (see extract() docs)
        and are not listed here to avoid misleading callers into expecting them.
        """
        return [
            "input_length", "is_long_input", "output_length",
            "uncertainty_score", "answer_confidence",
            "has_structured_format", "tool_success",
            "output_to_input_ratio", "step_number",
        ]

    @staticmethod
    def optional_feature_names() -> List[str]:
        """Features present only when their inputs are supplied."""
        return ["self_consistency", "verifier_score"]
