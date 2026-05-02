"""
utils/api.py
────────────
LLM backend adapters.

  MockLLM   — simulates realistic, feature-sensitive responses (no API key needed)
  OpenAILLM — real OpenAI chat completions (requires OPENAI_API_KEY env var)

Factory
-------
  get_llm_api(backend="mock", **kwargs) → MockLLM | OpenAILLM
"""
from __future__ import annotations

import os
import random
import time
from threading import Lock
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Mock LLM
# ──────────────────────────────────────────────────────────────────────────────

class MockLLM:
    """
    A fast, deterministic-ish mock that reacts to prompt characteristics
    so the automaton's branching logic has meaningful signal to respond to.

    Behaviours
    ──────────
    - Prompts containing "verify" / "double-check"  → verification response
    - Prompts containing "decompose" / "step by step" → structured decomposition
    - Very long prompts (>200 words)                 → verbose hedged response
    - Simple short prompts                           → confident direct answer
    - Random chance (uncertainty_rate)               → adds uncertainty phrases
    """

    _ANSWER_POOL = {
        "capital":  ["Paris", "Berlin", "Tokyo", "London", "Washington D.C.", "Ottawa"],
        "number":   ["42", "17", "256", "3.14159", "100", "2048"],
        "science":  ["hydrogen", "photosynthesis", "gravity", "entropy", "electron"],
        "concept":  ["supervised learning", "backpropagation", "natural selection",
                     "thermodynamics", "Turing completeness"],
    }

    _UNCERTAINTY_HEDGE = [
        "I'm not entirely certain, but {ans}",
        "It's possibly {ans}, though I'd recommend verifying this.",
        "I think the answer might be {ans}, but I'm not fully confident.",
        "Perhaps {ans} — though this is uncertain.",
    ]

    _CONFIDENT = [
        "The answer is {ans}.",
        "{ans}.",
        "Based on the provided information: {ans}.",
        "Definitively: {ans}.",
    ]

    _VERIFICATION_RESPONSES = [
        "After careful review, the previous answer appears correct.",
        "I've double-checked and can confirm: the answer is correct.",
        "Upon verification, a correction is needed — the precise answer is 42.",
        "Verification complete: the response aligns with known facts.",
    ]

    _DECOMP_STEPS = [
        "Step 1: Identify the core components of the problem.",
        "Step 2: Apply the relevant principle or formula.",
        "Step 3: Compute intermediate results.",
        "Step 4: Synthesise the final answer from the components.",
        "Final Answer: Based on the systematic decomposition, the result is 42.",
    ]

    _LONG_INPUT_RESPONSES = [
        ("Given the complexity and length of this input, I will address the key points: "
         "The central question relates to the primary topic. Based on available evidence "
         "and careful reasoning, the most supported conclusion is that the answer depends "
         "on contextual factors that require careful evaluation."),
        ("This is a complex, multi-faceted question. To handle it properly: "
         "First, I note the key entities involved. Second, I identify the relevant "
         "relationships. Third, I apply domain knowledge to arrive at a reasoned answer. "
         "The conclusion is that the answer is approximately correct within this context."),
    ]

    def __init__(
        self,
        uncertainty_rate: float = 0.30,
        latency:          float = 0.02,
        seed:             Optional[int] = None,
    ):
        self.uncertainty_rate = uncertainty_rate
        self.latency          = latency
        self.call_count       = 0
        self.total_tokens     = 0
        self._lock            = Lock()
        # Use an instance-level RNG so concurrent workers never share state.
        # Module-level random.seed() is NOT thread-safe and was removed.
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    def call(
        self,
        prompt:     str,
        role:       str = "user",
        max_tokens: int = 256,
    ) -> Tuple[str, int]:
        """
        Simulate an LLM call.

        Returns
        -------
        (response_text, token_count)
        """
        time.sleep(self.latency)
        with self._lock:
            self.call_count += 1

        prompt_lower = prompt.lower()
        prompt_words = len(prompt.split())

        # ── Route by prompt content ───────────────────────────────────
        if any(kw in prompt_lower for kw in ("verify", "double-check", "is this correct", "confirm")):
            response = self.rng.choice(self._VERIFICATION_RESPONSES)

        elif any(kw in prompt_lower for kw in ("decompose", "break down", "step by step", "systematically")):
            n_steps = self.rng.randint(3, 5)
            response = " ".join(self._DECOMP_STEPS[:n_steps])

        elif prompt_words > 200:
            response = self.rng.choice(self._LONG_INPUT_RESPONSES)
            # Long inputs often produce hedged answers
            if self.rng.random() < self.uncertainty_rate + 0.15:
                response = "I'm not entirely certain about all aspects. " + response

        else:
            # Standard answer
            category = self.rng.choice(list(self._ANSWER_POOL.keys()))
            ans = self.rng.choice(self._ANSWER_POOL[category])

            is_complex = prompt_words > 60
            if is_complex and self.rng.random() < self.uncertainty_rate:
                template = self.rng.choice(self._UNCERTAINTY_HEDGE)
            else:
                template = self.rng.choice(self._CONFIDENT)
            response = template.format(ans=ans)

        # ── Truncate to token budget ──────────────────────────────────
        words = response.split()
        # rough: 1 word ≈ 1.3 tokens
        max_words = int(max_tokens / 1.3)
        if len(words) > max_words:
            response = " ".join(words[:max_words]) + "..."

        tokens = int((len(prompt.split()) + len(response.split())) * 1.3)
        with self._lock:
            self.total_tokens += tokens
        return response, tokens

    def __repr__(self) -> str:
        return (
            f"MockLLM(uncertainty_rate={self.uncertainty_rate}, "
            f"calls={self.call_count})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Real OpenAI LLM  (optional)
# ──────────────────────────────────────────────────────────────────────────────

class OpenAILLM:
    """
    Thin wrapper around the OpenAI chat-completions endpoint.
    Requires:  pip install openai  +  OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        model:   str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
    ):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package not installed — run: pip install openai"
            ) from exc

        self.model        = model
        self.client       = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        self.call_count   = 0
        self.total_tokens = 0
        self._lock        = Lock()

    def call(
        self,
        prompt:     str,
        role:       str = "user",
        max_tokens: int = 256,
    ) -> Tuple[str, int]:
        with self._lock:
            self.call_count += 1
        safe_role = role if role in ("system", "user", "assistant") else "user"
        resp = self.client.chat.completions.create(
            model      = self.model,
            messages   = [{"role": safe_role, "content": prompt}],
            max_tokens = max_tokens,
        )
        text   = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens
        with self._lock:
            self.total_tokens += tokens
        return text, tokens

    def __repr__(self) -> str:
        return f"OpenAILLM(model={self.model!r}, calls={self.call_count})"


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_llm_api(backend: str = "mock", **kwargs):
    """
    Return an LLM backend.

    backend : "mock"   → MockLLM (default, no API key needed)
              "openai" → OpenAILLM (requires OPENAI_API_KEY)

    Falls back to MockLLM automatically if OPENAI_API_KEY is absent.
    """
    if backend == "openai":
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAILLM(**kwargs)
        else:
            print("[api] OPENAI_API_KEY not set — falling back to MockLLM.")
    return MockLLM(**kwargs)
