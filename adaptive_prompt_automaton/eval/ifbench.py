"""
eval/ifbench.py
───────────────
IFBench — Instruction-Following Benchmark for the Adaptive Prompt Automaton.

Modelled after IFEval (Google, arXiv:2311.07911) with verifiable, rule-based
constraints.  Each task pairs a question with an explicit, machine-checkable
instruction constraint.  Compliance is measured by deterministic *parsers* —
no LLM judge required.

Six constraint types / parsers
────────────────────────────────
  KeywordParser    — response must include / must not include specific words
  LengthParser     — word or sentence count constraints (exact / min / max)
  FormatParser     — structural format (bullet list, numbered list, JSON, code)
  StartEndParser   — response must start or end with a specific phrase
  CaseParser       — character-case requirement (upper / lower / title)
  CompositeParser  — two constraints from different types combined (AND logic)

Public API
──────────
  make_ifbench_benchmark() → Tuple[BenchmarkSuite, BenchmarkSuite]
      Returns (train_suite, test_suite).

  ifbench_reward(episode, task) → float
      0.0 – 1.0 compliance score using the task's own parser.

  PARSERS : Dict[str, type]
      Registry mapping parser name → class, for --parser CLI selection.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .benchmarks import BenchmarkSuite, Task
from ..core.executor import Episode


# ──────────────────────────────────────────────────────────────────────────────
# Base parser
# ──────────────────────────────────────────────────────────────────────────────

class BaseParser:
    """
    A constraint checker.  Call as:  score = parser(response_text) → float 0‥1
    """
    name: str = "base"

    def check(self, response: str) -> float:
        raise NotImplementedError

    def __call__(self, response: str) -> float:
        return self.check(response.strip())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ──────────────────────────────────────────────────────────────────────────────
# 1. KeywordParser
# ──────────────────────────────────────────────────────────────────────────────

class KeywordParser(BaseParser):
    """
    Checks keyword inclusion / exclusion.

    Parameters
    ──────────
    must_include : list of str
        All of these words must appear in the response (case-insensitive).
    must_exclude : list of str
        None of these words may appear in the response (case-insensitive).
    """
    name = "keyword"

    def __init__(
        self,
        must_include: Optional[List[str]] = None,
        must_exclude: Optional[List[str]] = None,
    ):
        self.must_include = [w.lower() for w in (must_include or [])]
        self.must_exclude = [w.lower() for w in (must_exclude or [])]

    def check(self, response: str) -> float:
        lowered = response.lower()
        passed = sum(w in lowered for w in self.must_include)
        failed = sum(w in lowered for w in self.must_exclude)
        total  = len(self.must_include) + len(self.must_exclude)
        if total == 0:
            return 1.0
        return max(0.0, (passed + (len(self.must_exclude) - failed)) / total)

    def __repr__(self) -> str:
        return (f"KeywordParser(include={self.must_include}, "
                f"exclude={self.must_exclude})")


# ──────────────────────────────────────────────────────────────────────────────
# 2. LengthParser
# ──────────────────────────────────────────────────────────────────────────────

class LengthParser(BaseParser):
    """
    Checks word-count or sentence-count constraints.

    Parameters
    ──────────
    unit     : "words" | "sentences"
    operator : "exact" | "min" | "max" | "between"
    value    : int or (int, int) for "between"
    """
    name = "length"

    def __init__(
        self,
        unit:     str          = "words",
        operator: str          = "max",
        value:    Any          = 50,
    ):
        self.unit     = unit
        self.operator = operator
        self.value    = value

    def _count(self, text: str) -> int:
        if self.unit == "sentences":
            return max(1, len(re.split(r'[.!?]+', text.strip())))
        return len(text.split())

    def check(self, response: str) -> float:
        n = self._count(response)
        if self.operator == "exact":
            return 1.0 if n == self.value else max(0.0, 1.0 - abs(n - self.value) / max(self.value, 1))
        if self.operator == "min":
            return 1.0 if n >= self.value else n / self.value
        if self.operator == "max":
            return 1.0 if n <= self.value else self.value / n
        if self.operator == "between":
            lo, hi = self.value
            if lo <= n <= hi:
                return 1.0
            if n < lo:
                return n / lo
            return hi / n
        return 0.5

    def __repr__(self) -> str:
        return f"LengthParser(unit={self.unit!r}, op={self.operator!r}, val={self.value})"


# ──────────────────────────────────────────────────────────────────────────────
# 3. FormatParser
# ──────────────────────────────────────────────────────────────────────────────

class FormatParser(BaseParser):
    """
    Checks whether the response uses a specific structural format.

    Formats: "bullet_list" | "numbered_list" | "json" | "code_block" | "table"
    """
    name = "format"

    _PATTERNS: Dict[str, re.Pattern] = {
        "bullet_list":   re.compile(r'^\s*[-*•]\s+.+', re.MULTILINE),
        "numbered_list": re.compile(r'^\s*\d+[\.\)]\s+.+', re.MULTILINE),
        "json":          re.compile(r'[\{\[][\s\S]*[\}\]]'),
        "code_block":    re.compile(r'```[\s\S]+?```'),
        "table":         re.compile(r'(\|.+\|[\r\n]+){2,}'),
    }

    def __init__(self, fmt: str):
        if fmt not in self._PATTERNS:
            raise ValueError(f"Unknown format: {fmt!r}. Choose from {list(self._PATTERNS)}")
        self.fmt     = fmt
        self.pattern = self._PATTERNS[fmt]

    def check(self, response: str) -> float:
        matches = self.pattern.findall(response)
        if self.fmt in ("bullet_list", "numbered_list"):
            # Require at least 2 items
            return 1.0 if len(matches) >= 2 else (0.5 if len(matches) == 1 else 0.0)
        return 1.0 if matches else 0.0

    def __repr__(self) -> str:
        return f"FormatParser(fmt={self.fmt!r})"


# ──────────────────────────────────────────────────────────────────────────────
# 4. StartEndParser
# ──────────────────────────────────────────────────────────────────────────────

class StartEndParser(BaseParser):
    """
    Checks whether the response starts or ends with a required phrase.

    Parameters
    ──────────
    starts_with : str or None
    ends_with   : str or None
    case_sensitive : bool (default False)
    """
    name = "startend"

    def __init__(
        self,
        starts_with:    Optional[str] = None,
        ends_with:      Optional[str] = None,
        case_sensitive: bool          = False,
    ):
        self.starts_with    = starts_with
        self.ends_with      = ends_with
        self.case_sensitive = case_sensitive

    def _norm(self, s: str) -> str:
        return s if self.case_sensitive else s.lower()

    def check(self, response: str) -> float:
        checks = []
        resp_n = self._norm(response)
        if self.starts_with is not None:
            checks.append(resp_n.startswith(self._norm(self.starts_with)))
        if self.ends_with is not None:
            checks.append(resp_n.endswith(self._norm(self.ends_with)))
        if not checks:
            return 1.0
        return sum(checks) / len(checks)

    def __repr__(self) -> str:
        return (f"StartEndParser(starts_with={self.starts_with!r}, "
                f"ends_with={self.ends_with!r})")


# ──────────────────────────────────────────────────────────────────────────────
# 5. CaseParser
# ──────────────────────────────────────────────────────────────────────────────

class CaseParser(BaseParser):
    """
    Checks character-case requirement.

    mode : "upper" | "lower" | "title"
    threshold : fraction of alphabetic chars that must satisfy the case rule
    """
    name = "case"

    def __init__(self, mode: str = "upper", threshold: float = 0.80):
        if mode not in ("upper", "lower", "title"):
            raise ValueError(f"Unknown case mode: {mode!r}")
        self.mode      = mode
        self.threshold = threshold

    def check(self, response: str) -> float:
        alpha = [c for c in response if c.isalpha()]
        if not alpha:
            return 0.5
        if self.mode == "upper":
            frac = sum(c.isupper() for c in alpha) / len(alpha)
        elif self.mode == "lower":
            frac = sum(c.islower() for c in alpha) / len(alpha)
        else:  # title: check words
            words = response.split()
            if not words:
                return 0.5
            frac = sum(w[0].isupper() for w in words if w) / len(words)
        return 1.0 if frac >= self.threshold else frac / self.threshold

    def __repr__(self) -> str:
        return f"CaseParser(mode={self.mode!r}, threshold={self.threshold})"


# ──────────────────────────────────────────────────────────────────────────────
# 6. CompositeParser
# ──────────────────────────────────────────────────────────────────────────────

class CompositeParser(BaseParser):
    """
    Combines two or more parsers with AND logic.
    Score = mean of all sub-parser scores.
    """
    name = "composite"

    def __init__(self, *parsers: BaseParser):
        if len(parsers) < 2:
            raise ValueError("CompositeParser requires at least 2 sub-parsers")
        self.parsers = list(parsers)

    def check(self, response: str) -> float:
        scores = [p(response) for p in self.parsers]
        return sum(scores) / len(scores)

    def __repr__(self) -> str:
        return f"CompositeParser({', '.join(repr(p) for p in self.parsers)})"


# ──────────────────────────────────────────────────────────────────────────────
# Parser registry
# ──────────────────────────────────────────────────────────────────────────────

PARSERS: Dict[str, type] = {
    "keyword":   KeywordParser,
    "length":    LengthParser,
    "format":    FormatParser,
    "startend":  StartEndParser,
    "case":      CaseParser,
    "composite": CompositeParser,
}


# ──────────────────────────────────────────────────────────────────────────────
# IFTask  — Task subclass carrying a parser instance
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IFTask(Task):
    """
    A Task annotated with a machine-checkable instruction-following constraint.

    Extra fields
    ────────────
    parser          : BaseParser instance that grades compliance (0–1)
    constraint_type : human-readable constraint category
    constraint_desc : one-line description of the constraint (for tables)
    """
    parser:          BaseParser = field(default_factory=lambda: LengthParser())
    constraint_type: str        = "length"
    constraint_desc: str        = ""


# ──────────────────────────────────────────────────────────────────────────────
# Reward bridge
# ──────────────────────────────────────────────────────────────────────────────

def ifbench_reward(episode: Episode, task: IFTask) -> float:
    """
    Score an episode on an IFTask.

    Returns a value in [0, 1]:
      0.0  — complete non-compliance
      1.0  — full compliance

    Includes a small fluency guard: if the response is shorter than 3 words,
    compliance is capped at 0.1.
    """
    text = episode.final_output or ""
    if len(text.split()) < 3:
        return 0.1
    return float(task.parser(text))


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark builders
# ──────────────────────────────────────────────────────────────────────────────

def make_ifbench_benchmark() -> Tuple[BenchmarkSuite, BenchmarkSuite]:
    """
    Build the IFBench train/test split.

    Returns
    ───────
    (train_suite, test_suite) : both are BenchmarkSuite objects.

    The benchmark contains 48 IFTasks (8 per parser type × 6 types),
    split 24/24 into train and test.  The task distribution is:

        Parser type     Tasks   Example constraint
        ──────────────  ──────  ─────────────────────────────────────────
        keyword           8     "Include the word 'specifically'"
        length            8     "Answer in fewer than 30 words"
        format            8     "Use a numbered list with at least 3 items"
        startend          8     "Start your answer with 'In summary,'"
        case              8     "Write your answer in ALL CAPS"
        composite         8     keyword + length  /  format + startend
    """
    train = BenchmarkSuite("IFBench-Train")
    test  = BenchmarkSuite("IFBench-Test")

    all_tasks: List[IFTask] = []

    # ── 1. Keyword tasks ──────────────────────────────────────────────────────
    keyword_specs = [
        # (question, must_include, must_exclude, difficulty)
        ("Explain how photosynthesis works.",
         ["chlorophyll", "sunlight"], [],              "medium"),
        ("What is machine learning?",
         ["algorithm", "data"],       [],              "easy"),
        ("Describe the water cycle.",
         ["evaporation", "precipitation"], [],         "easy"),
        ("What causes thunder during a storm?",
         ["lightning", "sound"],      [],              "easy"),
        ("How does the immune system fight viruses?",
         ["antibody", "cell"],        ["maybe", "perhaps"], "medium"),
        ("Explain quantum entanglement briefly.",
         ["particle", "correlation"], ["not sure"],   "hard"),
        ("What is the role of mitochondria in a cell?",
         ["energy", "atp"],           [],              "medium"),
        ("How does HTTPS secure web traffic?",
         ["encryption", "certificate"], ["possibly"],  "medium"),
    ]
    for i, (q, inc, exc, diff) in enumerate(keyword_specs):
        parser      = KeywordParser(must_include=inc, must_exclude=exc)
        inc_str     = "+".join(inc) if inc else ""
        exc_str     = "¬"+"+".join(exc) if exc else ""
        cstr        = f"must include [{inc_str}]" + (f" exclude [{exc_str}]" if exc else "")
        instr       = (f"  [Constraint: your answer MUST include the "
                       f"{'word' if len(inc)==1 else 'words'} "
                       f"{', '.join(repr(w) for w in inc)}"
                       + (f" and must NOT use {', '.join(repr(w) for w in exc)}" if exc else "")
                       + ".]")
        all_tasks.append(IFTask(
            task_id         = f"if_kw_{i}",
            input_text      = q + "\n" + instr,
            expected        = "",
            category        = "ifbench",
            difficulty      = diff,
            parser          = parser,
            constraint_type = "keyword",
            constraint_desc = cstr,
        ))

    # ── 2. Length tasks ───────────────────────────────────────────────────────
    length_specs = [
        # (question, unit, operator, value, difficulty)
        ("What is the speed of light?",         "words",     "max",     20,      "easy"),
        ("Explain DNA replication.",             "words",     "between", (30,60), "medium"),
        ("Name three programming languages.",   "words",     "max",     15,      "easy"),
        ("Describe the French Revolution.",     "sentences", "between", (2, 4),  "medium"),
        ("What is the Pythagorean theorem?",    "words",     "exact",   25,      "hard"),
        ("How does a transistor work?",         "words",     "min",     20,      "medium"),
        ("What is Newton's first law?",         "sentences", "max",     2,       "easy"),
        ("Explain gradient descent.",           "words",     "between", (25,55), "medium"),
    ]
    for i, (q, unit, op, val, diff) in enumerate(length_specs):
        parser   = LengthParser(unit=unit, operator=op, value=val)
        val_str  = f"{val[0]}–{val[1]}" if isinstance(val, tuple) else str(val)
        cstr     = f"{op} {val_str} {unit}"
        instr    = f"  [Constraint: answer in {op} {val_str} {unit}.]"
        all_tasks.append(IFTask(
            task_id         = f"if_len_{i}",
            input_text      = q + "\n" + instr,
            expected        = "",
            category        = "ifbench",
            difficulty      = diff,
            parser          = parser,
            constraint_type = "length",
            constraint_desc = cstr,
        ))

    # ── 3. Format tasks ───────────────────────────────────────────────────────
    format_specs = [
        # (question, fmt, difficulty)
        ("List the planets of the solar system.",           "bullet_list",   "easy"),
        ("Name 4 benefits of exercise.",                    "bullet_list",   "easy"),
        ("List 3 causes of World War I.",                   "numbered_list", "medium"),
        ("Give 4 tips for writing clean code.",             "numbered_list", "easy"),
        ("Represent a person named Alice aged 30 as JSON.", "json",          "medium"),
        ("Show a Python hello-world example.",              "code_block",    "easy"),
        ("Compare Python and JavaScript across 3 traits.",  "table",         "hard"),
        ("List 5 machine learning algorithms.",             "numbered_list", "easy"),
    ]
    for i, (q, fmt, diff) in enumerate(format_specs):
        parser = FormatParser(fmt=fmt)
        cstr   = f"format={fmt}"
        instr  = {
            "bullet_list":   "  [Constraint: use a bullet-point list (–/*/•) with at least 2 items.]",
            "numbered_list": "  [Constraint: use a numbered list (1. / 2. …) with at least 2 items.]",
            "json":          "  [Constraint: respond with valid JSON only, no prose.]",
            "code_block":    "  [Constraint: wrap your code in a markdown code block (``` … ```).]",
            "table":         "  [Constraint: format your answer as a markdown table.]",
        }[fmt]
        all_tasks.append(IFTask(
            task_id         = f"if_fmt_{i}",
            input_text      = q + "\n" + instr,
            expected        = "",
            category        = "ifbench",
            difficulty      = diff,
            parser          = parser,
            constraint_type = "format",
            constraint_desc = cstr,
        ))

    # ── 4. StartEnd tasks ─────────────────────────────────────────────────────
    startend_specs = [
        # (question, starts_with, ends_with, difficulty)
        ("Summarise the theory of evolution.",      "In summary,",   None,                "easy"),
        ("What is blockchain?",                     "Blockchain is", None,                "easy"),
        ("Explain the big bang theory.",            "The universe",  None,                "medium"),
        ("What is the halting problem?",            None,            "undecidable.",      "medium"),
        ("Describe supervised learning.",           "Supervised",    None,                "easy"),
        ("What is entropy in information theory?", "Entropy",       "uncertainty.",      "hard"),
        ("Explain the concept of recursion.",       "Recursion is",  None,                "easy"),
        ("What is Moore's Law?",                    None,            "transistors.",      "medium"),
    ]
    for i, (q, sw, ew, diff) in enumerate(startend_specs):
        parser = StartEndParser(starts_with=sw, ends_with=ew)
        parts  = []
        if sw: parts.append(f"start with '{sw}'")
        if ew: parts.append(f"end with '{ew}'")
        cstr   = " and ".join(parts)
        instr_parts = []
        if sw: instr_parts.append(f"start your answer with the exact words '{sw}'")
        if ew: instr_parts.append(f"end your answer with the exact words '{ew}'")
        instr  = "  [Constraint: " + " and ".join(instr_parts) + ".]"
        all_tasks.append(IFTask(
            task_id         = f"if_se_{i}",
            input_text      = q + "\n" + instr,
            expected        = "",
            category        = "ifbench",
            difficulty      = diff,
            parser          = parser,
            constraint_type = "startend",
            constraint_desc = cstr,
        ))

    # ── 5. Case tasks ─────────────────────────────────────────────────────────
    case_specs = [
        # (question, mode, difficulty)
        ("What is the capital of Australia?",  "upper", "easy"),
        ("Name the tallest mountain on Earth.", "upper", "easy"),
        ("What element has symbol Fe?",         "upper", "easy"),
        ("What does CPU stand for?",            "upper", "easy"),
        ("define artificial intelligence",      "lower", "easy"),
        ("what is the boiling point of water?", "lower", "easy"),
        ("List Three Types Of Cloud Computing.", "title", "medium"),
        ("Name Four Planets With Rings.",        "title", "medium"),
    ]
    for i, (q, mode, diff) in enumerate(case_specs):
        parser = CaseParser(mode=mode, threshold=0.75)
        cstr   = f"case={mode}"
        label  = {"upper": "ALL CAPS", "lower": "all lowercase", "title": "Title Case"}[mode]
        instr  = f"  [Constraint: write your entire answer in {label}.]"
        all_tasks.append(IFTask(
            task_id         = f"if_case_{i}",
            input_text      = q + "\n" + instr,
            expected        = "",
            category        = "ifbench",
            difficulty      = diff,
            parser          = parser,
            constraint_type = "case",
            constraint_desc = cstr,
        ))

    # ── 6. Composite tasks (keyword + length or format + startend) ────────────
    composite_specs = [
        # (question, parser_a, parser_b, difficulty, cstr)
        (
            "Explain neural networks briefly.\n"
            "  [Constraint: include 'neuron' and 'layer', max 40 words.]",
            KeywordParser(must_include=["neuron", "layer"]),
            LengthParser(unit="words", operator="max", value=40),
            "medium", "keyword(neuron,layer) + length(≤40w)",
        ),
        (
            "What is climate change?\n"
            "  [Constraint: include 'carbon' and 'temperature', max 35 words.]",
            KeywordParser(must_include=["carbon", "temperature"]),
            LengthParser(unit="words", operator="max", value=35),
            "easy", "keyword(carbon,temperature) + length(≤35w)",
        ),
        (
            "List three types of databases.\n"
            "  [Constraint: use a numbered list AND start with 'There are'.]",
            FormatParser("numbered_list"),
            StartEndParser(starts_with="There are"),
            "medium", "format(numbered_list) + startend('There are')",
        ),
        (
            "List four cloud providers.\n"
            "  [Constraint: use a bullet list AND start with 'The main'.]",
            FormatParser("bullet_list"),
            StartEndParser(starts_with="The main"),
            "easy", "format(bullet_list) + startend('The main')",
        ),
        (
            "Describe REST API in one sentence.\n"
            "  [Constraint: include 'HTTP' and answer in max 20 words.]",
            KeywordParser(must_include=["http"]),
            LengthParser(unit="words", operator="max", value=20),
            "medium", "keyword(HTTP) + length(≤20w)",
        ),
        (
            "Name three sorting algorithms.\n"
            "  [Constraint: numbered list AND end with 'time complexity'.]",
            FormatParser("numbered_list"),
            StartEndParser(ends_with="time complexity"),
            "hard", "format(numbered_list) + startend(ends:'time complexity')",
        ),
        (
            "What is an API?\n"
            "  [Constraint: include 'interface' and answer in max 25 words.]",
            KeywordParser(must_include=["interface"]),
            LengthParser(unit="words", operator="max", value=25),
            "easy", "keyword(interface) + length(≤25w)",
        ),
        (
            "What is containerisation in software?\n"
            "  [Constraint: include 'docker' and max 30 words.]",
            KeywordParser(must_include=["docker"]),
            LengthParser(unit="words", operator="max", value=30),
            "medium", "keyword(docker) + length(≤30w)",
        ),
    ]
    for i, (q, pa, pb, diff, cstr) in enumerate(composite_specs):
        parser = CompositeParser(pa, pb)
        all_tasks.append(IFTask(
            task_id         = f"if_comp_{i}",
            input_text      = q,
            expected        = "",
            category        = "ifbench",
            difficulty      = diff,
            parser          = parser,
            constraint_type = "composite",
            constraint_desc = cstr,
        ))

    # ── Train / test split (alternating so each type is represented in both) ──
    for idx, task in enumerate(all_tasks):
        if idx % 2 == 0:
            train.add(task)
        else:
            test.add(task)

    return train, test


# ──────────────────────────────────────────────────────────────────────────────
# Per-parser accuracy helper
# ──────────────────────────────────────────────────────────────────────────────

def per_parser_accuracy(
    episodes:   List[Episode],
    tasks:      List[IFTask],
    threshold:  float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-parser-type compliance statistics.

    Returns
    ───────
    dict mapping parser_type → {"mean_score": float, "pass_rate": float, "n": int}
    """
    from collections import defaultdict
    buckets: Dict[str, List[float]] = defaultdict(list)

    for ep, task in zip(episodes, tasks):
        score = ifbench_reward(ep, task)
        buckets[task.constraint_type].append(score)

    result = {}
    for ptype, scores in buckets.items():
        result[ptype] = {
            "mean_score": sum(scores) / len(scores),
            "pass_rate":  sum(s >= threshold for s in scores) / len(scores),
            "n":          len(scores),
        }
    return result
