from __future__ import annotations

import random

from apa.core.automaton import Automaton, Guard, State, Transition
from apa.core.executor import AutomatonExecutor, ExecutorConfig
from apa.core.mutation import mutate_automaton


def test_guard_evaluation():
    assert Guard(feature="x", op=">", value=0.5).evaluate({"x": 0.9})
    assert Guard(feature="x", op=">=", value=1.0).evaluate({"x": 1.0})
    assert Guard(feature="x", op="<", value=2.0).evaluate({"x": 1.5})
    assert Guard(feature="x", op="<=", value=2.0).evaluate({"x": 2.0})
    assert Guard(feature="x", op="==", value=2.0).evaluate({"x": 2.0})
    assert not Guard(feature="x", op=">", value=1.0).evaluate({"x": 0.1})


def test_executor_path_and_stop():
    automaton = Automaton(
        states={
            "s0": State(state_id="s0", template="draft", terminal=False),
            "s1": State(state_id="s1", template="verify", terminal=True),
        },
        transitions=[
            Transition(source="s0", target="s1", guard=Guard(feature="quality_proxy", op=">=", value=0.2), priority=0)
        ],
        start_state="s0",
    )
    executor = AutomatonExecutor(automaton, ExecutorConfig(max_steps=3))

    def step_eval(_prompt, _ctx, step_idx):
        return (1.0 if step_idx == 0 else 0.0), {"quality_proxy": 1.0}

    trace = executor.run({}, step_eval)
    assert trace.path == ["s0", "s1"]
    assert trace.steps[0].next_state == "s1"


def test_mutation_preserves_valid_automaton():
    automaton = Automaton(
        states={
            "s0": State(state_id="s0", template="a", terminal=False),
            "s1": State(state_id="s1", template="b", terminal=True),
        },
        transitions=[
            Transition(source="s0", target="s1", guard=Guard(feature="quality_proxy", op=">=", value=0.5), priority=0)
        ],
        start_state="s0",
    )
    child = mutate_automaton(automaton, random.Random(0))
    child.validate()
    assert child.start_state == "s0"
