from __future__ import annotations

from apa.core.automaton import Automaton
from apa.core.optimizer import APAConfig, APAOptimizer


def _eval_episode(_automaton: Automaton, example, _idx: int):
    score = float(example)
    return score, {"quality_proxy": score}, ["draft", "finish"]


def test_optimizer_is_deterministic_for_seed():
    trainset = [0.1, 0.3, 0.5, 0.7]
    valset = [0.2, 0.4]

    o1 = APAOptimizer(APAConfig(seed=42, max_metric_calls=10, reflection_minibatch_size=2))
    c1 = o1.compile(trainset, valset, _eval_episode)

    o2 = APAOptimizer(APAConfig(seed=42, max_metric_calls=10, reflection_minibatch_size=2))
    c2 = o2.compile(trainset, valset, _eval_episode)

    assert c1.val_score == c2.val_score
    assert c1.train_score == c2.train_score
    assert sorted(c1.automaton.states.keys()) == sorted(c2.automaton.states.keys())
