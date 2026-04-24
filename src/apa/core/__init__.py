"""APA core primitives and optimization."""

from .automaton import State, Guard, Transition, Automaton, ExecutionTrace
from .executor import AutomatonExecutor
from .optimizer import APAOptimizer, APAConfig

__all__ = [
    "State",
    "Guard",
    "Transition",
    "Automaton",
    "ExecutionTrace",
    "AutomatonExecutor",
    "APAOptimizer",
    "APAConfig",
]
