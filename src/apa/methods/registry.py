from __future__ import annotations

from apa.methods.apa_runner import APAMethodRunner
from apa.methods.gepa_runner import GEPAMethodRunner
from apa.methods.mipro_runner import MIPROv2MethodRunner


def get_method_runner(method: str):
    key = method.lower()
    if key == "apa":
        return APAMethodRunner()
    if key == "gepa":
        return GEPAMethodRunner()
    if key == "miprov2":
        return MIPROv2MethodRunner()
    raise KeyError(f"Unknown method '{method}'. Available: apa, gepa, miprov2")


def list_methods() -> list[str]:
    return ["apa", "gepa", "miprov2"]
