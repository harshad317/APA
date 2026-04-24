from __future__ import annotations

import random


try:
    import numpy as np
except Exception:  # pragma: no cover - optional
    np = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
